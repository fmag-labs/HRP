"""
Database and MLflow backup utilities.

Provides automated backup with verification and restore capabilities.
Designed for scheduled execution as part of the HRP data pipeline.

Usage:
    # Programmatic
    from hrp.data.backup import create_backup, verify_backup, restore_backup

    backup_info = create_backup()
    if verify_backup(backup_info['path']):
        print("Backup verified successfully")

    # CLI
    python -m hrp.data.backup --backup
    python -m hrp.data.backup --verify /path/to/backup
    python -m hrp.data.backup --restore /path/to/backup --target-dir /path/to/target
    python -m hrp.data.backup --rotate --keep-days 30
    python -m hrp.data.backup --list
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from hrp.agents.jobs import IngestionJob

# Default paths
DEFAULT_DATA_DIR = Path.home() / "hrp-data"
DEFAULT_BACKUP_DIR = DEFAULT_DATA_DIR / "backups"


def _get_data_dir() -> Path:
    """Get the HRP data directory from environment or default."""
    data_dir = os.getenv("HRP_DATA_DIR")
    if data_dir:
        return Path(data_dir).expanduser()

    db_path = os.getenv("HRP_DB_PATH")
    if db_path:
        return Path(db_path).expanduser().parent

    return DEFAULT_DATA_DIR


def _calculate_checksum(file_path: Path) -> str:
    """
    Calculate SHA-256 checksum for a file.

    Args:
        file_path: Path to the file

    Returns:
        Hex digest of SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_directory_size(path: Path) -> int:
    """
    Calculate total size of a directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes
    """
    total = 0
    if path.is_file():
        return path.stat().st_size

    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def create_backup(
    data_dir: Path | None = None,
    backup_dir: Path | None = None,
    include_mlflow: bool = True,
) -> dict[str, Any]:
    """
    Create a backup of the database and MLflow artifacts.

    Creates a timestamped backup directory containing:
    - hrp.duckdb: Copy of the database file
    - mlflow/: Copy of MLflow artifacts (if include_mlflow=True and exists)
    - checksums.txt: SHA-256 checksums for verification
    - metadata.json: Backup metadata (timestamp, size, version)

    Args:
        data_dir: Source data directory (default: ~/hrp-data)
        backup_dir: Destination backup directory (default: ~/hrp-data/backups)
        include_mlflow: Whether to include MLflow artifacts (default: True)

    Returns:
        Dictionary with backup metadata:
        - path: Path to the backup directory
        - timestamp: ISO format timestamp
        - size_bytes: Total backup size in bytes
        - size_mb: Total backup size in megabytes
        - files: List of backed up files
    """
    # Resolve paths
    data_dir = Path(data_dir) if data_dir else _get_data_dir()
    backup_dir = Path(backup_dir) if backup_dir else DEFAULT_BACKUP_DIR

    # Create backup directory if needed
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped backup folder
    timestamp = datetime.now()
    backup_name = f"backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    backup_path = backup_dir / backup_name
    backup_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating backup at {backup_path}")

    files_backed_up = []
    checksums = {}

    # Copy database file
    db_source = data_dir / "hrp.duckdb"
    if db_source.exists():
        db_dest = backup_path / "hrp.duckdb"
        shutil.copy2(db_source, db_dest)
        checksums["hrp.duckdb"] = _calculate_checksum(db_dest)
        files_backed_up.append("hrp.duckdb")
        logger.debug(f"Backed up database: {db_source}")
    else:
        logger.warning(f"Database file not found: {db_source}")

    # Copy MLflow directory
    if include_mlflow:
        mlflow_source = data_dir / "mlflow"
        if mlflow_source.exists() and mlflow_source.is_dir():
            mlflow_dest = backup_path / "mlflow"
            shutil.copytree(mlflow_source, mlflow_dest)

            # Calculate checksums for MLflow files
            for file_path in mlflow_dest.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(backup_path)
                    checksums[str(rel_path)] = _calculate_checksum(file_path)
                    files_backed_up.append(str(rel_path))

            logger.debug(f"Backed up MLflow: {mlflow_source}")
        else:
            logger.info("MLflow directory not found, skipping")

    # Write checksums file
    checksums_file = backup_path / "checksums.txt"
    with open(checksums_file, "w") as f:
        for filename, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {filename}\n")

    # Calculate total size
    total_size = _get_directory_size(backup_path)

    # Write metadata
    metadata = {
        "timestamp": timestamp.isoformat(),
        "created_at": timestamp.timestamp(),
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "files": files_backed_up,
        "include_mlflow": include_mlflow,
        "hrp_version": "0.1.0",  # TODO: Get from package version
        "source_dir": str(data_dir),
    }

    metadata_file = backup_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"Backup created: {backup_path} "
        f"({len(files_backed_up)} files, {metadata['size_mb']} MB)"
    )

    return {
        "path": str(backup_path),
        "timestamp": timestamp.isoformat(),
        "size_bytes": total_size,
        "size_mb": metadata["size_mb"],
        "files": files_backed_up,
    }


def verify_backup(backup_path: Path) -> bool:
    """
    Verify backup integrity using checksums.

    Checks that:
    1. The backup directory exists
    2. checksums.txt file exists
    3. All files listed in checksums.txt exist
    4. All checksums match

    Args:
        backup_path: Path to the backup directory

    Returns:
        True if backup is valid, False otherwise
    """
    backup_path = Path(backup_path)

    # Check directory exists
    if not backup_path.exists() or not backup_path.is_dir():
        logger.error(f"Backup directory does not exist: {backup_path}")
        return False

    # Check checksums file exists
    checksums_file = backup_path / "checksums.txt"
    if not checksums_file.exists():
        logger.error(f"Checksums file missing: {checksums_file}")
        return False

    # Parse checksums
    expected_checksums = {}
    with open(checksums_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("  ", 1)
                if len(parts) == 2:
                    checksum, filename = parts
                    expected_checksums[filename] = checksum

    # Verify each file
    for filename, expected_checksum in expected_checksums.items():
        file_path = backup_path / filename

        if not file_path.exists():
            logger.error(f"File missing from backup: {filename}")
            return False

        actual_checksum = _calculate_checksum(file_path)
        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {filename}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
            return False

    logger.info(f"Backup verified successfully: {backup_path}")
    return True


def restore_backup(
    backup_path: Path,
    target_dir: Path | None = None,
) -> bool:
    """
    Restore database and MLflow from backup.

    Validates the backup before restoring to ensure integrity.

    Args:
        backup_path: Path to the backup directory
        target_dir: Target directory to restore to (default: original location)

    Returns:
        True if restore successful, False otherwise
    """
    backup_path = Path(backup_path)
    target_dir = Path(target_dir) if target_dir else _get_data_dir()

    # Validate backup first
    if not verify_backup(backup_path):
        logger.error("Backup verification failed, aborting restore")
        return False

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Restoring backup from {backup_path} to {target_dir}")

    try:
        # Restore database
        db_backup = backup_path / "hrp.duckdb"
        if db_backup.exists():
            db_target = target_dir / "hrp.duckdb"
            shutil.copy2(db_backup, db_target)
            logger.debug(f"Restored database to {db_target}")

        # Restore MLflow
        mlflow_backup = backup_path / "mlflow"
        if mlflow_backup.exists():
            mlflow_target = target_dir / "mlflow"
            if mlflow_target.exists():
                shutil.rmtree(mlflow_target)
            shutil.copytree(mlflow_backup, mlflow_target)
            logger.debug(f"Restored MLflow to {mlflow_target}")

        logger.info(f"Restore completed successfully to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"Restore failed: {e}")
        return False


def rotate_backups(
    backup_dir: Path | None = None,
    keep_days: int = 30,
) -> int:
    """
    Remove backups older than keep_days.

    Identifies backup directories by their naming pattern (backup_YYYYMMDD_HHMMSS)
    or metadata.json timestamp.

    Args:
        backup_dir: Directory containing backups (default: ~/hrp-data/backups)
        keep_days: Number of days of backups to keep (default: 30)

    Returns:
        Number of backups deleted
    """
    backup_dir = Path(backup_dir) if backup_dir else DEFAULT_BACKUP_DIR

    if not backup_dir.exists():
        logger.info(f"Backup directory does not exist: {backup_dir}")
        return 0

    cutoff_date = datetime.now() - timedelta(days=keep_days)
    deleted_count = 0

    for item in backup_dir.iterdir():
        if not item.is_dir() or not item.name.startswith("backup_"):
            continue

        # Try to get timestamp from metadata
        metadata_file = item / "metadata.json"
        backup_time = None

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                if "timestamp" in metadata:
                    backup_time = datetime.fromisoformat(metadata["timestamp"])
                elif "created_at" in metadata:
                    backup_time = datetime.fromtimestamp(metadata["created_at"])
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.debug(f"Could not parse metadata for {item}: {e}")

        # Fallback: parse from directory name
        if backup_time is None:
            try:
                # Extract timestamp from backup_YYYYMMDD_HHMMSS
                time_str = item.name.replace("backup_", "")
                backup_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            except ValueError:
                logger.warning(f"Could not determine age of backup: {item}")
                continue

        # Delete if older than cutoff
        if backup_time < cutoff_date:
            try:
                shutil.rmtree(item)
                deleted_count += 1
                logger.info(f"Deleted old backup: {item} (created {backup_time})")
            except Exception as e:
                logger.error(f"Failed to delete backup {item}: {e}")

    logger.info(f"Rotation complete: deleted {deleted_count} backups older than {keep_days} days")
    return deleted_count


def list_backups(
    backup_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """
    List all available backups with metadata.

    Args:
        backup_dir: Directory containing backups (default: ~/hrp-data/backups)

    Returns:
        List of backup info dictionaries, sorted newest first
    """
    backup_dir = Path(backup_dir) if backup_dir else DEFAULT_BACKUP_DIR

    if not backup_dir.exists():
        return []

    backups = []

    for item in backup_dir.iterdir():
        if not item.is_dir() or not item.name.startswith("backup_"):
            continue

        backup_info = {
            "path": str(item),
            "name": item.name,
        }

        # Try to load metadata
        metadata_file = item / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                backup_info.update({
                    "timestamp": metadata.get("timestamp"),
                    "size_mb": metadata.get("size_mb"),
                    "files_count": len(metadata.get("files", [])),
                })
            except (json.JSONDecodeError, ValueError):
                pass

        # Fallback: parse timestamp from name
        if "timestamp" not in backup_info:
            try:
                time_str = item.name.replace("backup_", "")
                backup_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                backup_info["timestamp"] = backup_time.isoformat()
            except ValueError:
                backup_info["timestamp"] = None

        backups.append(backup_info)

    # Sort by timestamp, newest first
    backups.sort(key=lambda x: x.get("timestamp") or "", reverse=True)

    return backups


class BackupJob(IngestionJob):
    """
    Scheduled job for weekly database and MLflow backups.

    Creates a backup, verifies integrity, and rotates old backups.
    Integrates with the HRP job scheduler infrastructure.
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        backup_dir: Path | None = None,
        include_mlflow: bool = True,
        keep_days: int = 30,
        job_id: str = "weekly_backup",
        max_retries: int = 2,
        retry_backoff: float = 60.0,
        dependencies: list[str] | None = None,
    ):
        """
        Initialize backup job.

        Args:
            data_dir: Source data directory (default: ~/hrp-data)
            backup_dir: Backup destination directory (default: ~/hrp-data/backups)
            include_mlflow: Whether to backup MLflow artifacts (default: True)
            keep_days: Number of days of backups to retain (default: 30)
            job_id: Unique identifier for this job
            max_retries: Maximum retry attempts on failure
            retry_backoff: Seconds between retries
            dependencies: Job IDs that must complete first
        """
        super().__init__(job_id, max_retries, retry_backoff, dependencies)
        self.data_dir = Path(data_dir) if data_dir else None
        self.backup_dir = Path(backup_dir) if backup_dir else None
        self.include_mlflow = include_mlflow
        self.keep_days = keep_days

    def execute(self) -> dict[str, Any]:
        """
        Execute the backup job.

        Creates a backup, verifies it, and rotates old backups.

        Returns:
            Dictionary with backup results:
            - backup_path: Path to the new backup
            - size_mb: Backup size in megabytes
            - verified: Whether backup passed verification
            - deleted_old: Number of old backups removed
            - records_fetched: Files backed up (for base class compatibility)
            - records_inserted: Same as records_fetched
        """
        logger.info("Starting backup job")

        # Create backup
        backup_info = create_backup(
            data_dir=self.data_dir,
            backup_dir=self.backup_dir,
            include_mlflow=self.include_mlflow,
        )

        backup_path = Path(backup_info["path"])

        # Verify backup
        verified = verify_backup(backup_path)
        if not verified:
            raise RuntimeError(f"Backup verification failed: {backup_path}")

        # Rotate old backups
        deleted = rotate_backups(
            backup_dir=self.backup_dir,
            keep_days=self.keep_days,
        )

        files_count = len(backup_info.get("files", []))

        return {
            "backup_path": str(backup_path),
            "size_mb": backup_info["size_mb"],
            "verified": verified,
            "deleted_old": deleted,
            "records_fetched": files_count,
            "records_inserted": files_count,
        }


def main() -> int:
    """
    CLI entry point for backup operations.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="HRP Backup Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create backup:
    python -m hrp.data.backup --backup

  Verify backup:
    python -m hrp.data.backup --verify /path/to/backup

  Restore backup:
    python -m hrp.data.backup --restore /path/to/backup --target-dir /path/to/target

  Rotate old backups:
    python -m hrp.data.backup --rotate --keep-days 30

  List backups:
    python -m hrp.data.backup --list
""",
    )

    # Action flags
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a new backup",
    )
    parser.add_argument(
        "--restore",
        type=str,
        metavar="PATH",
        help="Restore from backup at PATH",
    )
    parser.add_argument(
        "--verify",
        type=str,
        metavar="PATH",
        help="Verify backup integrity at PATH",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate old backups",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backups",
    )

    # Options
    parser.add_argument(
        "--data-dir",
        type=str,
        help=f"Source data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        help=f"Backup directory (default: {DEFAULT_BACKUP_DIR})",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        help="Target directory for restore",
    )
    parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Days of backups to keep for rotation (default: 30)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Exclude MLflow artifacts from backup",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output except errors",
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")

    # Parse paths
    data_dir = Path(args.data_dir) if args.data_dir else None
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    target_dir = Path(args.target_dir) if args.target_dir else None

    try:
        if args.backup:
            result = create_backup(
                data_dir=data_dir,
                backup_dir=backup_dir,
                include_mlflow=not args.no_mlflow,
            )
            print(f"Backup created: {result['path']}")
            print(f"Size: {result['size_mb']} MB")
            print(f"Files: {len(result['files'])}")
            return 0

        elif args.verify:
            is_valid = verify_backup(Path(args.verify))
            if is_valid:
                print(f"Backup is valid: {args.verify}")
                return 0
            else:
                print(f"Backup verification FAILED: {args.verify}")
                return 1

        elif args.restore:
            success = restore_backup(
                backup_path=Path(args.restore),
                target_dir=target_dir,
            )
            if success:
                print(f"Restore completed: {target_dir or _get_data_dir()}")
                return 0
            else:
                print("Restore FAILED")
                return 1

        elif args.rotate:
            deleted = rotate_backups(
                backup_dir=backup_dir,
                keep_days=args.keep_days,
            )
            print(f"Deleted {deleted} old backups")
            return 0

        elif args.list:
            backups = list_backups(backup_dir=backup_dir)
            if not backups:
                print("No backups found")
            else:
                print(f"Found {len(backups)} backups:\n")
                for b in backups:
                    size = f"{b.get('size_mb', '?')} MB" if b.get('size_mb') else "unknown size"
                    print(f"  {b['name']}: {size}")
            return 0

        else:
            parser.print_help()
            return 0

    except Exception as e:
        logger.error(f"Backup operation failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
