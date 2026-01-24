"""
Tests for database and MLflow backup utilities.

Tests cover:
- create_backup() - creates timestamped backups with checksums
- verify_backup() - validates backup integrity
- restore_backup() - restores from backup with validation
- rotate_backups() - removes old backups
- BackupJob - scheduled backup job integration
- CLI interface

Minimum coverage target: 85%
"""

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hrp.data.db import DatabaseManager


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def backup_test_env():
    """Create isolated test environment for backup tests."""
    # Create temporary directories for source data and backups
    temp_base = tempfile.mkdtemp()
    data_dir = Path(temp_base) / "hrp-data"
    backup_dir = Path(temp_base) / "backups"

    data_dir.mkdir(parents=True)
    backup_dir.mkdir(parents=True)

    # Create test database file
    db_path = data_dir / "hrp.duckdb"
    db_path.write_bytes(b"test database content for verification")

    # Create test MLflow directory with artifacts
    mlflow_dir = data_dir / "mlflow"
    mlflow_dir.mkdir()
    (mlflow_dir / "experiment_1").mkdir()
    (mlflow_dir / "experiment_1" / "run_abc").mkdir()
    (mlflow_dir / "experiment_1" / "run_abc" / "artifacts.json").write_text(
        '{"model": "test"}'
    )
    (mlflow_dir / "experiment_1" / "run_abc" / "model.pkl").write_bytes(
        b"binary model data"
    )

    # Reset singleton
    DatabaseManager.reset()

    yield {
        "temp_base": temp_base,
        "data_dir": data_dir,
        "backup_dir": backup_dir,
        "db_path": db_path,
        "mlflow_dir": mlflow_dir,
    }

    # Cleanup
    DatabaseManager.reset()
    shutil.rmtree(temp_base, ignore_errors=True)


@pytest.fixture
def backup_with_multiple_old(backup_test_env):
    """Create backup directory with multiple old backups for rotation tests."""
    backup_dir = backup_test_env["backup_dir"]

    # Create backups with different ages
    now = datetime.now()
    backups = []

    for days_old in [1, 10, 20, 35, 45]:
        backup_time = now - timedelta(days=days_old)
        backup_name = f"backup_{backup_time.strftime('%Y%m%d_%H%M%S')}"
        backup_path = backup_dir / backup_name
        backup_path.mkdir()

        # Create metadata with timestamp
        metadata = {
            "timestamp": backup_time.isoformat(),
            "created_at": backup_time.timestamp(),
        }
        (backup_path / "metadata.json").write_text(json.dumps(metadata))
        (backup_path / "hrp.duckdb").write_bytes(b"test db")

        backups.append({
            "path": backup_path,
            "days_old": days_old,
            "timestamp": backup_time,
        })

    backup_test_env["existing_backups"] = backups
    return backup_test_env


# =============================================================================
# Test create_backup
# =============================================================================


class TestCreateBackup:
    """Tests for create_backup() function."""

    def test_creates_timestamped_directory(self, backup_test_env):
        """create_backup should create a timestamped backup directory."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Check result structure
        assert "path" in result
        assert "timestamp" in result
        assert "size_bytes" in result
        assert "size_mb" in result

        # Check directory exists
        backup_path = Path(result["path"])
        assert backup_path.exists()
        assert backup_path.is_dir()

        # Check timestamp format in directory name
        assert backup_path.name.startswith("backup_")

    def test_copies_database_file(self, backup_test_env):
        """create_backup should copy the DuckDB database file."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        backup_path = Path(result["path"])
        backup_db = backup_path / "hrp.duckdb"

        assert backup_db.exists()
        assert backup_db.read_bytes() == backup_test_env["db_path"].read_bytes()

    def test_copies_mlflow_directory(self, backup_test_env):
        """create_backup should copy the MLflow directory."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            include_mlflow=True,
        )

        backup_path = Path(result["path"])
        backup_mlflow = backup_path / "mlflow"

        assert backup_mlflow.exists()
        assert (backup_mlflow / "experiment_1" / "run_abc" / "artifacts.json").exists()
        assert (backup_mlflow / "experiment_1" / "run_abc" / "model.pkl").exists()

    def test_exclude_mlflow(self, backup_test_env):
        """create_backup with include_mlflow=False should skip MLflow."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            include_mlflow=False,
        )

        backup_path = Path(result["path"])
        backup_mlflow = backup_path / "mlflow"

        assert not backup_mlflow.exists()

    def test_generates_checksums(self, backup_test_env):
        """create_backup should generate checksums.txt file."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        backup_path = Path(result["path"])
        checksums_file = backup_path / "checksums.txt"

        assert checksums_file.exists()
        content = checksums_file.read_text()
        assert "hrp.duckdb" in content

    def test_creates_metadata_json(self, backup_test_env):
        """create_backup should create metadata.json with backup info."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        backup_path = Path(result["path"])
        metadata_file = backup_path / "metadata.json"

        assert metadata_file.exists()
        metadata = json.loads(metadata_file.read_text())

        assert "timestamp" in metadata
        assert "size_bytes" in metadata
        assert "files" in metadata
        assert "hrp_version" in metadata

    def test_handles_missing_mlflow_directory(self, backup_test_env):
        """create_backup should handle missing MLflow directory gracefully."""
        from hrp.data.backup import create_backup

        # Remove MLflow directory
        shutil.rmtree(backup_test_env["mlflow_dir"])

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            include_mlflow=True,
        )

        assert result is not None
        assert Path(result["path"]).exists()

    def test_returns_size_information(self, backup_test_env):
        """create_backup should return accurate size information."""
        from hrp.data.backup import create_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        assert result["size_bytes"] > 0
        assert result["size_mb"] >= 0  # May be 0 for small test files


# =============================================================================
# Test verify_backup
# =============================================================================


class TestVerifyBackup:
    """Tests for verify_backup() function."""

    def test_valid_backup_returns_true(self, backup_test_env):
        """verify_backup should return True for valid backup."""
        from hrp.data.backup import create_backup, verify_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        is_valid = verify_backup(Path(result["path"]))
        assert is_valid is True

    def test_corrupted_database_returns_false(self, backup_test_env):
        """verify_backup should return False when database is corrupted."""
        from hrp.data.backup import create_backup, verify_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Corrupt the database file
        backup_path = Path(result["path"])
        db_file = backup_path / "hrp.duckdb"
        db_file.write_bytes(b"corrupted data")

        is_valid = verify_backup(backup_path)
        assert is_valid is False

    def test_missing_checksums_returns_false(self, backup_test_env):
        """verify_backup should return False when checksums.txt is missing."""
        from hrp.data.backup import create_backup, verify_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Remove checksums file
        backup_path = Path(result["path"])
        (backup_path / "checksums.txt").unlink()

        is_valid = verify_backup(backup_path)
        assert is_valid is False

    def test_missing_database_returns_false(self, backup_test_env):
        """verify_backup should return False when database file is missing."""
        from hrp.data.backup import create_backup, verify_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Remove database file
        backup_path = Path(result["path"])
        (backup_path / "hrp.duckdb").unlink()

        is_valid = verify_backup(backup_path)
        assert is_valid is False

    def test_nonexistent_backup_returns_false(self, backup_test_env):
        """verify_backup should return False for nonexistent path."""
        from hrp.data.backup import verify_backup

        fake_path = backup_test_env["backup_dir"] / "nonexistent_backup"
        is_valid = verify_backup(fake_path)
        assert is_valid is False


# =============================================================================
# Test restore_backup
# =============================================================================


class TestRestoreBackup:
    """Tests for restore_backup() function."""

    def test_restores_database_to_target(self, backup_test_env):
        """restore_backup should restore database to target directory."""
        from hrp.data.backup import create_backup, restore_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Create new target directory
        restore_dir = Path(backup_test_env["temp_base"]) / "restored"
        restore_dir.mkdir()

        success = restore_backup(
            backup_path=Path(result["path"]),
            target_dir=restore_dir,
        )

        assert success is True
        assert (restore_dir / "hrp.duckdb").exists()

    def test_restores_mlflow_artifacts(self, backup_test_env):
        """restore_backup should restore MLflow artifacts."""
        from hrp.data.backup import create_backup, restore_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            include_mlflow=True,
        )

        restore_dir = Path(backup_test_env["temp_base"]) / "restored"
        restore_dir.mkdir()

        success = restore_backup(
            backup_path=Path(result["path"]),
            target_dir=restore_dir,
        )

        assert success is True
        assert (restore_dir / "mlflow" / "experiment_1" / "run_abc" / "artifacts.json").exists()

    def test_validates_before_restore(self, backup_test_env):
        """restore_backup should validate backup before restoring."""
        from hrp.data.backup import create_backup, restore_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Corrupt the backup
        backup_path = Path(result["path"])
        (backup_path / "hrp.duckdb").write_bytes(b"corrupted")

        restore_dir = Path(backup_test_env["temp_base"]) / "restored"
        restore_dir.mkdir()

        success = restore_backup(
            backup_path=backup_path,
            target_dir=restore_dir,
        )

        assert success is False

    def test_returns_false_for_invalid_backup(self, backup_test_env):
        """restore_backup should return False for invalid backup."""
        from hrp.data.backup import restore_backup

        fake_path = backup_test_env["backup_dir"] / "nonexistent"
        restore_dir = Path(backup_test_env["temp_base"]) / "restored"
        restore_dir.mkdir()

        success = restore_backup(
            backup_path=fake_path,
            target_dir=restore_dir,
        )

        assert success is False

    def test_restore_to_different_location(self, backup_test_env):
        """restore_backup should work with different target location."""
        from hrp.data.backup import create_backup, restore_backup

        result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        # Create a completely different restore location
        alt_restore = Path(backup_test_env["temp_base"]) / "alternate" / "location"
        alt_restore.mkdir(parents=True)

        success = restore_backup(
            backup_path=Path(result["path"]),
            target_dir=alt_restore,
        )

        assert success is True
        assert (alt_restore / "hrp.duckdb").exists()


# =============================================================================
# Test rotate_backups
# =============================================================================


class TestRotateBackups:
    """Tests for rotate_backups() function."""

    def test_removes_old_backups(self, backup_with_multiple_old):
        """rotate_backups should remove backups older than keep_days."""
        from hrp.data.backup import rotate_backups

        backup_dir = backup_with_multiple_old["backup_dir"]

        # Keep last 30 days
        deleted = rotate_backups(backup_dir=backup_dir, keep_days=30)

        # Should delete 35-day and 45-day old backups
        assert deleted == 2

        # Verify remaining backups
        remaining = list(backup_dir.glob("backup_*"))
        assert len(remaining) == 3  # 1, 10, 20 days old

    def test_keeps_recent_backups(self, backup_with_multiple_old):
        """rotate_backups should keep backups within keep_days."""
        from hrp.data.backup import rotate_backups

        backup_dir = backup_with_multiple_old["backup_dir"]

        # Keep last 50 days (all backups)
        deleted = rotate_backups(backup_dir=backup_dir, keep_days=50)

        assert deleted == 0

        remaining = list(backup_dir.glob("backup_*"))
        assert len(remaining) == 5

    def test_returns_deleted_count(self, backup_with_multiple_old):
        """rotate_backups should return count of deleted backups."""
        from hrp.data.backup import rotate_backups

        backup_dir = backup_with_multiple_old["backup_dir"]

        deleted = rotate_backups(backup_dir=backup_dir, keep_days=15)

        # Should delete 20, 35, 45 day old backups
        assert deleted == 3

    def test_empty_directory(self, backup_test_env):
        """rotate_backups should handle empty backup directory."""
        from hrp.data.backup import rotate_backups

        deleted = rotate_backups(
            backup_dir=backup_test_env["backup_dir"],
            keep_days=30,
        )

        assert deleted == 0

    def test_nonexistent_directory(self, backup_test_env):
        """rotate_backups should handle nonexistent directory."""
        from hrp.data.backup import rotate_backups

        fake_dir = Path(backup_test_env["temp_base"]) / "nonexistent"

        deleted = rotate_backups(backup_dir=fake_dir, keep_days=30)

        assert deleted == 0


# =============================================================================
# Test list_backups
# =============================================================================


class TestListBackups:
    """Tests for list_backups() function."""

    def test_lists_all_backups(self, backup_with_multiple_old):
        """list_backups should return all backups with metadata."""
        from hrp.data.backup import list_backups

        backups = list_backups(backup_dir=backup_with_multiple_old["backup_dir"])

        assert len(backups) == 5
        for backup in backups:
            assert "path" in backup
            assert "timestamp" in backup

    def test_sorted_by_date(self, backup_with_multiple_old):
        """list_backups should return backups sorted newest first."""
        from hrp.data.backup import list_backups

        backups = list_backups(backup_dir=backup_with_multiple_old["backup_dir"])

        # Verify sorted newest to oldest
        timestamps = [b["timestamp"] for b in backups]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_empty_directory(self, backup_test_env):
        """list_backups should return empty list for empty directory."""
        from hrp.data.backup import list_backups

        backups = list_backups(backup_dir=backup_test_env["backup_dir"])

        assert backups == []


# =============================================================================
# Test BackupJob
# =============================================================================


class TestBackupJob:
    """Tests for BackupJob scheduled job class."""

    def test_job_creates_backup(self, backup_test_env):
        """BackupJob should create a backup on execution."""
        from hrp.data.backup import BackupJob

        job = BackupJob(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        result = job.execute()

        assert "backup_path" in result
        assert Path(result["backup_path"]).exists()

    def test_job_verifies_backup(self, backup_test_env):
        """BackupJob should verify backup after creation."""
        from hrp.data.backup import BackupJob

        job = BackupJob(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        result = job.execute()

        assert result.get("verified") is True

    def test_job_rotates_old_backups(self, backup_with_multiple_old):
        """BackupJob should rotate old backups."""
        from hrp.data.backup import BackupJob

        job = BackupJob(
            data_dir=backup_with_multiple_old["data_dir"],
            backup_dir=backup_with_multiple_old["backup_dir"],
            keep_days=30,
        )

        result = job.execute()

        assert "deleted_old" in result
        assert result["deleted_old"] >= 2  # 35 and 45 day old

    def test_job_returns_size_info(self, backup_test_env):
        """BackupJob should return backup size information."""
        from hrp.data.backup import BackupJob

        job = BackupJob(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        result = job.execute()

        assert "size_mb" in result
        assert "records_fetched" in result  # Required by base class
        assert "records_inserted" in result


# =============================================================================
# Test CLI Interface
# =============================================================================


class TestCLI:
    """Tests for backup CLI interface."""

    def test_backup_command(self, backup_test_env):
        """CLI --backup should create a backup."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', [
            'backup.py',
            '--backup',
            '--data-dir', str(backup_test_env["data_dir"]),
            '--backup-dir', str(backup_test_env["backup_dir"]),
        ]):
            # Should not raise
            result = main()
            assert result == 0

    def test_backup_command_no_mlflow(self, backup_test_env):
        """CLI --backup --no-mlflow should exclude MLflow."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', [
            'backup.py',
            '--backup',
            '--no-mlflow',
            '--data-dir', str(backup_test_env["data_dir"]),
            '--backup-dir', str(backup_test_env["backup_dir"]),
        ]):
            result = main()
            assert result == 0

    def test_verify_command(self, backup_test_env):
        """CLI --verify should verify a backup."""
        from hrp.data.backup import create_backup, main
        import sys

        # Create a backup first
        backup_result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        with patch.object(sys, 'argv', [
            'backup.py',
            '--verify', str(backup_result["path"]),
        ]):
            result = main()
            assert result == 0  # Valid backup

    def test_restore_command(self, backup_test_env):
        """CLI --restore should restore a backup."""
        from hrp.data.backup import create_backup, main
        import sys

        backup_result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
        )

        restore_dir = Path(backup_test_env["temp_base"]) / "cli_restore"
        restore_dir.mkdir()

        with patch.object(sys, 'argv', [
            'backup.py',
            '--restore', str(backup_result["path"]),
            '--target-dir', str(restore_dir),
        ]):
            result = main()
            assert result == 0
            assert (restore_dir / "hrp.duckdb").exists()

    def test_rotate_command(self, backup_with_multiple_old):
        """CLI --rotate should rotate old backups."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', [
            'backup.py',
            '--rotate',
            '--backup-dir', str(backup_with_multiple_old["backup_dir"]),
            '--keep-days', '30',
        ]):
            result = main()
            assert result == 0

    def test_list_command(self, backup_with_multiple_old):
        """CLI --list should list all backups."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', [
            'backup.py',
            '--list',
            '--backup-dir', str(backup_with_multiple_old["backup_dir"]),
        ]):
            result = main()
            assert result == 0

    def test_list_command_empty(self, backup_test_env):
        """CLI --list with no backups should show message."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', [
            'backup.py',
            '--list',
            '--backup-dir', str(backup_test_env["backup_dir"]),
        ]):
            result = main()
            assert result == 0

    def test_verify_command_invalid(self, backup_test_env):
        """CLI --verify should return 1 for invalid backup."""
        from hrp.data.backup import main
        import sys

        fake_backup = backup_test_env["backup_dir"] / "nonexistent"

        with patch.object(sys, 'argv', [
            'backup.py',
            '--verify', str(fake_backup),
        ]):
            result = main()
            assert result == 1

    def test_restore_command_invalid(self, backup_test_env):
        """CLI --restore should return 1 for invalid backup."""
        from hrp.data.backup import main
        import sys

        fake_backup = backup_test_env["backup_dir"] / "nonexistent"
        restore_dir = Path(backup_test_env["temp_base"]) / "cli_restore_fail"
        restore_dir.mkdir()

        with patch.object(sys, 'argv', [
            'backup.py',
            '--restore', str(fake_backup),
            '--target-dir', str(restore_dir),
        ]):
            result = main()
            assert result == 1

    def test_no_action_shows_help(self, backup_test_env, capsys):
        """CLI with no action should show help."""
        from hrp.data.backup import main
        import sys

        with patch.object(sys, 'argv', ['backup.py']):
            result = main()
            assert result == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for helper functions and edge cases."""

    def test_get_data_dir_from_hrp_data_dir(self, backup_test_env, monkeypatch):
        """_get_data_dir should use HRP_DATA_DIR env var."""
        from hrp.data.backup import _get_data_dir

        test_path = str(backup_test_env["data_dir"])
        monkeypatch.setenv("HRP_DATA_DIR", test_path)
        monkeypatch.delenv("HRP_DB_PATH", raising=False)

        result = _get_data_dir()
        assert str(result) == test_path

    def test_get_data_dir_from_hrp_db_path(self, backup_test_env, monkeypatch):
        """_get_data_dir should use parent of HRP_DB_PATH env var."""
        from hrp.data.backup import _get_data_dir

        test_path = str(backup_test_env["db_path"])
        monkeypatch.delenv("HRP_DATA_DIR", raising=False)
        monkeypatch.setenv("HRP_DB_PATH", test_path)

        result = _get_data_dir()
        assert result == backup_test_env["db_path"].parent

    def test_get_directory_size(self, backup_test_env):
        """_get_directory_size should calculate total size."""
        from hrp.data.backup import _get_directory_size

        size = _get_directory_size(backup_test_env["data_dir"])
        assert size > 0

    def test_get_directory_size_single_file(self, backup_test_env):
        """_get_directory_size should work for a single file."""
        from hrp.data.backup import _get_directory_size

        size = _get_directory_size(backup_test_env["db_path"])
        assert size == backup_test_env["db_path"].stat().st_size


class TestBackupIntegration:
    """Integration tests for complete backup/restore workflow."""

    def test_full_backup_restore_cycle(self, backup_test_env):
        """Full backup -> corrupt -> restore -> verify cycle."""
        from hrp.data.backup import create_backup, restore_backup, verify_backup

        # Create backup
        backup_result = create_backup(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            include_mlflow=True,
        )
        backup_path = Path(backup_result["path"])

        # Verify backup is valid
        assert verify_backup(backup_path) is True

        # "Corrupt" original data by modifying it
        original_db = backup_test_env["db_path"]
        original_content = original_db.read_bytes()
        original_db.write_bytes(b"corrupted data!")

        # Restore from backup
        restore_success = restore_backup(
            backup_path=backup_path,
            target_dir=backup_test_env["data_dir"],
        )

        assert restore_success is True

        # Verify restoration matches original
        restored_content = original_db.read_bytes()
        assert restored_content == original_content

    def test_backup_with_scheduled_job(self, backup_test_env):
        """BackupJob integration with scheduler infrastructure."""
        from hrp.data.backup import BackupJob

        job = BackupJob(
            data_dir=backup_test_env["data_dir"],
            backup_dir=backup_test_env["backup_dir"],
            job_id="daily_backup",
        )

        # Simulate scheduled job run
        result = job.execute()

        assert result is not None
        assert "backup_path" in result
        assert result.get("verified") is True
