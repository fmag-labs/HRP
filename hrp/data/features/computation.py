"""
Feature computation engine for HRP.

Computes features at specific versions to ensure reproducibility.
"""

from datetime import date
from typing import Any, Callable

import pandas as pd
from loguru import logger

from hrp.data.db import get_db
from hrp.data.features.registry import FeatureRegistry


# Feature computation functions
def compute_momentum_20d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 20-day momentum (trailing return).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with momentum values
    """
    # Extract close prices
    close = prices["close"].unstack(level="symbol")

    # Calculate 20-day return
    momentum = close.pct_change(20)

    # Stack back to multi-index format
    result = momentum.stack(level="symbol", future_stack=True)

    return result.to_frame(name="momentum_20d")


def compute_volatility_60d(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 60-day volatility (annualized standard deviation of returns).

    Args:
        prices: Price DataFrame with MultiIndex (date, symbol) and 'close' column

    Returns:
        DataFrame with volatility values
    """
    # Extract close prices
    close = prices["close"].unstack(level="symbol")

    # Calculate daily returns
    returns = close.pct_change()

    # Calculate 60-day rolling volatility (annualized)
    volatility = returns.rolling(window=60).std() * (252**0.5)

    # Stack back to multi-index format
    result = volatility.stack(level="symbol", future_stack=True)

    return result.to_frame(name="volatility_60d")


# Registry of feature computation functions
FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "momentum_20d": compute_momentum_20d,
    "volatility_60d": compute_volatility_60d,
}


class FeatureComputer:
    """
    Computes features for symbols at specific versions.

    The computer loads feature definitions from the registry and
    computes features based on price data. Supports computing
    features at specific versions for reproducibility.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize the feature computer.

        Args:
            db_path: Optional path to database (defaults to standard location)
        """
        self.db = get_db(db_path)
        self.registry = FeatureRegistry(db_path)
        logger.debug("Feature computer initialized")

    def compute_features(
        self,
        symbols: list[str],
        dates: list[date] | pd.DatetimeIndex,
        feature_names: list[str],
        version: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute features for given symbols and dates.

        Args:
            symbols: List of stock symbols
            dates: List of dates or DatetimeIndex
            feature_names: List of feature names to compute
            version: Optional version string. If None, uses latest active version.

        Returns:
            DataFrame with MultiIndex (symbol, date) and columns for each feature

        Raises:
            ValueError: If feature definition not found or invalid data
        """
        # Convert dates to list if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Validate all features exist in registry
        feature_versions = {}
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, version)
            if not feature_def:
                version_str = f" version {version}" if version else ""
                raise ValueError(f"Feature '{feature_name}'{version_str} not found in registry")
            feature_versions[feature_name] = feature_def["version"]

        logger.info(
            f"Computing {len(feature_names)} features for {len(symbols)} symbols "
            f"across {len(dates_list)} dates (versions: {feature_versions})"
        )

        # Load price data
        prices = self._load_price_data(symbols, min(dates_list), max(dates_list))

        if prices.empty:
            raise ValueError(f"No price data found for {symbols} from {min(dates_list)} to {max(dates_list)}")

        # Compute each feature
        results = []
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, feature_versions[feature_name])
            feature_data = self._compute_feature(prices, feature_name, feature_def)
            results.append(feature_data)

        # Combine all features
        if results:
            result_df = pd.concat(results, axis=1)
            # Filter to requested dates
            date_index = pd.to_datetime(dates_list)
            result_df = result_df.reindex(result_df.index.intersection(date_index))
            return result_df
        else:
            # Return empty DataFrame with correct structure
            index = pd.MultiIndex.from_product(
                [symbols, dates_list],
                names=["symbol", "date"]
            )
            return pd.DataFrame(index=index)

    def _load_price_data(
        self,
        symbols: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """
        Load price data from database.

        Args:
            symbols: List of stock symbols
            start: Start date
            end: End date

        Returns:
            DataFrame with MultiIndex (date, symbol) and price columns
        """
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        query = f"""
            SELECT symbol, date, open, high, low, close, adj_close, volume
            FROM prices
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
            ORDER BY date, symbol
        """

        df = self.db.fetchdf(query, (start, end))

        if df.empty:
            logger.warning(f"No price data found for {symbols} from {start} to {end}")
            return pd.DataFrame()

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Set multi-index
        df = df.set_index(["date", "symbol"])

        return df

    def _compute_feature(
        self,
        prices: pd.DataFrame,
        feature_name: str,
        feature_def: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Compute a single feature from price data.

        Args:
            prices: Price DataFrame with MultiIndex (date, symbol)
            feature_name: Name of the feature
            feature_def: Feature definition from registry

        Returns:
            DataFrame with feature values, indexed by (date, symbol)
        """
        logger.debug(f"Computing feature: {feature_name} ({feature_def['version']})")

        # Get the computation function
        compute_fn = FEATURE_FUNCTIONS.get(feature_name)

        if compute_fn is None:
            logger.warning(
                f"No computation function found for {feature_name}, returning NaN"
            )
            result = pd.DataFrame(
                index=prices.index,
                columns=[feature_name],
                dtype=float,
            )
            result[feature_name] = float("nan")
            return result

        # Compute the feature
        try:
            result = compute_fn(prices)
            # Ensure result is a DataFrame with the feature name as column
            if isinstance(result, pd.Series):
                result = result.to_frame(name=feature_name)
            elif isinstance(result, pd.DataFrame) and feature_name not in result.columns:
                # Rename first column to feature_name if needed
                result.columns = [feature_name]

            logger.debug(f"Feature {feature_name} computed: {len(result)} rows")
            return result

        except Exception as e:
            logger.error(f"Error computing {feature_name}: {e}")
            raise

    def get_stored_features(
        self,
        symbols: list[str],
        dates: list[date] | pd.DatetimeIndex,
        feature_names: list[str],
        version: str | None = None,
    ) -> pd.DataFrame:
        """
        Retrieve pre-computed features from the database.

        Args:
            symbols: List of stock symbols
            dates: List of dates or DatetimeIndex
            feature_names: List of feature names to retrieve
            version: Optional version string. If None, uses latest version.

        Returns:
            DataFrame with MultiIndex (date, symbol) and columns for each feature

        Raises:
            ValueError: If feature definition not found
        """
        # Convert dates to list if needed
        if isinstance(dates, pd.DatetimeIndex):
            dates_list = dates.tolist()
        else:
            dates_list = list(dates)

        # Resolve versions
        feature_versions = {}
        for feature_name in feature_names:
            feature_def = self.registry.get(feature_name, version)
            if not feature_def:
                version_str = f" version {version}" if version else ""
                raise ValueError(f"Feature '{feature_name}'{version_str} not found in registry")
            feature_versions[feature_name] = feature_def["version"]

        # Build query
        symbols_str = ",".join(f"'{s}'" for s in symbols)
        features_str = ",".join(f"'{f}'" for f in feature_names)

        query = f"""
            SELECT symbol, date, feature_name, value, version
            FROM features
            WHERE symbol IN ({symbols_str})
              AND date >= ?
              AND date <= ?
              AND feature_name IN ({features_str})
            ORDER BY date, symbol, feature_name
        """

        df = self.db.fetchdf(query, (min(dates_list), max(dates_list)))

        if df.empty:
            logger.warning(
                f"No stored features found for {feature_names} "
                f"(symbols: {symbols}, dates: {min(dates_list)} to {max(dates_list)})"
            )
            # Return empty DataFrame with correct structure
            index = pd.MultiIndex.from_product(
                [pd.to_datetime(dates_list), symbols],
                names=["date", "symbol"]
            )
            return pd.DataFrame(index=index, columns=feature_names)

        # Filter by version
        version_filter = df.apply(
            lambda row: row["version"] == feature_versions.get(row["feature_name"]),
            axis=1,
        )
        df = df[version_filter]

        # Convert date to datetime
        df["date"] = pd.to_datetime(df["date"])

        # Pivot to get features as columns
        result = df.pivot_table(
            index=["date", "symbol"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        )

        # Ensure all requested features are present
        for feature_name in feature_names:
            if feature_name not in result.columns:
                result[feature_name] = float("nan")

        return result[feature_names]


def register_default_features(db_path: str | None = None) -> None:
    """
    Register default example features in the registry.

    This registers momentum_20d and volatility_60d features.

    Args:
        db_path: Optional path to database (defaults to standard location)
    """
    from hrp.data.features.registry import FeatureRegistry

    registry = FeatureRegistry(db_path)

    # Define features to register
    features = [
        {
            "feature_name": "momentum_20d",
            "version": "v1",
            "computation_fn": compute_momentum_20d,
            "description": "20-day momentum (trailing return). Calculated as pct_change(20).",
        },
        {
            "feature_name": "volatility_60d",
            "version": "v1",
            "computation_fn": compute_volatility_60d,
            "description": "60-day annualized volatility. Rolling std of returns * sqrt(252).",
        },
    ]

    # Register each feature (skip if already exists)
    for feature in features:
        try:
            existing = registry.get(feature["feature_name"], feature["version"])
            if existing:
                logger.debug(
                    f"Feature {feature['feature_name']} ({feature['version']}) already registered"
                )
                continue

            registry.register_feature(
                feature_name=feature["feature_name"],
                computation_fn=feature["computation_fn"],
                version=feature["version"],
                description=feature["description"],
                is_active=True,
            )
            logger.info(f"Registered feature: {feature['feature_name']} ({feature['version']})")

        except Exception as e:
            # Feature might already exist, that's okay
            logger.debug(f"Could not register {feature['feature_name']}: {e}")
