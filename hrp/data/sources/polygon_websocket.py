"""
Polygon.io WebSocket client for real-time market data streaming.

Handles connection lifecycle, automatic reconnection, message dispatch,
and heartbeat monitoring for intraday bar streaming.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from loguru import logger
from polygon import WebSocketClient
from polygon.websocket.models import Market

from hrp.utils.config import get_config


@dataclass
class WebSocketConfig:
    """Configuration for Polygon WebSocket client."""

    api_key: str
    market: str = "stocks"  # "stocks", "crypto", "forex"
    max_reconnect_delay: int = 60  # Max seconds between reconnect attempts
    heartbeat_timeout: int = 30  # Seconds without messages before warning
    queue_max_size: int = 10000  # Max messages in queue before blocking


class PolygonWebSocketClient:
    """
    Thread-safe WebSocket client for Polygon.io real-time data.

    Features:
    - Automatic reconnection with exponential backoff
    - Heartbeat monitoring during market hours
    - Thread-safe message queue for cross-thread communication
    - Graceful shutdown with buffer flush
    """

    def __init__(self, config: WebSocketConfig | None = None):
        """
        Initialize WebSocket client.

        Args:
            config: WebSocket configuration. If None, loads from app config.
        """
        if config is None:
            app_config = get_config()
            config = WebSocketConfig(
                api_key=app_config.api.polygon_api_key,
            )

        if not config.api_key:
            raise ValueError(
                "POLYGON_API_KEY not found. Set it in .env or pass to constructor."
            )

        self.config = config
        self._client: WebSocketClient | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._message_queue: deque = deque(maxlen=config.queue_max_size)
        self._queue_lock = threading.Lock()
        self._callbacks: list[Callable] = []
        self._last_message_time: float = 0
        self._reconnect_count: int = 0
        self._is_connected: bool = False
        self._subscriptions: dict[str, list[str]] = {}  # {channel: [symbols]}
        self._messages_dropped: int = 0
        self._drop_warning_threshold: int = int(config.queue_max_size * 0.9)

        logger.info("PolygonWebSocketClient initialized")

    def register_callback(self, callback: Callable[[list[dict]], None]) -> None:
        """
        Register a callback for incoming messages.

        Args:
            callback: Function that accepts a list of message dicts
        """
        self._callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")

    def start(self, symbols: list[str], channels: list[str] = None) -> None:
        """
        Start WebSocket connection and subscribe to channels.

        Args:
            symbols: List of ticker symbols (e.g., ['AAPL', 'MSFT'])
            channels: List of channels (e.g., ['AM', 'T']). Defaults to ['AM'].

        Note: Runs in a daemon thread, so won't block the main thread.
        """
        if self._thread and self._thread.is_alive():
            logger.warning("WebSocket client already running")
            return

        if channels is None:
            channels = ["AM"]  # Minute aggregates by default

        self._stop_event.clear()
        self._subscriptions = {ch: symbols for ch in channels}

        # Start WebSocket in daemon thread
        self._thread = threading.Thread(
            target=self._run_websocket,
            args=(symbols, channels),
            daemon=True,
            name="PolygonWebSocket",
        )
        self._thread.start()
        logger.info(
            f"WebSocket client started: {len(symbols)} symbols, channels: {channels}"
        )

    def stop(self) -> None:
        """
        Gracefully stop WebSocket connection.

        Flushes message queue and disconnects cleanly.
        """
        logger.info("Stopping WebSocket client...")
        self._stop_event.set()

        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket client: {e}")

        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("WebSocket thread did not terminate in time")

        self._is_connected = False
        logger.info(
            f"WebSocket client stopped. Reconnects: {self._reconnect_count}, "
            f"Messages in queue: {len(self._message_queue)}"
        )

    def is_connected(self) -> bool:
        """Check if WebSocket is currently connected."""
        return self._is_connected

    def get_stats(self) -> dict:
        """
        Get connection statistics.

        Returns:
            Dict with reconnect_count, queue_size, last_message_time
        """
        return {
            "reconnect_count": self._reconnect_count,
            "queue_size": len(self._message_queue),
            "messages_dropped": self._messages_dropped,
            "last_message_time": datetime.fromtimestamp(self._last_message_time)
            if self._last_message_time
            else None,
            "is_connected": self._is_connected,
        }

    def _run_websocket(self, symbols: list[str], channels: list[str]) -> None:
        """
        Run WebSocket connection with automatic reconnection.

        Args:
            symbols: Symbols to subscribe to
            channels: Channels to subscribe to (e.g., ['AM', 'T'])
        """
        delay = 5  # Initial reconnect delay in seconds

        while not self._stop_event.is_set():
            try:
                self._connect_and_run(symbols, channels)
                # If we get here, connection closed normally
                if not self._stop_event.is_set():
                    # Unexpected disconnect - reconnect
                    self._reconnect_count += 1
                    logger.warning(
                        f"WebSocket disconnected unexpectedly. "
                        f"Reconnecting in {delay}s... (attempt #{self._reconnect_count})"
                    )
                    time.sleep(delay)
                    # Exponential backoff: 5s → 10s → 20s → 40s → 60s (max)
                    delay = min(delay * 2, self.config.max_reconnect_delay)
            except Exception as e:
                self._is_connected = False
                if not self._stop_event.is_set():
                    logger.error(
                        f"WebSocket error: {e}. Reconnecting in {delay}s...",
                        exc_info=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, self.config.max_reconnect_delay)

        logger.info("WebSocket loop terminated")

    def _connect_and_run(self, symbols: list[str], channels: list[str]) -> None:
        """
        Connect to WebSocket and run message loop.

        Args:
            symbols: Symbols to subscribe to
            channels: Channels to subscribe to
        """
        # Convert market string to Market enum
        market_map = {
            "stocks": Market.Stocks,
            "crypto": Market.Crypto,
            "forex": Market.Forex,
        }
        market = market_map.get(self.config.market.lower(), Market.Stocks)

        # Create WebSocket client
        self._client = WebSocketClient(
            api_key=self.config.api_key,
            market=market,
            feed="delayed",  # or "realtime" based on subscription tier
        )

        # Build subscription strings (e.g., "AM.AAPL", "AM.MSFT")
        subscriptions = []
        for channel in channels:
            for symbol in symbols:
                subscriptions.append(f"{channel}.{symbol}")

        logger.info(f"Subscribing to {len(subscriptions)} streams: {subscriptions[:5]}...")

        # Subscribe and run
        self._is_connected = True
        self._last_message_time = time.time()

        try:
            # The run() method blocks until connection closes
            self._client.run(
                self._handle_message,
                subscriptions=subscriptions,
            )
        finally:
            self._is_connected = False

    def _handle_message(self, msgs: list[dict]) -> None:
        """
        Handle incoming WebSocket messages.

        Args:
            msgs: List of message dicts from Polygon
        """
        if not msgs:
            return

        self._last_message_time = time.time()

        # Add to thread-safe queue, track dropped messages
        with self._queue_lock:
            for msg in msgs:
                if len(self._message_queue) >= self._message_queue.maxlen:
                    self._messages_dropped += 1
                self._message_queue.append(msg)
            queue_len = len(self._message_queue)

        if queue_len >= self._drop_warning_threshold:
            logger.warning(
                f"WebSocket message queue at {queue_len}/{self.config.queue_max_size} "
                f"({self._messages_dropped} messages dropped total)"
            )

        # Dispatch to registered callbacks
        for callback in self._callbacks:
            try:
                callback(msgs)
            except Exception as e:
                logger.error(
                    f"Error in callback {callback.__name__}: {e}", exc_info=True
                )

    def monitor_heartbeat(self) -> bool:
        """
        Check if messages are flowing (heartbeat).

        Returns:
            True if messages received recently, False if stale

        Note: Should only be called during market hours
        """
        if not self._last_message_time:
            return False

        elapsed = time.time() - self._last_message_time
        if elapsed > self.config.heartbeat_timeout:
            logger.warning(
                f"No messages received in {elapsed:.1f}s (timeout: {self.config.heartbeat_timeout}s)"
            )
            return False

        return True

    def get_pending_messages(self) -> list[dict]:
        """
        Retrieve and clear pending messages from queue.

        Returns:
            List of message dicts
        """
        with self._queue_lock:
            messages = list(self._message_queue)
            self._message_queue.clear()
            return messages

    def __enter__(self) -> PolygonWebSocketClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - ensures clean shutdown."""
        self.stop()
