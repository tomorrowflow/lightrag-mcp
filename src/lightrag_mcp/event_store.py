"""
Event store implementation for LightRAG MCP server resumability support.

This module provides both in-memory and persistent event storage solutions
for maintaining session state and enabling resumability in streaming operations.
"""

import asyncio
import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import uuid4

from mcp.types import JSONRPCMessage

from .config import EVENT_RETENTION_HOURS, MAX_EVENTS_PER_SESSION

logger = logging.getLogger(__name__)


@dataclass
class StoredEvent:
    """
    Represents a stored event with metadata.
    """
    event_id: str
    session_id: str
    message: JSONRPCMessage
    timestamp: datetime
    stream_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "stream_id": self.stream_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoredEvent":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            session_id=data["session_id"],
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            stream_id=data.get("stream_id"),
        )


class EventStoreError(Exception):
    """Base exception for event store operations."""
    pass


class EventStoreCleanupError(EventStoreError):
    """Exception raised during event cleanup operations."""
    pass


class EventStore(ABC):
    """
    Abstract base class for event store implementations.

    Provides the interface for storing and retrieving JSON-RPC messages
    for resumability support in streaming operations.
    """

    @abstractmethod
    async def store_event(
        self,
        session_id: str,
        message: JSONRPCMessage,
        stream_id: Optional[str] = None
    ) -> str:
        """
        Store an event and return its unique event ID.

        Args:
            session_id: Unique identifier for the session
            message: The JSON-RPC message to store
            stream_id: Optional stream identifier for grouping events

        Returns:
            Unique event ID for the stored event
        """
        pass

    @abstractmethod
    async def get_events_after(
        self,
        session_id: str,
        last_event_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StoredEvent]:
        """
        Retrieve events after a specific event ID for resumability.

        Args:
            session_id: Session identifier
            last_event_id: Get events after this event ID (None for all events)
            stream_id: Optional stream identifier to filter events
            limit: Maximum number of events to return

        Returns:
            List of events in chronological order
        """
        pass

    @abstractmethod
    async def cleanup_old_events(self, retention_hours: Optional[int] = None) -> int:
        """
        Remove events older than the retention period.

        Args:
            retention_hours: Hours to retain events (uses config default if None)

        Returns:
            Number of events cleaned up
        """
        pass

    @abstractmethod
    async def get_session_event_count(self, session_id: str) -> int:
        """
        Get the total number of events for a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of events in the session
        """
        pass


class InMemoryEventStore(EventStore):
    """
    In-memory event store implementation using collections.deque.

    This implementation is suitable for development and testing but not
    for production use where persistence across restarts is required.
    """

    def __init__(self, max_events_per_session: int = MAX_EVENTS_PER_SESSION):
        """
        Initialize the in-memory event store.

        Args:
            max_events_per_session: Maximum events to keep per session
        """
        self.max_events_per_session = max_events_per_session
        self._lock = asyncio.Lock()

        # session_id -> deque of StoredEvent
        self._session_events: Dict[str, deque] = {}

        # event_id -> StoredEvent for quick lookup
        self._event_index: Dict[str, StoredEvent] = {}

        logger.info(f"Initialized in-memory event store with max {max_events_per_session} events per session")

    async def store_event(
        self,
        session_id: str,
        message: JSONRPCMessage,
        stream_id: Optional[str] = None
    ) -> str:
        """Store an event and return its unique event ID."""
        async with self._lock:
            event_id = str(uuid4())
            timestamp = datetime.utcnow()

            event = StoredEvent(
                event_id=event_id,
                session_id=session_id,
                message=message,
                timestamp=timestamp,
                stream_id=stream_id,
            )

            # Get or create deque for this session
            if session_id not in self._session_events:
                self._session_events[session_id] = deque(maxlen=self.max_events_per_session)

            # Handle deque maxlen - remove oldest event from index if deque is full
            session_deque = self._session_events[session_id]
            if len(session_deque) == self.max_events_per_session:
                oldest_event = session_deque[0]
                self._event_index.pop(oldest_event.event_id, None)

            # Add new event
            session_deque.append(event)
            self._event_index[event_id] = event

            logger.debug(f"Stored event {event_id} for session {session_id}")
            return event_id

    async def get_events_after(
        self,
        session_id: str,
        last_event_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StoredEvent]:
        """Retrieve events after a specific event ID."""
        async with self._lock:
            if session_id not in self._session_events:
                logger.warning(f"No events found for session {session_id}")
                return []

            session_events = self._session_events[session_id]

            # Find the starting point
            start_index = 0
            if last_event_id:
                if last_event_id not in self._event_index:
                    logger.warning(f"Event ID {last_event_id} not found in store")
                    return []

                # Find events after the specified event
                found_last = False
                for i, event in enumerate(session_events):
                    if event.event_id == last_event_id:
                        start_index = i + 1
                        found_last = True
                        break

                if not found_last:
                    logger.warning(f"Event ID {last_event_id} not found in session {session_id}")
                    return []

            # Get events from start_index onwards
            events = []
            for i in range(start_index, len(session_events)):
                event = session_events[i]

                # Filter by stream_id if specified
                if stream_id is not None and event.stream_id != stream_id:
                    continue

                events.append(event)

                # Apply limit if specified
                if limit and len(events) >= limit:
                    break

            logger.debug(f"Retrieved {len(events)} events for session {session_id} after event {last_event_id}")
            return events

    async def cleanup_old_events(self, retention_hours: Optional[int] = None) -> int:
        """Remove events older than the retention period."""
        if retention_hours is None:
            retention_hours = EVENT_RETENTION_HOURS

        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
            cleaned_count = 0

            # Clean up old events
            events_to_remove = []
            for event_id, event in self._event_index.items():
                if event.timestamp < cutoff_time:
                    events_to_remove.append(event_id)

            for event_id in events_to_remove:
                event = self._event_index.pop(event_id)

                # Remove from session deque
                if event.session_id in self._session_events:
                    try:
                        # Find and remove the event from the deque
                        session_deque = self._session_events[event.session_id]
                        # Convert to list to find and remove the specific event
                        event_list = list(session_deque)
                        event_list = [e for e in event_list if e.event_id != event_id]
                        self._session_events[event.session_id] = deque(
                            event_list,
                            maxlen=self.max_events_per_session
                        )
                    except (ValueError, IndexError):
                        pass  # Event not found in deque

                cleaned_count += 1

            # Clean up empty sessions
            empty_sessions = [
                session_id for session_id, events in self._session_events.items()
                if len(events) == 0
            ]
            for session_id in empty_sessions:
                del self._session_events[session_id]

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} events older than {retention_hours} hours")

            return cleaned_count

    async def get_session_event_count(self, session_id: str) -> int:
        """Get the total number of events for a session."""
        async with self._lock:
            if session_id not in self._session_events:
                return 0
            return len(self._session_events[session_id])


class FileEventStore(EventStore):
    """
    File-based persistent event store implementation.

    This implementation stores events as JSON files on disk, providing
    persistence across application restarts but with higher latency
    than in-memory storage.
    """

    def __init__(
        self,
        storage_path: str = "./event_store",
        max_events_per_session: int = MAX_EVENTS_PER_SESSION,
    ):
        """
        Initialize the file-based event store.

        Args:
            storage_path: Directory path for storing event files
            max_events_per_session: Maximum events to keep per session
        """
        self.storage_path = Path(storage_path)
        self.max_events_per_session = max_events_per_session
        self._lock = asyncio.Lock()

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory index for faster lookups
        self._event_index: Dict[str, StoredEvent] = {}
        self._session_events: Dict[str, deque] = {}

        # Load existing events on startup
        self._load_events()

        logger.info(f"Initialized file event store at {self.storage_path}")

    def _get_session_file(self, session_id: str) -> Path:
        """Get the file path for a session's events."""
        # Use a subdirectory structure to avoid too many files in one directory
        session_dir = self.storage_path / session_id[:2] / session_id[2:4]
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir / f"{session_id}.json"

    def _get_event_file(self, event_id: str) -> Path:
        """Get the file path for a specific event."""
        event_dir = self.storage_path / "events" / event_id[:2] / event_id[2:4]
        event_dir.mkdir(parents=True, exist_ok=True)
        return event_dir / f"{event_id}.json"

    def _load_events(self):
        """Load existing events from disk into memory."""
        try:
            # Load session files
            session_files = list(self.storage_path.rglob("*.json"))

            for session_file in session_files:
                if session_file.parent.name == "events":
                    continue  # Skip individual event files

                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)

                    session_id = session_data.get("session_id")
                    if not session_id:
                        continue

                    # Load events for this session
                    events = []
                    for event_data in session_data.get("events", []):
                        try:
                            event = StoredEvent.from_dict(event_data)
                            events.append(event)
                            self._event_index[event.event_id] = event
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Failed to load event from {session_file}: {e}")

                    if events:
                        self._session_events[session_id] = deque(
                            events,
                            maxlen=self.max_events_per_session
                        )

                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to load session file {session_file}: {e}")

            logger.info(f"Loaded {len(self._event_index)} events from disk")

        except Exception as e:
            logger.error(f"Failed to load events from disk: {e}")

    def _save_session_events(self, session_id: str):
        """Save all events for a session to disk."""
        try:
            if session_id not in self._session_events:
                return

            session_file = self._get_session_file(session_id)
            events_data = {
                "session_id": session_id,
                "events": [event.to_dict() for event in self._session_events[session_id]]
            }

            # Atomic write using temporary file
            temp_file = session_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(events_data, f, indent=2, ensure_ascii=False)

            temp_file.replace(session_file)

        except Exception as e:
            logger.error(f"Failed to save session {session_id} events: {e}")

    def _save_event_file(self, event: StoredEvent):
        """Save individual event to its own file."""
        try:
            event_file = self._get_event_file(event.event_id)
            with open(event_file, 'w', encoding='utf-8') as f:
                json.dump(event.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save event {event.event_id}: {e}")

    async def store_event(
        self,
        session_id: str,
        message: JSONRPCMessage,
        stream_id: Optional[str] = None
    ) -> str:
        """Store an event and return its unique event ID."""
        async with self._lock:
            event_id = str(uuid4())
            timestamp = datetime.utcnow()

            event = StoredEvent(
                event_id=event_id,
                session_id=session_id,
                message=message,
                timestamp=timestamp,
                stream_id=stream_id,
            )

            # Get or create deque for this session
            if session_id not in self._session_events:
                self._session_events[session_id] = deque(maxlen=self.max_events_per_session)

            # Handle deque maxlen - remove oldest event
            session_deque = self._session_events[session_id]
            if len(session_deque) == self.max_events_per_session:
                oldest_event = session_deque[0]
                self._event_index.pop(oldest_event.event_id, None)

                # Remove old event file
                try:
                    old_event_file = self._get_event_file(oldest_event.event_id)
                    if old_event_file.exists():
                        old_event_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove old event file: {e}")

            # Add new event
            session_deque.append(event)
            self._event_index[event_id] = event

            # Save to disk (in background to avoid blocking)
            try:
                self._save_session_events(session_id)
                self._save_event_file(event)
            except Exception as e:
                logger.error(f"Failed to persist event {event_id}: {e}")

            logger.debug(f"Stored event {event_id} for session {session_id}")
            return event_id

    async def get_events_after(
        self,
        session_id: str,
        last_event_id: Optional[str] = None,
        stream_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[StoredEvent]:
        """Retrieve events after a specific event ID."""
        async with self._lock:
            if session_id not in self._session_events:
                logger.warning(f"No events found for session {session_id}")
                return []

            session_events = self._session_events[session_id]

            # Find the starting point
            start_index = 0
            if last_event_id:
                if last_event_id not in self._event_index:
                    logger.warning(f"Event ID {last_event_id} not found in store")
                    return []

                # Find events after the specified event
                found_last = False
                for i, event in enumerate(session_events):
                    if event.event_id == last_event_id:
                        start_index = i + 1
                        found_last = True
                        break

                if not found_last:
                    logger.warning(f"Event ID {last_event_id} not found in session {session_id}")
                    return []

            # Get events from start_index onwards
            events = []
            for i in range(start_index, len(session_events)):
                event = session_events[i]

                # Filter by stream_id if specified
                if stream_id is not None and event.stream_id != stream_id:
                    continue

                events.append(event)

                # Apply limit if specified
                if limit and len(events) >= limit:
                    break

            logger.debug(f"Retrieved {len(events)} events for session {session_id} after event {last_event_id}")
            return events

    async def cleanup_old_events(self, retention_hours: Optional[int] = None) -> int:
        """Remove events older than the retention period."""
        if retention_hours is None:
            retention_hours = EVENT_RETENTION_HOURS

        async with self._lock:
            cutoff_time = datetime.utcnow() - timedelta(hours=retention_hours)
            cleaned_count = 0

            # Clean up old events
            events_to_remove = []
            for event_id, event in self._event_index.items():
                if event.timestamp < cutoff_time:
                    events_to_remove.append(event_id)

            for event_id in events_to_remove:
                event = self._event_index.pop(event_id)

                # Remove from session deque
                if event.session_id in self._session_events:
                    try:
                        session_deque = self._session_events[event.session_id]
                        event_list = list(session_deque)
                        event_list = [e for e in event_list if e.event_id != event_id]
                        self._session_events[event.session_id] = deque(
                            event_list,
                            maxlen=self.max_events_per_session
                        )
                    except (ValueError, IndexError):
                        pass

                # Remove event file
                try:
                    event_file = self._get_event_file(event_id)
                    if event_file.exists():
                        event_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove event file: {e}")

                cleaned_count += 1

            # Save updated session files
            affected_sessions = set()
            for event_id in events_to_remove:
                if event_id in self._event_index:
                    # Event wasn't actually removed, skip
                    continue
                # Find the event that was removed to get its session_id
                # We need to track this before removing events
                for event in list(self._event_index.values()):
                    if event.event_id == event_id:
                        affected_sessions.add(event.session_id)
                        break

            for session_id in affected_sessions:
                if session_id in self._session_events:
                    self._save_session_events(session_id)

            # Clean up empty sessions
            empty_sessions = [
                session_id for session_id, events in self._session_events.items()
                if len(events) == 0
            ]
            for session_id in empty_sessions:
                del self._session_events[session_id]
                # Remove session file
                try:
                    session_file = self._get_session_file(session_id)
                    if session_file.exists():
                        session_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to remove session file: {e}")

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} events older than {retention_hours} hours")

            return cleaned_count

    async def get_session_event_count(self, session_id: str) -> int:
        """Get the total number of events for a session."""
        async with self._lock:
            if session_id not in self._session_events:
                return 0
            return len(self._session_events[session_id])


def create_event_store(storage_type: str = "memory", **kwargs) -> EventStore:
    """
    Factory function to create the appropriate event store implementation.

    Args:
        storage_type: "memory" for InMemoryEventStore, "file" for FileEventStore
        **kwargs: Additional arguments passed to the store constructor

    Returns:
        Configured event store instance

    Raises:
        ValueError: If storage_type is not supported
    """
    if storage_type.lower() == "memory":
        return InMemoryEventStore(**kwargs)
    elif storage_type.lower() == "file":
        return FileEventStore(**kwargs)
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}. Use 'memory' or 'file'.")