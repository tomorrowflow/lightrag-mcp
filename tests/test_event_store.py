"""
Tests for event store functionality.

This module tests the event store implementations including:
- In-memory event store functionality
- File-based event store tests
- Resumability feature tests (Last-Event-ID header)
- Event cleanup and retention tests
- Thread safety tests
"""

import asyncio
import json
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from lightrag_mcp.event_store import (
    EventStore, InMemoryEventStore, FileEventStore, StoredEvent,
    create_event_store, EventStoreError, EventStoreCleanupError
)


class TestStoredEvent:
    """Test StoredEvent dataclass functionality."""

    def test_stored_event_creation(self):
        """Test creating a StoredEvent."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        timestamp = datetime.utcnow()

        event = StoredEvent(
            event_id="test-123",
            session_id="session-456",
            message=message,
            timestamp=timestamp,
            stream_id="stream-789"
        )

        assert event.event_id == "test-123"
        assert event.session_id == "session-456"
        assert event.message == message
        assert event.timestamp == timestamp
        assert event.stream_id == "stream-789"

    def test_stored_event_to_dict(self):
        """Test converting StoredEvent to dictionary."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        timestamp = datetime.utcnow()

        event = StoredEvent(
            event_id="test-123",
            session_id="session-456",
            message=message,
            timestamp=timestamp,
            stream_id="stream-789"
        )

        data = event.to_dict()

        assert data["event_id"] == "test-123"
        assert data["session_id"] == "session-456"
        assert data["message"] == message
        assert data["stream_id"] == "stream-789"
        # Timestamp should be ISO format string
        assert isinstance(data["timestamp"], str)

    def test_stored_event_from_dict(self):
        """Test creating StoredEvent from dictionary."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        timestamp_str = "2023-01-01T12:00:00"

        data = {
            "event_id": "test-123",
            "session_id": "session-456",
            "message": message,
            "timestamp": timestamp_str,
            "stream_id": "stream-789"
        }

        event = StoredEvent.from_dict(data)

        assert event.event_id == "test-123"
        assert event.session_id == "session-456"
        assert event.message == message
        assert event.stream_id == "stream-789"
        assert isinstance(event.timestamp, datetime)

    def test_stored_event_from_dict_without_stream_id(self):
        """Test creating StoredEvent from dictionary without stream_id."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        timestamp_str = "2023-01-01T12:00:00"

        data = {
            "event_id": "test-123",
            "session_id": "session-456",
            "message": message,
            "timestamp": timestamp_str
        }

        event = StoredEvent.from_dict(data)

        assert event.event_id == "test-123"
        assert event.session_id == "session-456"
        assert event.message == message
        assert event.stream_id is None


class TestInMemoryEventStore:
    """Test InMemoryEventStore functionality."""

    @pytest.mark.asyncio
    async def test_store_event(self, in_memory_event_store):
        """Test storing an event."""
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}
        session_id = "test-session"

        event_id = await in_memory_event_store.store_event(
            session_id=session_id,
            message=message,
            stream_id="test-stream"
        )

        assert isinstance(event_id, str)
        assert len(event_id) > 0

        # Verify event was stored
        events = await in_memory_event_store.get_events_after(session_id)
        assert len(events) == 1
        assert events[0].event_id == event_id
        assert events[0].session_id == session_id
        assert events[0].message == message
        assert events[0].stream_id == "test-stream"

    @pytest.mark.asyncio
    async def test_get_events_after_with_last_event_id(self, in_memory_event_store):
        """Test getting events after a specific event ID."""
        session_id = "test-session"
        messages = [
            {"jsonrpc": "2.0", "id": 1, "method": "test1"},
            {"jsonrpc": "2.0", "id": 2, "method": "test2"},
            {"jsonrpc": "2.0", "id": 3, "method": "test3"}
        ]

        # Store events
        event_ids = []
        for msg in messages:
            event_id = await in_memory_event_store.store_event(session_id, msg)
            event_ids.append(event_id)

        # Get all events
        all_events = await in_memory_event_store.get_events_after(session_id)
        assert len(all_events) == 3

        # Get events after first event
        events_after_first = await in_memory_event_store.get_events_after(
            session_id, last_event_id=event_ids[0]
        )
        assert len(events_after_first) == 2
        assert events_after_first[0].event_id == event_ids[1]
        assert events_after_first[1].event_id == event_ids[2]

        # Get events after non-existent event
        events_after_invalid = await in_memory_event_store.get_events_after(
            session_id, last_event_id="invalid-id"
        )
        assert len(events_after_invalid) == 0

    @pytest.mark.asyncio
    async def test_get_events_after_with_stream_filter(self, in_memory_event_store):
        """Test getting events filtered by stream ID."""
        session_id = "test-session"

        # Store events with different stream IDs
        await in_memory_event_store.store_event(
            session_id, {"method": "test1"}, stream_id="stream1"
        )
        await in_memory_event_store.store_event(
            session_id, {"method": "test2"}, stream_id="stream2"
        )
        await in_memory_event_store.store_event(
            session_id, {"method": "test3"}, stream_id="stream1"
        )

        # Get all events
        all_events = await in_memory_event_store.get_events_after(session_id)
        assert len(all_events) == 3

        # Get events for stream1 only
        stream1_events = await in_memory_event_store.get_events_after(
            session_id, stream_id="stream1"
        )
        assert len(stream1_events) == 2
        for event in stream1_events:
            assert event.stream_id == "stream1"

        # Get events for stream2 only
        stream2_events = await in_memory_event_store.get_events_after(
            session_id, stream_id="stream2"
        )
        assert len(stream2_events) == 1
        assert stream2_events[0].stream_id == "stream2"

    @pytest.mark.asyncio
    async def test_get_events_after_with_limit(self, in_memory_event_store):
        """Test getting events with limit."""
        session_id = "test-session"

        # Store multiple events
        for i in range(10):
            await in_memory_event_store.store_event(
                session_id, {"method": f"test{i}"}
            )

        # Get events with limit
        limited_events = await in_memory_event_store.get_events_after(
            session_id, limit=5
        )
        assert len(limited_events) == 5

        # Get all events
        all_events = await in_memory_event_store.get_events_after(session_id)
        assert len(all_events) == 10

    @pytest.mark.asyncio
    async def test_max_events_per_session(self):
        """Test maximum events per session limit."""
        store = InMemoryEventStore(max_events_per_session=3)
        session_id = "test-session"

        # Store events up to limit
        for i in range(5):  # Store more than limit
            await store.store_event(session_id, {"method": f"test{i}"})

        # Should only keep the last 3 events
        events = await store.get_events_after(session_id)
        assert len(events) == 3

        # Verify they're the most recent ones
        assert events[0].message["method"] == "test2"
        assert events[1].message["method"] == "test3"
        assert events[2].message["method"] == "test4"

    @pytest.mark.asyncio
    async def test_cleanup_old_events(self, in_memory_event_store):
        """Test cleanup of old events."""
        session_id = "test-session"

        # Store some events
        for i in range(3):
            await in_memory_event_store.store_event(session_id, {"method": f"test{i}"})

        # Manually set old timestamps for some events
        events = list(in_memory_event_store._session_events[session_id])
        old_time = datetime.utcnow() - timedelta(hours=25)
        events[0].timestamp = old_time
        events[1].timestamp = old_time

        # Cleanup events older than 24 hours
        cleaned_count = await in_memory_event_store.cleanup_old_events(retention_hours=24)
        assert cleaned_count == 2

        # Should have only 1 event left
        remaining_events = await in_memory_event_store.get_events_after(session_id)
        assert len(remaining_events) == 1
        assert remaining_events[0].message["method"] == "test2"

    @pytest.mark.asyncio
    async def test_get_session_event_count(self, in_memory_event_store):
        """Test getting session event count."""
        session_id1 = "session1"
        session_id2 = "session2"

        # No events initially
        assert await in_memory_event_store.get_session_event_count(session_id1) == 0

        # Store events in different sessions
        await in_memory_event_store.store_event(session_id1, {"method": "test1"})
        await in_memory_event_store.store_event(session_id1, {"method": "test2"})
        await in_memory_event_store.store_event(session_id2, {"method": "test3"})

        assert await in_memory_event_store.get_session_event_count(session_id1) == 2
        assert await in_memory_event_store.get_session_event_count(session_id2) == 1
        assert await in_memory_event_store.get_session_event_count("nonexistent") == 0

    @pytest.mark.asyncio
    async def test_thread_safety(self, in_memory_event_store):
        """Test thread safety of in-memory event store."""
        session_id = "test-session"
        results = []

        async def store_events_async():
            """Async function to store events."""
            for i in range(10):
                await in_memory_event_store.store_event(
                    session_id, {"method": f"async{i}"}
                )

        def store_events_sync():
            """Sync function to store events (run in thread)."""
            async def _store():
                for i in range(10):
                    await in_memory_event_store.store_event(
                        session_id, {"method": f"sync{i}"}
                    )
            asyncio.run(_store())

        # Run async and sync operations concurrently
        await asyncio.gather(
            store_events_async(),
            asyncio.to_thread(store_events_sync)
        )

        # Should have 20 events total
        events = await in_memory_event_store.get_events_after(session_id)
        assert len(events) == 20

        # Verify all events are present
        async_methods = [e.message["method"] for e in events if "async" in e.message["method"]]
        sync_methods = [e.message["method"] for e in events if "sync" in e.message["method"]]

        assert len(async_methods) == 10
        assert len(sync_methods) == 10


class TestFileEventStore:
    """Test FileEventStore functionality."""

    @pytest.mark.asyncio
    async def test_file_store_initialization(self, temp_dir):
        """Test file event store initialization."""
        storage_path = os.path.join(temp_dir, "event_store")
        store = FileEventStore(storage_path=storage_path, max_events_per_session=100)

        assert os.path.exists(storage_path)
        assert store.storage_path == os.path.join(storage_path)
        assert store.max_events_per_session == 100

    @pytest.mark.asyncio
    async def test_file_store_event_persistence(self, file_event_store):
        """Test that events are persisted to files."""
        session_id = "test-session"
        message = {"jsonrpc": "2.0", "id": 1, "method": "test"}

        # Store event
        event_id = await file_event_store.store_event(session_id, message)

        # Verify event file exists
        event_file = file_event_store._get_event_file(event_id)
        assert os.path.exists(event_file)

        # Verify session file exists
        session_file = file_event_store._get_session_file(session_id)
        assert os.path.exists(session_file)

        # Verify we can retrieve the event
        events = await file_event_store.get_events_after(session_id)
        assert len(events) == 1
        assert events[0].event_id == event_id
        assert events[0].message == message

    @pytest.mark.asyncio
    async def test_file_store_event_recovery(self, temp_dir):
        """Test that events are recovered from files on restart."""
        storage_path = os.path.join(temp_dir, "event_store")

        # Create and populate store
        store1 = FileEventStore(storage_path=storage_path, max_events_per_session=100)
        session_id = "test-session"

        await store1.store_event(session_id, {"method": "test1"})
        await store1.store_event(session_id, {"method": "test2"})

        # Create new store instance (simulating restart)
        store2 = FileEventStore(storage_path=storage_path, max_events_per_session=100)

        # Should have recovered the events
        events = await store2.get_events_after(session_id)
        assert len(events) == 2
        assert events[0].message["method"] == "test1"
        assert events[1].message["method"] == "test2"

    @pytest.mark.asyncio
    async def test_file_store_cleanup_deletes_files(self, file_event_store):
        """Test that cleanup removes old event files."""
        session_id = "test-session"

        # Store events
        event_ids = []
        for i in range(3):
            event_id = await file_event_store.store_event(
                session_id, {"method": f"test{i}"}
            )
            event_ids.append(event_id)

        # Manually set old timestamps
        events = list(file_event_store._session_events[session_id])
        old_time = datetime.utcnow() - timedelta(hours=25)
        events[0].timestamp = old_time
        events[1].timestamp = old_time

        # Save updated session file
        file_event_store._save_session_events(session_id)

        # Cleanup
        cleaned_count = await file_event_store.cleanup_old_events(retention_hours=24)

        assert cleaned_count == 2

        # Verify old event files are deleted
        old_event_file1 = file_event_store._get_event_file(event_ids[0])
        old_event_file2 = file_event_store._get_event_file(event_ids[1])
        assert not os.path.exists(old_event_file1)
        assert not os.path.exists(old_event_file2)

        # Verify new event file still exists
        new_event_file = file_event_store._get_event_file(event_ids[2])
        assert os.path.exists(new_event_file)

    @pytest.mark.asyncio
    async def test_file_store_max_events_enforcement(self, temp_dir):
        """Test max events per session in file store."""
        storage_path = os.path.join(temp_dir, "event_store")
        store = FileEventStore(storage_path=storage_path, max_events_per_session=2)
        session_id = "test-session"

        # Store more events than limit
        for i in range(4):
            await store.store_event(session_id, {"method": f"test{i}"})

        # Should only keep 2 events
        events = await store.get_events_after(session_id)
        assert len(events) == 2

        # Verify they're the most recent
        assert events[0].message["method"] == "test2"
        assert events[1].message["method"] == "test3"

    @pytest.mark.asyncio
    async def test_file_store_corruption_recovery(self, temp_dir):
        """Test recovery from corrupted event files."""
        storage_path = os.path.join(temp_dir, "event_store")
        store = FileEventStore(storage_path=storage_path, max_events_per_session=100)

        session_id = "test-session"
        session_file = store._get_session_file(session_id)

        # Create corrupted session file
        os.makedirs(os.path.dirname(session_file), exist_ok=True)
        with open(session_file, 'w') as f:
            f.write("invalid json")

        # Create new store instance - should handle corruption gracefully
        store2 = FileEventStore(storage_path=storage_path, max_events_per_session=100)

        # Should still work despite corrupted file
        events = await store2.get_events_after(session_id)
        assert len(events) == 0  # No valid events loaded


class TestEventStoreFactory:
    """Test event store factory function."""

    def test_create_memory_store(self):
        """Test creating in-memory event store."""
        store = create_event_store("memory", max_events_per_session=50)
        assert isinstance(store, InMemoryEventStore)
        assert store.max_events_per_session == 50

    def test_create_file_store(self, temp_dir):
        """Test creating file-based event store."""
        storage_path = os.path.join(temp_dir, "test_store")
        store = create_event_store("file", storage_path=storage_path, max_events_per_session=75)
        assert isinstance(store, FileEventStore)
        assert store.storage_path == storage_path
        assert store.max_events_per_session == 75

    def test_create_invalid_store_type(self):
        """Test creating store with invalid type."""
        with pytest.raises(ValueError, match="Unsupported storage type"):
            create_event_store("invalid")


class TestResumabilityFeatures:
    """Test resumability features using Last-Event-ID."""

    @pytest.mark.asyncio
    async def test_resumability_with_last_event_id(self, in_memory_event_store):
        """Test resumability using Last-Event-ID header simulation."""
        session_id = "test-session"

        # Simulate a series of events
        events_data = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "query"}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "insert"}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "get_docs"}},
        ]

        stored_events = []
        for event_data in events_data:
            event_id = await in_memory_event_store.store_event(session_id, event_data)
            stored_events.append(event_id)

        # Simulate client reconnecting with Last-Event-ID
        last_event_id = stored_events[1]  # After "tools/call query"

        # Get events after the last known event
        new_events = await in_memory_event_store.get_events_after(
            session_id, last_event_id=last_event_id
        )

        # Should get events 2 and 3 (indices 2, 3 in events_data)
        assert len(new_events) == 2
        assert new_events[0].message["id"] == 3
        assert new_events[1].message["id"] == 4

    @pytest.mark.asyncio
    async def test_resumability_with_stream_filtering(self, in_memory_event_store):
        """Test resumability with stream-specific filtering."""
        session_id = "test-session"

        # Store events in different streams
        streams = ["notifications", "data", "errors", "notifications"]
        event_ids = []

        for i, stream in enumerate(streams):
            event_id = await in_memory_event_store.store_event(
                session_id,
                {"id": i + 1, "stream": stream},
                stream_id=stream
            )
            event_ids.append(event_id)

        # Simulate resuming notifications stream after first notification
        notifications_after_first = await in_memory_event_store.get_events_after(
            session_id,
            last_event_id=event_ids[0],  # First notification
            stream_id="notifications"
        )

        # Should only get the second notification
        assert len(notifications_after_first) == 1
        assert notifications_after_first[0].stream_id == "notifications"
        assert notifications_after_first[0].message["id"] == 4  # Last event


class TestEventStoreErrors:
    """Test error handling in event stores."""

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self, in_memory_event_store):
        """Test error handling during cleanup."""
        # This is mainly for file store, but test the interface
        result = await in_memory_event_store.cleanup_old_events(retention_hours=1)
        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_invalid_session_access(self, in_memory_event_store):
        """Test accessing non-existent session."""
        events = await in_memory_event_store.get_events_after("nonexistent-session")
        assert events == []

        count = await in_memory_event_store.get_session_event_count("nonexistent-session")
        assert count == 0


class TestConcurrentAccess:
    """Test concurrent access to event stores."""

    @pytest.mark.asyncio
    async def test_concurrent_store_operations(self, in_memory_event_store):
        """Test concurrent store operations."""
        session_id = "test-session"

        async def store_concurrent(i):
            await in_memory_event_store.store_event(
                session_id, {"method": f"concurrent{i}"}
            )

        # Run multiple concurrent store operations
        tasks = [store_concurrent(i) for i in range(20)]
        await asyncio.gather(*tasks)

        # Verify all events were stored
        events = await in_memory_event_store.get_events_after(session_id)
        assert len(events) == 20

        # Verify no duplicates or corruption
        methods = [e.message["method"] for e in events]
        assert len(set(methods)) == 20  # All unique