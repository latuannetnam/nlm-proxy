"""Session store for maintaining NotebookLM conversation state across Open WebUI chats."""

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class SessionStore:
    """Thread-safe in-memory store for chat session to conversation ID mappings with TTL support."""

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize the session store.

        Args:
            ttl_seconds: Time-to-live for sessions in seconds (default: 24 hours)
        """
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, dict] = {}  # chat_id -> {conversation_id, timestamp}
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._shutdown = False

        # Start background cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start the background thread that periodically cleans up expired sessions."""
        def cleanup_loop():
            while not self._shutdown:
                time.sleep(300)  # Run every 5 minutes
                if not self._shutdown:
                    expired_count = self.cleanup_expired()
                    if expired_count > 0:
                        logger.info(f"[SESSION] Cleaned up {expired_count} expired sessions")

        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def get(self, chat_id: str) -> Optional[str]:
        """
        Get conversation_id for a chat_id if not expired.

        Args:
            chat_id: The Open WebUI chat ID

        Returns:
            conversation_id if found and not expired, None otherwise
        """
        with self._lock:
            if chat_id not in self._store:
                return None

            session = self._store[chat_id]
            age = time.time() - session["timestamp"]

            if age > self.ttl_seconds:
                # Expired, remove it
                del self._store[chat_id]
                logger.debug(f"[SESSION] Expired session removed: chat_id={chat_id}")
                return None

            logger.debug(
                f"[SESSION] Session loaded: chat_id={chat_id}, "
                f"conversation_id={session['conversation_id']} (age={int(age)}s)"
            )
            return session["conversation_id"]

    def set(self, chat_id: str, conversation_id: str) -> None:
        """
        Store or update a chat_id to conversation_id mapping.

        Args:
            chat_id: The Open WebUI chat ID
            conversation_id: The NotebookLM conversation ID
        """
        with self._lock:
            is_new = chat_id not in self._store
            self._store[chat_id] = {
                "conversation_id": conversation_id,
                "timestamp": time.time(),
            }
            action = "created" if is_new else "updated"
            logger.info(
                f"[SESSION] Session {action}: chat_id={chat_id}, "
                f"conversation_id={conversation_id}"
            )

    def delete(self, chat_id: str) -> bool:
        """
        Delete a specific session.

        Args:
            chat_id: The Open WebUI chat ID

        Returns:
            True if session was deleted, False if it didn't exist
        """
        with self._lock:
            if chat_id in self._store:
                conversation_id = self._store[chat_id]["conversation_id"]
                del self._store[chat_id]
                logger.info(
                    f"[SESSION] Session deleted: chat_id={chat_id}, "
                    f"conversation_id={conversation_id}"
                )
                return True
            return False

    def list_all(self) -> dict[str, dict]:
        """
        List all active sessions with their metadata.

        Returns:
            Dictionary of chat_id to session info (conversation_id, age_seconds)
        """
        with self._lock:
            current_time = time.time()
            return {
                chat_id: {
                    "conversation_id": session["conversation_id"],
                    "age_seconds": int(current_time - session["timestamp"]),
                    "expires_in_seconds": int(
                        self.ttl_seconds - (current_time - session["timestamp"])
                    ),
                }
                for chat_id, session in self._store.items()
            }

    def cleanup_expired(self) -> int:
        """
        Remove all expired sessions.

        Returns:
            Number of sessions removed
        """
        with self._lock:
            current_time = time.time()
            expired_chat_ids = [
                chat_id
                for chat_id, session in self._store.items()
                if (current_time - session["timestamp"]) > self.ttl_seconds
            ]

            for chat_id in expired_chat_ids:
                del self._store[chat_id]

            return len(expired_chat_ids)

    def get_stats(self) -> dict:
        """
        Get statistics about current sessions.

        Returns:
            Dictionary with session statistics
        """
        with self._lock:
            if not self._store:
                return {
                    "total_sessions": 0,
                    "ttl_seconds": self.ttl_seconds,
                }

            current_time = time.time()
            ages = [current_time - session["timestamp"] for session in self._store.values()]

            return {
                "total_sessions": len(self._store),
                "ttl_seconds": self.ttl_seconds,
                "oldest_session_age_seconds": int(max(ages)),
                "newest_session_age_seconds": int(min(ages)),
            }

    def shutdown(self):
        """Shutdown the cleanup thread."""
        self._shutdown = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)
