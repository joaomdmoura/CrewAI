"""
SQLite-based implementation of flow state persistence.
"""

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence


class SQLiteFlowPersistence(FlowPersistence):
    """SQLite-based implementation of flow state persistence.
    
    This class provides a simple, file-based persistence implementation using SQLite.
    It's suitable for development and testing, or for production use cases with
    moderate performance requirements.
    """
    
    db_path: str  # Type annotation for instance variable
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize SQLite persistence.
        
        Args:
            db_path: Path to the SQLite database file. If not provided, uses
                    db_storage_path() from utilities.paths.
                    
        Raises:
            ValueError: If db_path is invalid
        """
        from crewai.utilities.paths import db_storage_path
        # Get path from argument or default location
        path = db_path or db_storage_path()
        
        if not path:
            raise ValueError("Database path must be provided")
            
        self.db_path = path  # Now mypy knows this is str
        self.init_db()
    
    def init_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS flow_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                flow_uuid TEXT NOT NULL,
                method_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                state_json TEXT NOT NULL
            )
            """)
            # Add index for faster UUID lookups
            conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_flow_states_uuid 
            ON flow_states(flow_uuid)
            """)
    
    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: Union[Dict[str, Any], BaseModel],
    ) -> None:
        """Save the current flow state to SQLite.
        
        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """
        # Convert state_data to dict, handling both Pydantic and dict cases
        if isinstance(state_data, BaseModel):
            state_dict = dict(state_data)  # Use dict() for better type compatibility
        elif isinstance(state_data, dict):
            state_dict = state_data
        else:
            raise ValueError(
                f"state_data must be either a Pydantic BaseModel or dict, got {type(state_data)}"
            )
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            INSERT INTO flow_states (
                flow_uuid,
                method_name,
                timestamp,
                state_json
            ) VALUES (?, ?, ?, ?)
            """, (
                flow_uuid,
                method_name,
                datetime.utcnow().isoformat(),
                json.dumps(state_dict),
            ))
    
    def load_state(self, flow_uuid: str) -> Optional[Dict[str, Any]]:
        """Load the most recent state for a given flow UUID.
        
        Args:
            flow_uuid: Unique identifier for the flow instance
            
        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
            SELECT state_json
            FROM flow_states
            WHERE flow_uuid = ?
            ORDER BY id DESC
            LIMIT 1
            """, (flow_uuid,))
            row = cursor.fetchone()
            
        if row:
            return json.loads(row[0])
        return None
