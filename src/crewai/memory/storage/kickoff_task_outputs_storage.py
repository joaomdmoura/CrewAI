import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from crewai.task import Task
from crewai.utilities import Printer
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from crewai.utilities.paths import DatabaseStorage, DatabaseOperationError

logger = logging.getLogger(__name__)


class KickoffTaskOutputsSQLiteStorage:
    """
    An updated SQLite storage class for kickoff task outputs storage.
    """

    def __init__(self, db_storage: DatabaseStorage = DatabaseStorage()) -> None:
        self.db_path = f"{db_storage.db_storage_path}/latest_kickoff_task_outputs.db"
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the SQLite database and create the latest_kickoff_task_outputs table.

        This method sets up the database schema for storing task outputs. It creates
        a table with columns for task_id, expected_output, output (as JSON),
        task_index, inputs (as JSON), was_replayed flag, and timestamp.

        Raises:
            DatabaseOperationError: If database initialization fails due to SQLite errors.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS latest_kickoff_task_outputs (
                        task_id TEXT PRIMARY KEY,
                        expected_output TEXT,
                        output JSON,
                        task_index INTEGER,
                        inputs JSON,
                        was_replayed BOOLEAN,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
        except sqlite3.Error as e:
            error_msg = DatabaseError.format_error(DatabaseError.INIT_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e)

    def add(
        self,
        task: Task,
        output: Dict[str, Any],
        task_index: int,
        was_replayed: bool = False,
        inputs: Dict[str, Any] = {},
    ) -> None:
        """Add a new task output record to the database.

        Args:
            task: The Task object containing task details.
            output: Dictionary containing the task's output data.
            task_index: Integer index of the task in the sequence.
            was_replayed: Boolean indicating if this was a replay execution.
            inputs: Dictionary of input parameters used for the task.

        Raises:
            DatabaseOperationError: If saving the task output fails due to SQLite errors.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                cursor = conn.cursor()
                cursor.execute(
                    """
                INSERT OR REPLACE INTO latest_kickoff_task_outputs
                (task_id, expected_output, output, task_index, inputs, was_replayed)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                    (
                        str(task.id),
                        task.expected_output,
                        json.dumps(output, cls=CrewJSONEncoder),
                        task_index,
                        json.dumps(inputs, cls=CrewJSONEncoder),
                        was_replayed,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            error_msg = DatabaseError.format_error(DatabaseError.SAVE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e)

    def update(
        self,
        task_index: int,
        **kwargs: Any,
    ) -> None:
        """Update an existing task output record in the database.

        Updates fields of a task output record identified by task_index. The fields
        to update are provided as keyword arguments.

        Args:
            task_index: Integer index of the task to update.
            **kwargs: Arbitrary keyword arguments representing fields to update.
                     Values that are dictionaries will be JSON encoded.

        Raises:
            DatabaseOperationError: If updating the task output fails due to SQLite errors.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                cursor = conn.cursor()

                fields = []
                values = []
                for key, value in kwargs.items():
                    fields.append(f"{key} = ?")
                    values.append(
                        json.dumps(value, cls=CrewJSONEncoder)
                        if isinstance(value, dict)
                        else value
                    )

                query = f"UPDATE latest_kickoff_task_outputs SET {', '.join(fields)} WHERE task_index = ?"  # nosec
                values.append(task_index)

                cursor.execute(query, tuple(values))
                conn.commit()

                if cursor.rowcount == 0:
                    logger.warning(f"No row found with task_index {task_index}. No update performed.")
        except sqlite3.Error as e:
            error_msg = DatabaseError.format_error(DatabaseError.UPDATE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e)

    def load(self) -> List[Dict[str, Any]]:
        """Load all task output records from the database.

        Returns:
            List of dictionaries containing task output records, ordered by task_index.
            Each dictionary contains: task_id, expected_output, output, task_index,
            inputs, was_replayed, and timestamp.

        Raises:
            DatabaseOperationError: If loading task outputs fails due to SQLite errors.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT *
                FROM latest_kickoff_task_outputs
                ORDER BY task_index
                """)

                rows = cursor.fetchall()
                results = []
                for row in rows:
                    result = {
                        "task_id": row[0],
                        "expected_output": row[1],
                        "output": json.loads(row[2]),
                        "task_index": row[3],
                        "inputs": json.loads(row[4]),
                        "was_replayed": row[5],
                        "timestamp": row[6],
                    }
                    results.append(result)

                return results

        except sqlite3.Error as e:
            error_msg = DatabaseError.format_error(DatabaseError.LOAD_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e)

    def delete_all(self) -> None:
        """Delete all task output records from the database.

        This method removes all records from the latest_kickoff_task_outputs table.
        Use with caution as this operation cannot be undone.

        Raises:
            DatabaseOperationError: If deleting task outputs fails due to SQLite errors.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("BEGIN TRANSACTION")
                cursor = conn.cursor()
                cursor.execute("DELETE FROM latest_kickoff_task_outputs")
                conn.commit()
        except sqlite3.Error as e:
            error_msg = DatabaseError.format_error(DatabaseError.DELETE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e)
