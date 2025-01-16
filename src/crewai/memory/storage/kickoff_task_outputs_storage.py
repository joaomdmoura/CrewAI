import json
import sqlite3
from typing import Any, Dict, List, Optional

from crewai.task import Task
from crewai.utilities import Printer
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from crewai.utilities.paths import DatabaseStorage


class KickoffTaskOutputsSQLiteStorage:
    """
    An updated SQLite storage class for kickoff task outputs storage.
    """

    def __init__(self, db_storage: DatabaseStorage = DatabaseStorage()) -> None:
        self.db_path = f"{db_storage.db_storage_path}/latest_kickoff_task_outputs.db"
        self._printer: Printer = Printer()
        self._initialize_db()

    def _initialize_db(self):
        """
        Initializes the SQLite database and creates LTM table
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
            self._printer.print(
                content=f"SAVING KICKOFF TASK OUTPUTS ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def add(
        self,
        task: Task,
        output: Dict[str, Any],
        task_index: int,
        was_replayed: bool = False,
        inputs: Dict[str, Any] = {},
    ):
        try:
            with sqlite3.connect(self.db_path) as conn:
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
            self._printer.print(
                content=f"SAVING KICKOFF TASK OUTPUTS ERROR: An error occurred during database initialization: {e}",
                color="red",
            )

    def update(
        self,
        task_index: int,
        **kwargs,
    ):
        """
        Updates an existing row in the latest_kickoff_task_outputs table based on task_index.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
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
                    self._printer.print(
                        f"No row found with task_index {task_index}. No update performed.",
                        color="red",
                    )
        except sqlite3.Error as e:
            self._printer.print(f"UPDATE KICKOFF TASK OUTPUTS ERROR: {e}", color="red")

    def load(self) -> Optional[List[Dict[str, Any]]]:
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
            self._printer.print(
                content=f"LOADING KICKOFF TASK OUTPUTS ERROR: An error occurred while querying kickoff task outputs: {e}",
                color="red",
            )
            return None

    def delete_all(self):
        """
        Deletes all rows from the latest_kickoff_task_outputs table.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM latest_kickoff_task_outputs")
                conn.commit()
        except sqlite3.Error as e:
            self._printer.print(
                content=f"ERROR: Failed to delete all kickoff task outputs: {e}",
                color="red",
            )
