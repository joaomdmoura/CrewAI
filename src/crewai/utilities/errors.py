"""Error message definitions for CrewAI database operations."""
from typing import Optional


class DatabaseOperationError(Exception):
    """Base exception class for database operation errors."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        """Initialize the database operation error.

        Args:
            message: The error message to display
            original_error: The original exception that caused this error, if any
        """
        super().__init__(message)
        self.original_error = original_error


class DatabaseError:
    """Standardized error message templates for database operations."""

    INIT_ERROR: str = "Database initialization error: {}"
    SAVE_ERROR: str = "Error saving task outputs: {}"
    UPDATE_ERROR: str = "Error updating task outputs: {}"
    LOAD_ERROR: str = "Error loading task outputs: {}"
    DELETE_ERROR: str = "Error deleting task outputs: {}"

    @classmethod
    def format_error(cls, template: str, error: Exception) -> str:
        """Format an error message with the given template and error.

        Args:
            template: The error message template to use
            error: The exception to format into the template

        Returns:
            The formatted error message
        """
        return template.format(str(error))
