import os
from pathlib import Path

import appdirs


class DatabaseStorage:
    def __init__(
        self,
        app_author: str = "CrewAI",
        app_name: str = "",
        data_dir: Path | None = None,
    ):
        self.app_author = app_author
        self.app_name = app_name if app_name else self._get_project_directoy_name()
        self.db_storage_path = (
            data_dir
            if data_dir
            else Path(appdirs.user_data_dir(self.app_name, self.app_author))
        )
        self.db_storage_path.mkdir(parents=True, exist_ok=True)

    def _get_project_directoy_name(self) -> str:
        """Returns the current project directory name."""
        project_directory_name = os.environ.get("CREWAI_STORAGE_DIR")

        if project_directory_name:
            return project_directory_name
        else:
            cwd = Path.cwd()
            project_directory_name = cwd.name
            return project_directory_name
