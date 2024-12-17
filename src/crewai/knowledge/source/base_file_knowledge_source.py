from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Union

from pydantic import Field

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.utilities.constants import KNOWLEDGE_DIRECTORY
from crewai.utilities.logger import Logger


class BaseFileKnowledgeSource(BaseKnowledgeSource, ABC):
    """Base class for knowledge sources that load content from files."""

    _logger: Logger = Logger(verbose=True)
    file_path: Union[Path, List[Path], str, List[str]] = Field(
        default=None,
        description="[Deprecated] The path to the file. Use file_paths instead.",
    )
    file_paths: Union[Path, List[Path], str, List[str]] = Field(
        ..., description="The path to the file"
    )
    content: Dict[Path, str] = Field(init=False, default_factory=dict)
    storage: KnowledgeStorage = Field(default_factory=KnowledgeStorage)
    safe_file_paths: List[Path] = Field(default_factory=list)

    def model_post_init(self, _):
        """Post-initialization method to load content."""
        self.safe_file_paths = self._process_file_paths()
        self.validate_content()
        self.content = self.load_content()

    @abstractmethod
    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess file content. Should be overridden by subclasses. Assume that the file path is relative to the project root in the knowledge directory."""
        pass

    def validate_content(self):
        """Validate the paths."""
        for path in self.safe_file_paths:
            if not path.exists():
                self._logger.log(
                    "error",
                    f"File not found: {path}. Try adding sources to the knowledge directory. If it's inside the knowledge directory, use the relative path.",
                    color="red",
                )
                raise FileNotFoundError(f"File not found: {path}")
            if not path.is_file():
                self._logger.log(
                    "error",
                    f"Path is not a file: {path}",
                    color="red",
                )

    def _save_documents(self):
        """Save the documents to the storage."""
        self.storage.save(self.chunks)

    def convert_to_path(self, path: Union[Path, str]) -> Path:
        """Convert a path to a Path object."""
        return Path(KNOWLEDGE_DIRECTORY + "/" + path) if isinstance(path, str) else path

    def _process_file_paths(self) -> List[Path]:
        """Convert file_path to a list of Path objects."""

        # Check if old file_path is being used
        if hasattr(self, "file_path") and self.file_path is not None:
            self._logger.log(
                "warning",
                "The 'file_path' attribute is deprecated and will be removed in a future version. Please use 'file_paths' instead.",
                color="yellow",
            )
            paths = (
                [self.file_path]
                if isinstance(self.file_path, (str, Path))
                else self.file_path
            )
        else:
            paths = (
                [self.file_paths]
                if isinstance(self.file_paths, (str, Path))
                else self.file_paths
            )

        if not isinstance(paths, list):
            raise ValueError(
                "file_path/file_paths must be a Path, str, or a list of these types"
            )

        return [self.convert_to_path(path) for path in paths]
