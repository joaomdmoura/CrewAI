import csv
from typing import Dict, List
from pathlib import Path

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """A knowledge source that stores and queries CSV file content using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Load and preprocess CSV file content."""
        super().load_content()  # Validate the file path
        file_path_str = (
            str(self.file_path) if isinstance(self.file_path, Path) else self.file_path
        )
        with open(file_path_str, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            content = ""
            for row in reader:
                content += " ".join(row) + "\n"
        return {self.file_path: content}

    def add(self) -> None:
        """
        Add CSV file content to the knowledge source, chunk it, compute embeddings,
        and save the embeddings.
        """
        content_str = (
            str(self.content) if isinstance(self.content, dict) else self.content
        )
        new_chunks = self._chunk_text(content_str)
        self.chunks.extend(new_chunks)
        self.save_documents(metadata=self.metadata)

    def _chunk_text(self, text: str) -> List[str]:
        """Utility method to split text into chunks."""
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap)
        ]
