from crewai.memory.memory import Memory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.memory.storage.rag_storage import RAGStorage


class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, crew=None, embedder_config=None):
        storage = RAGStorage(type="short_term", embedder_config=embedder_config, crew=crew)
        super().__init__(storage)

    def save(self, item: ShortTermMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        super().save(item.data, item.metadata, item.agent)

    def search(self, query: str, score_threshold: float = 0.35):
        return self.storage.search(query=query, score_threshold=score_threshold)  # type: ignore # BUG? The reference is to the parent class, but the parent class does not have this parameters
