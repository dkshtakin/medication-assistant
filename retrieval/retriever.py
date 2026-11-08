from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int, searchname: str, retrieval_type: str):
        pass


class EmptyRetriever(Retriever):
    def retrieve(
        self,
        query: str = "",
        k: int = 3,
        searchname: str = None,
        retrieval_type: str = None,
    ):
        return {}
