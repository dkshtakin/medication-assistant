from fastapi import APIRouter
from models.search import SearchRequest, SearchResponse
from retrieval.retriever import EmptyRetriever
from retrieval.qdrant.hybrid import QdrantHybridRetriever

router = APIRouter()
retriever = QdrantHybridRetriever()


@router.post("/search")
async def search_query(request: SearchRequest) -> SearchResponse:
    chunks = retriever.retrieve(
        request.query, request.k, request.searchname, request.retrieval_type
    )
    return SearchResponse(chunks=chunks)
