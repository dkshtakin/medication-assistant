from fastapi import APIRouter
from models.search import SearchRequest, SearchResponse
from retrieval.retriever import EmptyRetriever

router = APIRouter()
retriever = EmptyRetriever()


@router.post("/search")
async def search_query(request: SearchRequest) -> SearchResponse:
    chunks = retriever.retrieve(
        request.query, request.k, request.searchname, request.retrieval_type
    )
    return SearchResponse(chunks=chunks)
