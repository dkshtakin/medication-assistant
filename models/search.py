from pydantic import BaseModel
from typing import Optional, List, Any


class SearchRequest(BaseModel):
    query: str
    k: int
    searchname: Optional[str] = None
    retrieval_type: Optional[str] = None


class SearchResponse(BaseModel):
    chunks: List[Any]
