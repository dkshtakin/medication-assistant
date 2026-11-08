from pydantic import BaseModel
from typing import Optional, Dict, Any


class SearchRequest(BaseModel):
    query: str
    k: int
    searchname: Optional[str] = None
    retrieval_type: Optional[str] = None


class SearchResponse(BaseModel):
    chunks: Dict[Any, Any]
