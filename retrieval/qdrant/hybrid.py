from retrieval.retriever import Retriever
from qdrant_client import models
from retrieval.qdrant.helper import QdrantHelper
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class QdrantHybridRetriever(Retriever):
    def __init__(self):
        self.qdrant_helper = QdrantHelper()
        self.collection_name = "embeddinggemma-300m_1280_16_hybrid"
        collection_info = self.qdrant_helper.client.get_collection(self.collection_name)
        assert collection_info.points_count > 0
        logger.info("QdrantHybridRetriever initialized")

    def retrieve(
        self,
        query: str = "",
        k: int = 3,
        searchname: str = None,
        retrieval_type: str = None,
    ):
        if retrieval_type is None:
            retrieval_type = "hybrid"

        query_sparse_embeddings = next(
            self.qdrant_helper.sparse_embeddings.query_embed(query)
        )
        query_sparse_embeddings = models.SparseVector(
            indices=query_sparse_embeddings.indices,
            values=query_sparse_embeddings.values,
        )
        query_dense_embeddings = self.qdrant_helper.dense_embeddings.encode_query(query)

        searchname_filter = (
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.searchname",
                        match=models.MatchValue(value=searchname.upper()),
                    )
                ]
            )
            if searchname is not None
            else None
        )

        points = []
        if retrieval_type == "hybrid":
            prefetch = [
                models.Prefetch(
                    query=query_dense_embeddings,
                    using="dense",
                    limit=k * 2,
                ),
                models.Prefetch(
                    query=query_sparse_embeddings,
                    using="sparse",
                    limit=k * 2,
                ),
            ]
            query_result = self.qdrant_helper.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                query_filter=searchname_filter,
                limit=k,
            )
            points += query_result.points

        if retrieval_type == "dense" or retrieval_type == "all":
            query_result = self.qdrant_helper.client.query_points(
                collection_name=self.collection_name,
                query=query_dense_embeddings,
                using="dense",
                query_filter=searchname_filter,
                limit=k,
            )
            points += query_result.points

        if retrieval_type == "sparse" or retrieval_type == "all":
            query_result = self.qdrant_helper.client.query_points(
                collection_name=self.collection_name,
                query=query_sparse_embeddings,
                using="sparse",
                query_filter=searchname_filter,
                limit=k,
            )
            points += query_result.points

        query_result = [point.payload for point in points]
        # query_result = [Document(**doc) for doc in query_result]
        return query_result
