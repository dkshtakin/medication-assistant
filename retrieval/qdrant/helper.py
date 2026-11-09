from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from huggingface_hub import snapshot_download, hf_hub_download
from qdrant_client import QdrantClient
import logging
from torch import device, cuda

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantHelper:
    def __init__(self):
        self.device = device("cuda" if cuda.is_available() else "cpu")
        logger.info(f"device: {self.device}")

        self.dense_embeds_name = "google/embeddinggemma-300m"
        self.embeds_local_dir = "embeddings/embeddinggemma-300m"
        self.sparse_embeds_name = "Qdrant/BM25"

        # first download
        first_download = False
        if first_download:
            _ = snapshot_download(
                repo_id=self.dense_embeds_name,
                local_dir=self.embeds_local_dir,
                ignore_patterns=["*.bin", "onnx/", "openvino/"],
            )

        query_encode_prefix = "task: search result | query: "
        doc_encode_prefix = "title: none | text: "
        logger.info(f"Embeddings doc_encode_prefix: {doc_encode_prefix}")
        logger.info(f"Embeddings query_encode_prefix: {query_encode_prefix}")
        self.dense_embeddings = SentenceTransformer(
            model_name_or_path=self.embeds_local_dir,
            prompts={"query": query_encode_prefix, "document": doc_encode_prefix},
            device=self.device,
            local_files_only=True,
        )
        self.sparse_embeddings = SparseTextEmbedding(model_name=self.sparse_embeds_name)
        logger.info("Embeddings initialized")

        self.client = QdrantClient(path="retrieval/qdrant/qdrant_data")
        logger.info("QdrantClient initialized")
