from typing import Any, Dict, List

from langchain_core.runnables import RunnableConfig

from document_rag_bot.db.qdrant import Fusion, QdrantManager
from document_rag_bot.db.schema import DocumentSchema
from document_rag_bot.services.embedding_models.dependencies import embed_query


def invoke(
    query_summary: str,
    number_k_retriever: int,
    selected_embeddings: List[str],
    rank_fusion_algorithm: str,
    config: RunnableConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter documents based on impacted functions and calculate similarity scores.

    :param query_summary: The processed question or intent extracted from the convo
    :param number_k_retriever: Number of K for retrieving top-k related documents
    :param selected_embeddings: List of selected embedding model IDs.
    :param rank_fusion_algorithm: rrf or dbsf
    :param config: The instance of workflow config
    :returns: A dictionary containing list of retrieved documents
    """
    qdrant_manager: QdrantManager = (
        config["configurable"]
        .get("app")
        .qdrant_managers[DocumentSchema.collection_name]  # type: ignore
    )
    retrieved_points = qdrant_manager.hybrid_search(
        named_vectors={
            deployment_name: embed_query(
                text=query_summary,
                deployment_name=deployment_name,
            )
            for deployment_name in selected_embeddings
        },
        fusion_type=Fusion.RRF if rank_fusion_algorithm == "rrf" else Fusion.DBSF,
        with_vectors=False,
        with_payload=True,
        limit=number_k_retriever,
    )

    return {
        "retrieved_documents": [
            DocumentSchema(**point.payload).model_dump()  # type: ignore
            for point in retrieved_points
        ],
    }
