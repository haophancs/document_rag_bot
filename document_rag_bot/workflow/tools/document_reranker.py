from typing import Any, Dict, List

import numpy as np
from FlagEmbedding import FlagReranker
from langchain_core.runnables import RunnableConfig


def _rerank_documents(
    reranker: FlagReranker,
    query: str,
    documents: List[str],
) -> List[int]:
    pairs = [(query, document) for document in documents]
    return np.argsort(reranker.compute_score(pairs, normalize=True))[::-1].tolist()


def invoke(
    query_summary: str,
    retrieved_documents: List[Dict[str, Any]],
    number_k_reranker: int,
    disable_reranker: bool,
    config: RunnableConfig,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Filter documents based on impacted functions and calculate similarity scores.

    :param query_summary: The processed question or intent extracted from the convo
    :param number_k_reranker: Number of K for retrieving top-k related documents
    :param retrieved_documents: List of retrieved documents
    :param disable_reranker: Whether to disable reranker
    :param config: The instance of workflow config
    :returns: A dictionary containing list of reranked documents
    """
    if disable_reranker:
        return {
            "reranked_documents": retrieved_documents,
            "relevant_documents": retrieved_documents,
        }
    reranker = config["configurable"].get("app").reranker  # type: ignore
    reranked_indices = _rerank_documents(
        reranker=reranker,
        query=query_summary,
        documents=[document["text"] for document in retrieved_documents],
    )
    reranked_documents = [retrieved_documents[idx] for idx in reranked_indices]

    _temp: Dict[str, List[str]] = {
        document["url"]: [] for document in reranked_documents[:number_k_reranker]
    }
    for document in reranked_documents:
        if document["url"] not in _temp:
            continue
        _temp[document["url"]].append(document["text"])

    relevant_documents = [
        {
            "text": "\n\n".join(_temp[document_url]),
            "url": document_url,
        }
        for document_url in _temp
    ]

    return {
        "reranked_documents": reranked_documents,
        "relevant_documents": relevant_documents,
    }
