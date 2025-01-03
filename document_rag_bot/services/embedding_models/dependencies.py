import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from langchain_core.embeddings import Embeddings
from loguru import logger

from document_rag_bot.services.embedding_models.manager import EmbeddingModelManager
from document_rag_bot.services.embedding_models.settings import (
    EmbeddingDeploymentConfig,
)
from document_rag_bot.settings import settings


def _adjust_lists(lists: List[List[float]], desired_length: int) -> List[List[float]]:
    return np.array(
        [
            np.pad(
                np.array(lst),
                (0, max(0, desired_length - len(lst))),
                constant_values=0,
            )[:desired_length]
            for lst in lists
        ],
    ).tolist()


def get_embedding_client(
    deployment_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    **kwargs: Any,
) -> Union[Embeddings, Tuple[Embeddings, EmbeddingDeploymentConfig]]:
    """
    Retrieve the embedding model client based on the provided model name.

    :param deployment_id: ID of the embedding model to retrieve.
    :param deployment_name: Name of the embedding model to retrieve.
    :returns: An instance of the specified embedding model client.
    :raises KeyError: If the embedding model name is not found in the available models.
    """
    return EmbeddingModelManager(settings=settings.embedding_model).get_embedding_model(
        deployment_id=deployment_id,
        deployment_name=deployment_name,
        **kwargs,
    )


def embed_query(
    text: str,
    deployment_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
) -> List[float]:
    """
    Generates an embedding vector for a single query text.

    :param deployment_id: ID of the embedding model to retrieve.
    :param deployment_name: Name of the embedding model to retrieve.
    :param text: The query text to be embedded.

    :returns: A list of floating-point numbers representing the embedding vector.
    """
    client, deployment_config = get_embedding_client(  # type: ignore
        deployment_id=deployment_id,
        deployment_name=deployment_name,
        return_config=True,
    )
    return _adjust_lists([client.embed_query(text)], deployment_config.dimensions)[0]


def embed_documents(
    texts: List[str],
    deployment_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    verbose: bool = True,
) -> List[List[float]]:
    """
    Generates embedding vectors for a list of documents.

    :param deployment_id: ID of the embedding model to retrieve.
    :param deployment_name: Name of the embedding model to retrieve.
    :param texts: A list of document texts to be embedded.
    :param verbose: Determine to print log or not.

    :returns: A list of lists, where each inner list contains floating-point numbers
        representing the embedding vector of a document.
    """
    all_embeddings: List[List[float]] = []

    client, deployment_config = get_embedding_client(  # type: ignore
        deployment_id=deployment_id,
        deployment_name=deployment_name,
        return_config=True,
    )

    batch_size = deployment_config.batch_size
    batch_delay = deployment_config.batch_delay

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = client.embed_documents(batch)

        all_embeddings.extend(batch_embeddings)
        if verbose:
            logger.info(
                "Document embedding creation progress "
                f"({deployment_config.deployment_name}): "
                f"{len(all_embeddings)}/{len(texts)}",
            )
        if len(all_embeddings) < len(texts):
            time.sleep(batch_delay)  # Delay after processing each batch

        if verbose:
            logger.info(
                f"Create embeddings for {len(texts)} with "
                f"{deployment_config.deployment_name} successfully",
            )
    return _adjust_lists(all_embeddings, deployment_config.dimensions)
