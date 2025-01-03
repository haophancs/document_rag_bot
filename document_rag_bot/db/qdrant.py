from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    Record,
    ScoredPoint,
    VectorParams,
)
from qdrant_client.models import (
    Filter,
    Fusion,
    FusionQuery,
    Prefetch,
)

from document_rag_bot.db.schema import DocumentSchema
from document_rag_bot.settings import settings


class VectorConfig(BaseModel):
    """Configuration for a named vector."""

    name: str
    size: int
    distance: Distance = Distance.COSINE


class QdrantManager:
    """Manager class for Qdrant vector DB operations with multi-vector support."""

    def __init__(self, collection_name: str, **kwargs: Any) -> None:
        """
        Initialize Qdrant manager.

        :param collection_name: Name of Qdrant collection to operate
        :param kwargs: kwargs params to feed to Qdrant client constructor
        """
        self.client = QdrantClient(**kwargs)
        self.collection_name = collection_name

    def create_collection(self, vector_configs: List[VectorConfig]) -> None:
        """
        Create a new collection with multiple vector configurations.

        :param vector_configs: List of vector configurations
        """
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)

        if exists:
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                config.name: VectorParams(
                    size=config.size,
                    distance=config.distance,
                )
                for config in vector_configs
            },
        )
        logger.info(
            "Created collection {0} with vector configs: {1}".format(
                self.collection_name,
                ", ".join(
                    f"{config.name} ({config.size} dims)" for config in vector_configs
                ),
            ),
        )

    def upload(
        self,
        objects: List[BaseModel],
        vector_field: str = "embeddings",
        **kwargs: Any,
    ) -> None:
        """
        upload objects with their named embeddings into Qdrant.

        :param objects: List of Pydantic models with embeddings
        :param vector_field: Name of the field containing the embeddings dictionary
        :param kwargs: kwargs params to feed to Qdrant upload_points method
        """
        points = []
        for obj in objects:
            obj_dict = obj.model_dump()
            embeddings = obj_dict.pop(vector_field, {})

            if not embeddings:
                continue

            point = PointStruct(
                id=obj_dict["id"],
                payload=obj_dict,
                vector=embeddings,  # Now supports multiple named vectors
            )
            points.append(point)

        if points:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
                **kwargs,
            )
            logger.info(f"uploaded {len(points)} points to {self.collection_name}")

    def hybrid_search(
        self,
        named_vectors: Dict[str, List[float]],
        fusion_type: Fusion = Fusion.RRF,
        filter_conditions: Optional[Dict[str, Any]] = None,
        offset: int = 0,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[ScoredPoint]:
        """
        Perform hybrid search across multiple named vectors.

        :param named_vectors: Dictionary mapping vector names to either:
        :param fusion_type: Type of fusion to use (RRF or DBSF)
        :param filter_conditions: Optional filtering conditions
        :param offset: Offset number of results to return
        :param limit: Maximum number of results to return
        :param with_payload: Whether to include payload in results
        :param with_vectors: Whether to include vectors in results

        :returns: List of search results
        """
        search_filter = Filter(**filter_conditions) if filter_conditions else None

        # Create prefetch queries for each named vector
        prefetch_queries = []
        for vector_name, vector_data in named_vectors.items():
            prefetch_queries.append(
                Prefetch(
                    query=vector_data,
                    using=vector_name,
                    filter=search_filter,
                ),
            )

        # Perform hybrid search
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(fusion=fusion_type),
            with_payload=with_payload,
            with_vectors=with_vectors,
            offset=offset,
            limit=limit,
        )

        return results.points  # type: ignore

    def delete(self, filter_conditions: Dict[str, Any]) -> None:
        """
        Delete points from collection based on filter conditions.

        :param filter_conditions: Filtering conditions for points to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(**filter_conditions),
            wait=True,
        )
        logger.info(f"Deleted points from {self.collection_name}")

    def get(
        self,
        filter_conditions: Dict[str, Any],
        offset: int = 0,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> List[Record]:
        """
        Get points from collection based on filter conditions.

        :param filter_conditions: Filtering conditions for points to delete
        :param offset: Offset number of results to return
        :param limit: Maximum number of results to return
        :param with_payload: Whether to include payload in results
        :param with_vectors: Whether to include vectors in results
        """
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(**filter_conditions),
            offset=offset,
            limit=limit,
            with_payload=with_payload,
            with_vectors=with_vectors,
        )
        return results[0]

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.

        :returns: Dictionary with collection information
        """
        return self.client.get_collection(self.collection_name).model_dump()


def create_qdrant_collection(qdrant_manager: QdrantManager) -> None:
    """Create Qdrant collections based on the defined Qdrant manager."""
    vector_configs: Dict[str, VectorConfig] = {}
    for (
        deployment_config
    ) in settings.embedding_model.available_deployment_configs.values():
        vector_configs[deployment_config.deployment_name] = VectorConfig(
            name=deployment_config.deployment_name,
            size=deployment_config.dimensions,
        )
    qdrant_manager.create_collection(list(vector_configs.values()))


def get_qdrant_manager() -> Dict[str, QdrantManager]:
    """Get dictionary of Qdrant manager instance by specific collection."""
    return {
        schema.collection_name: QdrantManager(
            collection_name=schema.collection_name,
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
            api_key=settings.qdrant_api_key,
        )
        for schema in (DocumentSchema,)
    }
