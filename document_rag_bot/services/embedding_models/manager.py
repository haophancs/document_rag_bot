import random
from typing import Any, Optional, Tuple, Union

from langchain_core.embeddings import Embeddings
from langchain_core.embeddings.fake import FakeEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings, OpenAIEmbeddings

from document_rag_bot.services.embedding_models.locals import TfidfEmbeddings
from document_rag_bot.services.embedding_models.settings import (
    EmbeddingAzureOpenAIConfig,
    EmbeddingDeploymentConfig,
    EmbeddingModelSettings,
    EmbeddingOllamaConfig,
    EmbeddingOpenAICompatibleConfig,
)


class EmbeddingModelManager:
    """Overall manager for embedding models usage."""

    def __init__(self, settings: EmbeddingModelSettings) -> None:
        """
        Initialize EmbeddingModelManager with settings.

        :param settings: Settings for managing embedding model configurations
        """
        self.settings = settings

    def _get_embedding_azure_openai_client(
        self,
        deployment_config: EmbeddingDeploymentConfig,
        provider_config: EmbeddingAzureOpenAIConfig,
        **kwargs: Any,
    ) -> AzureOpenAIEmbeddings:
        """Create embeddings for Azure OpenAI models."""
        return AzureOpenAIEmbeddings(
            api_version=provider_config.api_version,
            api_key=provider_config.api_key.get_secret_value(),  # type: ignore
            azure_endpoint=provider_config.endpoint,
            azure_deployment=deployment_config.deployment_name,
            **kwargs,
        )

    def _get_embedding_openai_compatible_client(
        self,
        deployment_config: EmbeddingDeploymentConfig,
        provider_config: EmbeddingOpenAICompatibleConfig,
        **kwargs: Any,
    ) -> OpenAIEmbeddings:
        """Create embeddings for OpenAI-compatible models."""
        return OpenAIEmbeddings(
            dimensions=deployment_config.dimensions,
            model=deployment_config.deployment_name,
            openai_api_key=provider_config.api_key.get_secret_value(),  # type: ignore
            openai_api_base=(
                provider_config.base_url if provider_config.base_url else None
            ),
            **kwargs,
        )

    def _get_embedding_ollama_client(
        self,
        deployment_config: EmbeddingDeploymentConfig,
        provider_config: EmbeddingOllamaConfig,
        **kwargs: Any,
    ) -> OllamaEmbeddings:
        """Create a client for Ollama chat models."""
        return OllamaEmbeddings(
            base_url=provider_config.base_url,
            model=deployment_config.deployment_name,
            # **kwargs,
        )

    def _get_local_embeddings(
        self,
        deployment_config: EmbeddingDeploymentConfig,
        **kwargs: Any,
    ) -> Embeddings:
        """Create fake embeddings for testing."""
        if deployment_config.deployment_name not in {"fake", "tfidf"}:
            raise ValueError(
                f"Invalid local embedding: {deployment_config.deployment_name}. "
                "Only 'fake' or 'tfidf' supported",
            )

        if deployment_config.deployment_name == "tfidf":
            if not deployment_config.file_path:
                raise ValueError(
                    "Vectorizer file path must not be None for local TF-IDF embedding",
                )
            return TfidfEmbeddings(vectorizer_path=deployment_config.file_path)

        return FakeEmbeddings(
            size=deployment_config.dimensions if deployment_config.dimensions else 768,
        )

    def get_embedding_model(
        self,
        deployment_id: Optional[str] = None,
        deployment_name: Optional[str] = None,
        return_config: bool = False,
        **kwargs: Any,
    ) -> Union[Embeddings, Tuple[Embeddings, EmbeddingDeploymentConfig]]:
        """Get an embedding model instance based on the model name."""
        if deployment_id:
            deployment_config = self.settings.available_deployment_configs.get(
                deployment_id,
            )
        elif deployment_name:
            deployment_id = random.choice(  # TODO: improve load balancing. # noqa: S311
                [
                    _id
                    for _id, cf in self.settings.available_deployment_configs.items()
                    if cf.deployment_name == deployment_name
                ],
            )
            deployment_config = self.settings.available_deployment_configs.get(
                deployment_id,
            )
        else:
            raise ValueError(
                "One of params 'deployment_id' or 'deployment_name' must be specified",
            )

        if deployment_config is None:
            raise ValueError(f"Unavailable model config for model ID: {deployment_id}")

        provider_config = self.settings.available_provider_configs.get(deployment_id)
        if provider_config is None:
            raise ValueError(
                f"Unavailable provider config for model ID: {deployment_id}",
            )

        if isinstance(provider_config, EmbeddingAzureOpenAIConfig):
            client: Embeddings = self._get_embedding_azure_openai_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        elif isinstance(provider_config, EmbeddingOpenAICompatibleConfig):
            client = self._get_embedding_openai_compatible_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        elif isinstance(provider_config, EmbeddingOllamaConfig):
            client = self._get_embedding_ollama_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        else:
            client = self._get_local_embeddings(deployment_config, **kwargs)

        if return_config:
            return client, deployment_config
        return client
