import random
from typing import Any, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama.chat_models import ChatOllama
from langchain_openai.chat_models import AzureChatOpenAI, ChatOpenAI

from document_rag_bot.services.chat_models.settings import (
    ChatAzureOpenAIConfig,
    ChatDeploymentConfig,
    ChatModelSettings,
    ChatOllamaConfig,
    ChatOpenAICompatibleConfig,
)


class ChatModelManager:
    """Overall manager for chat models usage."""

    def __init__(self, settings: ChatModelSettings) -> None:
        """
        Initialize ChatModelManager with settings.

        :param settings: Settings for managing chat model configurations
        """
        self.settings = settings

    def _get_chat_azure_openai_client(
        self,
        deployment_config: ChatDeploymentConfig,
        provider_config: ChatAzureOpenAIConfig,
        **kwargs: Any,
    ) -> AzureChatOpenAI:
        """Create a client for Azure OpenAI chat models."""
        return AzureChatOpenAI(
            api_version=provider_config.api_version,
            api_key=provider_config.api_key.get_secret_value(),  # type: ignore
            azure_endpoint=provider_config.endpoint,
            azure_deployment=deployment_config.deployment_name,
            **kwargs,
        )

    def _get_chat_openai_compatible_client(
        self,
        deployment_config: ChatDeploymentConfig,
        provider_config: ChatOpenAICompatibleConfig,
        **kwargs: Any,
    ) -> ChatOpenAI:
        """Create a client for OpenAI-compatible chat models."""
        return ChatOpenAI(
            model=deployment_config.deployment_name,  # type: ignore
            openai_api_key=provider_config.api_key.get_secret_value(),
            openai_api_base=(
                provider_config.base_url if provider_config.base_url else None
            ),
            **kwargs,
        )

    def _get_chat_ollama_client(
        self,
        deployment_config: ChatDeploymentConfig,
        provider_config: ChatOllamaConfig,
        **kwargs: Any,
    ) -> ChatOllama:
        """Create a client for Ollama chat models."""
        return ChatOllama(
            base_url=provider_config.base_url,
            model=deployment_config.deployment_name,
            num_ctx=deployment_config.context_length or 32768,
            keep_alive="-1m",
            # **kwargs,
        )

    def get_chat_model(
        self,
        deployment_id: Optional[str] = None,
        deployment_name: Optional[str] = None,
        return_config: bool = False,
        **kwargs: Any,
    ) -> Union[BaseChatModel, Tuple[BaseChatModel, ChatDeploymentConfig]]:
        """Get a chat model instance based on the model name."""
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

        if isinstance(provider_config, ChatAzureOpenAIConfig):
            client: BaseChatModel = self._get_chat_azure_openai_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        elif isinstance(provider_config, ChatOpenAICompatibleConfig):
            client = self._get_chat_openai_compatible_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        else:
            client = self._get_chat_ollama_client(
                deployment_config,
                provider_config,
                **kwargs,
            )

        if return_config:
            return client, deployment_config

        return client
