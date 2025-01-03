# document_rag_bot/services/chat_models/settings.py
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, SecretStr


class ChatDeploymentConfig(BaseModel):
    """Configuration for a specific model."""

    deployment_id: str
    deployment_name: str
    context_length: Optional[int] = None


class ChatAzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI API."""

    api_key: SecretStr
    api_version: str = "2024-02-15-preview"
    endpoint: str
    deployments: Dict[str, ChatDeploymentConfig]


class ChatOpenAICompatibleConfig(BaseModel):
    """Configuration for OpenAI Compatible API."""

    api_key: SecretStr
    base_url: Optional[str] = None
    deployments: Dict[str, ChatDeploymentConfig]


class ChatOllamaConfig(BaseModel):
    """Configuration for Ollama API."""

    base_url: str = "http://localhost:11434"
    deployments: Dict[str, ChatDeploymentConfig]


class ChatModelSettings(BaseModel):
    """Settings for chat model configuration."""

    enabled: bool = True

    azure_openai: Dict[str, ChatAzureOpenAIConfig] = Field(
        default_factory=dict,
        description="Multiple Azure OpenAI configurations",
    )
    openai_compatible: Dict[str, ChatOpenAICompatibleConfig] = Field(
        default_factory=dict,
        description="Multiple OpenAI Compatible configurations",
    )
    ollama: Dict[str, ChatOllamaConfig] = Field(
        default_factory=dict,
        description="Multiple Ollama configurations",
    )

    @property
    def available_deployment_configs(self) -> Dict[str, ChatDeploymentConfig]:
        """Get all available models grouped by API type."""
        configs = {}
        for provider_configs in (
            self.azure_openai,
            self.openai_compatible,
            self.ollama,
        ):
            for _, provider_config in provider_configs.items():  # type: ignore
                for model_config in provider_config.deployments.values():
                    if model_config.deployment_id in configs:
                        raise ValueError(
                            f"Model ID collision: {model_config.deployment_id}",
                        )
                    configs[model_config.deployment_id] = model_config
        return configs

    @property
    def available_provider_configs(
        self,
    ) -> Dict[
        str,
        Union[
            ChatAzureOpenAIConfig,
            ChatOpenAICompatibleConfig,
            ChatOllamaConfig,
        ],
    ]:
        """Get all available models grouped by API type."""
        configs = {}
        for provider_configs in (
            self.azure_openai,
            self.openai_compatible,
            self.ollama,
        ):
            for _, provider_config in provider_configs.items():  # type: ignore
                for model_config in provider_config.deployments.values():
                    if model_config.deployment_id in configs:
                        raise ValueError(
                            f"Model ID collision: {model_config.deployment_id}",
                        )
                    configs[model_config.deployment_id] = provider_config
        return configs
