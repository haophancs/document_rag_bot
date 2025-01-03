from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, SecretStr


class EmbeddingDeploymentConfig(BaseModel):
    """Configuration for a specific embedding model."""

    deployment_id: str
    deployment_name: str
    batch_size: int = 100
    batch_delay: int = 60
    dimensions: int
    file_path: Optional[str] = None


class EmbeddingAzureOpenAIConfig(BaseModel):
    """Configuration for Azure OpenAI API."""

    api_key: SecretStr
    api_version: str = "2024-02-15-preview"
    endpoint: str
    deployments: Dict[str, EmbeddingDeploymentConfig]


class EmbeddingOpenAICompatibleConfig(BaseModel):
    """Configuration for OpenAI Compatible API."""

    api_key: SecretStr
    deployments: Dict[str, EmbeddingDeploymentConfig]
    base_url: Optional[str] = None


class EmbeddingOllamaConfig(BaseModel):
    """Configuration for Ollama API."""

    base_url: str = "http://localhost:11434"
    deployments: Dict[str, EmbeddingDeploymentConfig]


class EmbeddingLocalConfig(BaseModel):
    """Configuration for Local Embeddings."""

    deployments: Dict[str, EmbeddingDeploymentConfig]


class EmbeddingModelSettings(BaseModel):
    """Settings for embedding model configuration."""

    azure_openai: Dict[str, EmbeddingAzureOpenAIConfig] = Field(
        default_factory=dict,
        description="Multiple Azure OpenAI configurations",
    )
    openai_compatible: Dict[str, EmbeddingOpenAICompatibleConfig] = Field(
        default_factory=dict,
        description="Multiple OpenAI Compatible configurations",
    )
    ollama: Dict[str, EmbeddingOllamaConfig] = Field(
        default_factory=dict,
        description="Multiple Ollama configurations",
    )
    local: Dict[str, EmbeddingLocalConfig] = Field(
        default_factory=dict,
        description="Local embedding configurations",
    )

    @property
    def available_deployment_configs(self) -> Dict[str, EmbeddingDeploymentConfig]:
        """Get all available models grouped by API type."""
        configs = {}
        for provider_configs in (
            self.azure_openai,
            self.openai_compatible,
            self.ollama,
            self.local,
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
            EmbeddingAzureOpenAIConfig,
            EmbeddingOpenAICompatibleConfig,
            EmbeddingOllamaConfig,
            EmbeddingLocalConfig,
        ],
    ]:
        """Get all available models grouped by API type."""
        configs = {}
        for provider_configs in (
            self.azure_openai,
            self.openai_compatible,
            self.ollama,
            self.local,
        ):
            for _, provider_config in provider_configs.items():  # type: ignore
                for model_config in provider_config.deployments.values():
                    if model_config.deployment_id in configs:
                        raise ValueError(
                            f"Model ID collision: {model_config.deployment_id}",
                        )
                    configs[model_config.deployment_id] = provider_config
        return configs
