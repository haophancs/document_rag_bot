from typing import Any, Dict, Optional, Tuple, Union

from langchain_core.language_models.chat_models import BaseChatModel

from document_rag_bot.services.chat_models.manager import ChatModelManager
from document_rag_bot.services.chat_models.settings import ChatDeploymentConfig
from document_rag_bot.settings import settings


def get_chat_client(
    deployment_id: Optional[str] = None,
    deployment_name: Optional[str] = None,
    temperature: Optional[float] = 0.0,
    seed: Optional[int] = 42,
    logprobs: Optional[bool] = None,
    logit_bias: Optional[Dict[int, int]] = None,
    **kwargs: Any,
) -> Union[BaseChatModel, Tuple[BaseChatModel, ChatDeploymentConfig]]:
    """
    Retrieve the chat model client based on the provided chat model name.

    :param deployment_id: ID of the chat model for text generation
    :param deployment_name: Name of the chat model for text generation
    :param temperature: Temperature param for text generation.
    :param seed: Random seed param for text generation.
    :param logprobs: Whether to return logprobs.
    :param logit_bias: Modify the likelihood of specified tokens appearing.
    :returns: An instance of the specified chat model client.
    :raises KeyError: If the chat model name is not found in the available models.
    """
    kwargs["temperature"] = temperature
    kwargs["seed"] = seed
    kwargs["logprobs"] = logprobs
    kwargs["logit_bias"] = logit_bias

    return ChatModelManager(settings=settings.chat_model).get_chat_model(
        deployment_id=deployment_id,
        deployment_name=deployment_name,
        **kwargs,
    )
