from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger


class Role(str, Enum):
    """Role in message."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """Message model."""

    role: Role
    content: str


class ValidationError(Exception):
    """Custom exception for input validation errors."""


def validate_message(message: Dict[str, Any]) -> Message:
    """
    Validate a single message and convert it to a Message object.

    :param message: Dictionary containing message data

    :returns: Validated message object

    :raises ValidationError: If message format is invalid
    """
    try:
        # Check required fields
        if not isinstance(message, dict):
            raise ValidationError("Message must be a dictionary")

        required_fields = {"role", "content"}
        missing_fields = required_fields - set(message.keys())
        if missing_fields:
            raise ValidationError(f"Missing required fields: {missing_fields}")

        # Validate role
        role = message.get("role")
        if not isinstance(role, str):
            raise ValidationError("Role must be a string")
        try:
            role = Role(role.lower())
        except ValueError:
            raise ValidationError(
                f"Invalid role: {role}. Must be one of {[r.value for r in Role]}",
            )

        # Validate content
        content = message.get("content")
        if not isinstance(content, str):
            raise ValidationError("Content must be a string")
        if not content.strip():
            raise ValidationError("Content cannot be empty")

        return Message(role=role, content=content)

    except ValidationError as err:
        logger.error(f"Message validation failed: {err!s}")
        raise
    except Exception as err:
        logger.error(f"Unexpected error during message validation: {err!s}")
        raise ValidationError(f"Validation failed: {err!s}")


def validate_messages(messages: List[Dict[str, Any]]) -> List[Message]:
    """
    Validate a list of messages.

    :param messages: List of message dictionaries

    :returns: List of validated message objects

    :raises ValidationError: If any message is invalid or if messages list is invalid
    """
    if not isinstance(messages, list):
        raise ValidationError("Messages must be a list")
    if not messages:
        raise ValidationError("Messages list cannot be empty")

    validated_messages = []
    for i, message in enumerate(messages):
        try:
            validated_message = validate_message(message)
            validated_messages.append(validated_message)
        except ValidationError as err:
            raise ValidationError(f"Invalid message at index {i}: {err!s}")

    return validated_messages


def invoke(
    messages: List[Dict[str, Any]],
    ask_follow_up: bool,
    config: Optional[RunnableConfig] = None,
) -> Dict[str, Any]:
    """
    Validate input data of workflow.

    :param messages: List of messages with role (user/assistant) and content (text)
    :param ask_follow_up: Whether to allow follow-up asking
    :param config: The instance of workflow config (optional)

    :returns: Dictionary containing validated input data

    :raises: ValidationError: If input validation fails
    """
    logger.info(f"Starting input validation for {len(messages)} messages")

    try:
        # Validate ask_follow_up
        if not isinstance(ask_follow_up, bool):
            raise ValidationError("ask_follow_up must be a boolean")

        # Validate messages
        validated_messages = validate_messages(messages)
        logger.info("Input validation completed successfully")

        # Convert validated messages back to dictionary format
        messages_dicts = [
            {"role": msg.role.value, "content": msg.content}
            for msg in validated_messages
        ]

        logger.info(
            f"Messages in conversation: {len(messages_dicts)}. "
            f"Follow-up asking: {ask_follow_up}",
        )

        return {
            "messages": messages_dicts,
            "ask_follow_up": ask_follow_up,
        }

    except ValidationError as err:
        logger.error(f"Input validation failed: {err!s}")
        raise
    except Exception as err:
        logger.error(f"Unexpected error during input validation: {err!s}")
        raise ValidationError(f"Validation failed: {err!s}")
