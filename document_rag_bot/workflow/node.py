import importlib
import json
from typing import Any, Dict, Optional, Type

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, create_model

from document_rag_bot.services.chat_models.dependencies import get_chat_client
from document_rag_bot.workflow.structs import NodeConfig, PromptTemplateConfig


class Node:
    """Represents a node in the workflow that can execute task."""

    def __init__(
        self,
        node_config: NodeConfig,
        template_config: PromptTemplateConfig,
    ) -> None:
        """
        Initialize a Node instance.

        :param node_config: Configuration for the node
        :param template_config: Configuration for prompt templates
        """
        self.config: NodeConfig = node_config
        self.templates: PromptTemplateConfig = template_config
        self.input_model: Optional[Type[BaseModel]] = (
            self._create_model(
                node_config.input_structure,
                "Input",
            )
            if node_config.input_structure
            else None
        )
        self.output_model: Optional[Type[BaseModel]] = (
            self._create_model(
                node_config.output_structure,
                "Output",
            )
            if node_config.output_structure
            else None
        )

        if self.config.chat_model:
            self.prompt_template: ChatPromptTemplate = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(self.templates.system),
                    HumanMessagePromptTemplate.from_template(self.templates.human),
                ],
            )
            self.chat_model: BaseChatModel = get_chat_client(  # type: ignore
                deployment_name=self.config.chat_model,
            )
        if not self.config.chat_model and not self.config.tool:
            raise ValueError(
                f"Node {self.node_id} has neither chat model nor tool configured",
            )

    @property
    def node_id(self) -> str:
        """
        Get the node ID from the configuration.

        :returns: The node ID as a string.
        """
        return self.config.node_id

    @staticmethod
    def _create_model(structure: Dict[str, Any], prefix: str) -> Type[BaseModel]:
        """
        Create a Pydantic model from a dictionary structure.

        :param structure: Dictionary describing the model structure
        :param prefix: prefix to append to the model name
        :returns: A new Pydantic model class
        """
        fields: Dict[str, Any] = {
            name: (
                eval(details.type),  # noqa: S307
                Field(description=details.description),
            )
            for name, details in structure.items()
        }
        return create_model(f"{prefix}Model", **fields)  # type: ignore

    def invoke(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """
        Invoke the node with the given state.

        :param state: Current state of the workflow
        :param config: Configuration settings.
        :returns: Updated state after the node's operation
        :raises ValueError: If the node has neither chat model nor tool configured
        """
        output_data = self._initialize_output()

        try:
            input_data = self._validate_input(state)
            if self.config.chat_model:
                output_data.update(self._process_chat_model(input_data))
            elif self.config.tool:
                output_data.update(self._process_tool(input_data, config))
            else:
                raise ValueError(
                    f"Node {self.node_id} has neither chat model nor tool configured",
                )

            output_data[f"{self.node_id}_success"] = [True]
            return output_data

        except KeyboardInterrupt as err:
            raise err

        except Exception as err:
            raise RuntimeError(f"Error in node {self.node_id}: {err}") from err

    def _initialize_output(self) -> Dict[str, Any]:
        """
        Initialize the output data structure.

        :returns: Initial output data dictionary with success flag set to False
        """
        return {
            f"{self.node_id}_success": [False],
        }

    def _validate_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and process input data using the input model if defined.

        :param state: Current state of the workflow
        :returns: Validated input data
        :raises ValidationError: If input validation fails
        """
        if self.input_model:
            return self.input_model(
                **{
                    k: v for k, v in state.items() if k in self.input_model.model_fields
                },
            ).model_dump()
        return {**state}

    def _format_chat_prompt(self, input_data: Dict[str, Any]) -> Any:
        """
        Format the chat prompt with node configuration and input data.

        :param input_data: Validated input data
        :returns: Formatted chat prompt
        """
        return self.prompt_template.format_messages(
            **{
                k: (
                    json.dumps(
                        {
                            field_name: "type: {}; description: {}".format(
                                field_details["type"],
                                field_details["description"],
                            )
                            for field_name, field_details in v.items()
                        },
                        indent=2,
                    )
                    if k in ["input_structure", "output_structure"]
                    else v
                )
                for k, v in self.config.model_dump().items()
            },
            input_data=json.dumps(input_data, indent=2, ensure_ascii=False),
        )

    def _process_chat_model(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data using the chat model.

        :param input_data: Validated input data
        :returns: Processed output data
        :raises RuntimeError: If LLM processing fails
        """
        assert self.output_model is not None  # noqa: S101
        return (
            self.chat_model.with_structured_output(
                self.output_model,  # type: ignore
            )
            .invoke(self._format_chat_prompt(input_data))
            .model_dump()  # type: ignore
        )

    def _process_tool(
        self,
        input_data: Dict[str, Any],
        config: RunnableConfig,
    ) -> Dict[str, Any]:
        """
        Process input data using the configured tool.

        :param input_data: Validated input data
        :param config: Configuration settings
        :returns: Tool processing output
        :raises RuntimeError: If tool processing fails
        """
        tool = importlib.import_module(self.config.tool)  # type: ignore
        return tool.invoke(**input_data, config=config)
