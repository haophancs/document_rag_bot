import operator
import threading
from typing import Annotated, Any, Dict, List, Type, TypedDict

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.pregel import RetryPolicy

from document_rag_bot.workflow.node import Node
from document_rag_bot.workflow.structs import WorkflowConfig
from document_rag_bot.workflow.utils import load_workflow_config


def create_workflow_schema(config: WorkflowConfig) -> Type[TypedDict]:  # type: ignore
    """
    Generate a TypedDict schema for the workflow state based on the given configuration.

    :param config: The WorkflowConfig instance containing metadata and node definitions.
    :returns: A TypedDict representing the state schema of the workflow.
    """
    fields = {
        **config.metadata.__annotations__,  # type: ignore
        **{
            "{0}_success".format(node_config.node_id): Annotated[
                List[bool],
                operator.add,
            ]
            for node_config in config.nodes
        },
    }
    fields.update(
        {
            field_name: eval(field_config.type)  # noqa: S307
            for node_config in config.nodes
            for field_name, field_config in (
                node_config.input_structure if node_config.input_structure else {}
            ).items()
        },
    )
    fields.update(
        {
            field_name: (
                eval(field_config.type)  # noqa: S307
                if not field_config.annotated_op
                else Annotated[
                    eval(field_config.type),  # noqa: S307
                    eval(field_config.annotated_op),  # noqa: S307
                ]
            )
            for node_config in config.nodes
            for field_name, field_config in (
                node_config.output_structure if node_config.output_structure else {}
            ).items()
        },
    )
    return TypedDict("StateSchema", fields, total=False)  # type: ignore


def create_workflow(config: WorkflowConfig) -> StateGraph:
    """
    Create a workflow graph based on the provided configuration.

    :param config: The workflow config containing node settings and graph structure
    :returns: A StateGraph object representing the constructed workflow
    """
    graph = StateGraph(
        create_workflow_schema(config),
        config_schema=WorkflowConfig,
    )
    for node_config in config.nodes:
        graph.add_node(
            node_config.node_id,
            Node(node_config, config.prompt_templates).invoke,  # type: ignore
            retry=RetryPolicy(
                retry_on=RuntimeError,
                max_attempts=node_config.max_retries,
            ),
        )
    for edge in config.graph.edge_list:
        graph.add_edge(edge.from_node, edge.to_node)
    for conditional_edges in config.graph.conditional_edges_list:
        graph.add_conditional_edges(
            conditional_edges.from_node,
            eval(conditional_edges.route_function),  # noqa: S307
            conditional_edges.intermediates,
        )
    graph.set_entry_point(config.graph.entrypoint)
    return graph


def run_workflow(
    input_data: Dict[str, Any],
    app: FastAPI,
) -> Dict[str, Dict[str, Any]]:
    """
    Run the workflow with the given input data.

    :param app: Instance of FastAPI application.
    :param input_data: A dictionary containing the input data for the workflow
    :return: A dictionary containing the results of the workflow
    """
    workflow_config = load_workflow_config()
    workflow = create_workflow(workflow_config).compile(checkpointer=MemorySaver())
    return workflow.invoke(
        input={
            **workflow_config.metadata.model_dump(),
            **input_data,
        },
        config={
            "recursion_limit": max(
                node_config.max_retries for node_config in workflow_config.nodes
            )
            * len(workflow_config.nodes)
            * 2
            + 1,
            "configurable": {
                "app": app,
                "thread_id": threading.get_ident(),
                **workflow_config.model_dump(),
            },
        },
    )
