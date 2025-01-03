import gc
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from loguru import logger

ANSWER_LIMIT_HIT = "LIMIT_HIT"
ANSWER_OUT_OF_SCOPE = "OUT_OF_SCOPE"


def invoke(config: RunnableConfig, **state: Any) -> Dict[str, Any]:
    """
    Orchestrate the workflow, each step is conditionally executed.

    :param config: The instance of workflow config.
    :param state: A dictionary containing the current state of the workflow.
    :return: An updated state dictionary with instructions for the next workflow step.
    """
    if any(state["answer_generator_success"]):
        config["configurable"].pop("app")
        state["next_nodes"] = ["result_composer"]

    elif any(state["document_reranker_success"]):
        state["next_nodes"] = ["answer_generator"]

    elif any(state["document_retriever_success"]):
        state["next_nodes"] = ["document_reranker"]

    elif any(state["conversation_analyst_success"]):
        if len(state["messages"]) == 1:
            state["query_summary"] = state["messages"][-1]["content"]
        logger.info(f"Query summarized from conversation: {state['query_summary']}")
        if state["should_stop"]:
            state["answer"] = ANSWER_OUT_OF_SCOPE
            state["next_nodes"] = ["result_composer"]
        elif not len(state["query_summary"]):
            state["next_nodes"] = state["conversation_analyst"]
        else:
            state["next_nodes"] = ["document_retriever"]

    elif (
        not state["ask_follow_up"] or len(state["messages"]) <= state["messages_limit"]
    ):
        if not state["ask_follow_up"]:
            state["messages"] = [state["messages"][-1]]
        state["next_nodes"] = ["conversation_analyst"]

    else:
        state["answer"] = ANSWER_LIMIT_HIT
        state["next_nodes"] = ["result_composer"]

    logger.info(
        "next node(s) in the workflow: {0}".format(
            [it.node if isinstance(it, Send) else it for it in state["next_nodes"]],
        ),
    )

    gc.collect()
    return state
