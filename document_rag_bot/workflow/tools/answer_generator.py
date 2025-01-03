from typing import Any, Dict

from langchain_core.runnables import RunnableConfig


def invoke(config: RunnableConfig, **state: Any) -> Dict[str, Any]:
    """
    Dummy invoker for answer generator node.

    :param config: The instance of workflow config.
    :param state: A dictionary containing the current state of the workflow.
    :return: An updated state dictionary with instructions for the next workflow step.
    """
    return {
        "answer": "Just a dummy answer.",
        "cited_document_ids": [1],
    }
