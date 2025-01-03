from typing import Any, Dict

from langchain_core.runnables import RunnableConfig


def invoke(config: RunnableConfig, **state: Any) -> Dict[str, Any]:
    """
    Dummy invoker for conversation analyst node.

    :param config: The instance of workflow config.
    :param state: A dictionary containing the current state of the workflow.
    :return: An updated state dictionary with instructions for the next workflow step.
    """
    return {
        "should_stop": False,
        "query_summary": state["messages"][-1]["content"],
    }
