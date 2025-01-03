from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

from document_rag_bot.workflow.tools.workflow_supervisor import (
    ANSWER_LIMIT_HIT,
    ANSWER_OUT_OF_SCOPE,
)


def invoke(config: RunnableConfig, **state: Any) -> Dict[str, Any]:
    """
    Compose final response of the workflow.

    :param config: The instance of workflow config.
    :param state: A dictionary containing the current state of the workflow.
    :return: An updated state dictionary with instructions for the next workflow step.
    """
    response = {
        "references": [],
        "answer": state["answer"],
    }
    reranked_documents = state.get("reranked_documents", [])

    if response["answer"] == ANSWER_LIMIT_HIT:
        response["answer"] = (
            "The conversation has reached the maximum message limit. "
            "Please clear the chat history or start a new session."
        )
    elif response["answer"] == ANSWER_OUT_OF_SCOPE:
        response["answer"] = (
            "This seems outside the scope of my domain knowledge. "
            "Let me know if you have any questions related to it!"
        )
    else:
        cited_document_ids = state["cited_document_ids"]
        for i in range(1, len(state["relevant_documents"]) + 1):
            if f"[{i}]" in response["answer"]:
                cited_document_ids.append(i)
        cited_document_ids = list(set(cited_document_ids))

        urls = [
            document["url"]
            for i, document in enumerate(state["relevant_documents"], 1)
            if i in cited_document_ids
        ]
        response["references"] = [
            {"indices": indices, "url": val}
            for val, indices in sorted(
                {
                    x: [i for i, v in enumerate(urls, 1) if v == x] for x in set(urls)
                }.items(),
                key=lambda x: x[1][0],
            )
        ]

    return {
        "result": {
            "response": response,
            "eval_data": {
                "user_input": state["messages"][-1]["content"],
                "response": response["answer"],
                "retrieved_contexts": [
                    document["text"] for document in reranked_documents
                ],
            },
        },
    }
