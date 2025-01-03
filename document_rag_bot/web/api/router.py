import ast
import contextlib
import json
from io import StringIO
from typing import Any, Dict, List, Type

import pandas as pd
from fastapi import (
    APIRouter,
    Body,
    File,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from document_rag_bot.db.qdrant import QdrantManager, create_qdrant_collection
from document_rag_bot.db.schema import DocumentSchema
from document_rag_bot.services.embedding_models.dependencies import embed_documents
from document_rag_bot.settings import settings
from document_rag_bot.workflow import run_workflow

api_router = APIRouter()


def parse_csv(
    file_content: bytes,
    validation_schema: Type[BaseModel],
) -> List[Dict[str, Any]]:
    """
    Parse CSV content into a list of dictionaries using Pandas.

    :param file_content: The content of the CSV file.

    :param validation_schema: The pydantic model as schema for validation.

    :returns: A list of dictionaries representing the CSV rows.
    """
    decoded_content = file_content.decode("utf-8")
    df = pd.read_csv(StringIO(decoded_content)).fillna("")
    for col in df.columns:
        with contextlib.suppress(ValueError, SyntaxError):
            df[col] = df[col].copy().apply(ast.literal_eval)
    if not validation_schema:
        return df.to_dict(orient="records")
    return [validation_schema(**it).model_dump() for it in df.to_dict(orient="records")]


if not settings.api_db_readonly:

    @api_router.post("/upload-documents")
    async def upload_documents(
        request: Request,
        documents_file: UploadFile = File(...),
    ) -> JSONResponse:
        """
        API endpoint to upload and store documents.

        :param request: FastAPI Request.

        :param documents_file: The uploaded file containing documents

        :returns: JSONResponse containing the result of the upload operation
        """
        # Parse and validate documents
        documents = [
            DocumentSchema(**document)
            for document in parse_csv(await documents_file.read(), DocumentSchema)
        ]

        embed_names = {
            deployment_config.deployment_name
            for deployment_config in (
                settings.embedding_model.available_deployment_configs.values()
            )
        }.difference(
            set.intersection(
                *map(
                    set,
                    [list(document.embeddings.keys()) for document in documents],
                ),
            ),
        )
        document_texts = [str(document) for document in documents]
        for deployment_name in embed_names:
            embeddings = embed_documents(
                texts=document_texts,
                deployment_name=deployment_name,
                verbose=True,
            )
            # Update documents with embeddings
            for document, embedding in zip(documents, embeddings):
                document.embeddings[deployment_name] = embedding
            logger.info(
                f"Successfully created embeddings for {len(documents)} "
                f"documents with {deployment_name}",
            )

        # Upsert documents with their embeddings
        qdrant_managers: Dict[str, QdrantManager] = request.app.qdrant_managers
        create_qdrant_collection(qdrant_managers[DocumentSchema.collection_name])
        qdrant_managers[DocumentSchema.collection_name].upload(
            documents,  # type: ignore
            batch_size=settings.qdrant_upload_batch_size,
            max_retries=settings.qdrant_upload_max_retries,
            parallel=settings.qdrant_upload_parallel,
            wait=True,
        )

        return JSONResponse(content={"message": "documents uploaded successfully"})

    @api_router.delete("/delete-documents")
    async def delete_documents(
        request: Request,
        filter_conditions: Dict[str, Any] = Body(
            default_factory=dict,
            description="Filtering conditions for points to delete.",
        ),
    ) -> JSONResponse:
        """
        API endpoint to delete documents.

        :param request: FastAPI Request.

        :param filter_conditions: Filtering conditions for points to delete.

        :returns: JSONResponse containing the result of the delete operation.
        """
        # Delete from Qdrant
        request.app.qdrant_managers[DocumentSchema.collection_name].delete(
            filter_conditions,
        )

        return JSONResponse(content={"message": "All documents deleted successfully"})


@api_router.get("/health")
def health_check() -> JSONResponse:
    """Checks the health of a project."""
    return JSONResponse(content={"detail": "OK"})


@api_router.get("/settings")
def get_settings() -> JSONResponse:
    """Checks current API settings."""
    return JSONResponse(content=json.loads(settings.json()))


@api_router.post("/filter-documents")
async def filter_documents(
    request: Request,
    filter_conditions: Dict[str, Any] = Body(
        default_factory=dict,
        description="Filtering conditions for points to delete.",
    ),
    offset: int = Query(default=0, ge=0, description="Offset results"),
    limit: int = Query(default=10, ge=1, le=50, description="Limit number of results"),
) -> JSONResponse:
    """
    API endpoint to get documents.

    :param request: FastAPI Request.

    :param filter_conditions: Filtering conditions for points to delete.

    :param limit: Limit number of results

    :param offset: Offset results

    :returns: JSONResponse containing the result of the get operation
    """
    qdrant_managers: Dict[str, QdrantManager] = request.app.qdrant_managers
    return JSONResponse(
        content=[
            DocumentSchema(**record.payload).model_dump(  # type: ignore
                exclude={"embeddings"},
            )
            for record in qdrant_managers[DocumentSchema.collection_name].get(
                filter_conditions=filter_conditions,
                offset=offset,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        ],
    )


@api_router.post("/run-workflow")
async def api_run_workflow(
    request: Request,
    messages: List[Dict[str, Any]] = Body(
        default_factory=list,
        description="List of messages",
    ),
    ask_follow_up: bool = Query(default=True),
    return_eval_data: bool = Query(default=False),
) -> JSONResponse:
    """
    API endpoint to run a workflow based on selected messages.

    :param request: FastAPI Request.

    :param messages: Full content of JSON list of selected messages.

    :param ask_follow_up: Whether to allow follow-up asking.

    :param return_eval_data: Whether to return data for evaluation.

    :returns: JSONResponse containing the result of the workflow execution.
    """
    result = run_workflow(
        input_data={"messages": messages, "ask_follow_up": ask_follow_up},
        app=request.app,
    )["result"]
    if not return_eval_data:
        return JSONResponse(content={"response": result["response"]})
    return JSONResponse(
        content={
            "response": result["response"],
            "eval_data": result["eval_data"],
        },
    )
