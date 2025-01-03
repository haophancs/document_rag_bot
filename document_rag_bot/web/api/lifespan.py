from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict

from fastapi import FastAPI
from FlagEmbedding import FlagReranker
from prometheus_fastapi_instrumentator.instrumentation import (
    PrometheusFastApiInstrumentator,
)

from document_rag_bot.db.qdrant import (
    QdrantManager,
    get_qdrant_manager,
)
from document_rag_bot.settings import settings


def setup_prometheus(app: FastAPI) -> None:  # pragma: no cover
    """
    Enables prometheus integration.

    :param app: current application.
    """
    PrometheusFastApiInstrumentator(should_group_status_codes=False).instrument(
        app,
    ).expose(app, should_gzip=True, name="prometheus_metrics")


@asynccontextmanager
async def lifespan_setup(
    app: FastAPI,
) -> AsyncGenerator[None, None]:  # pragma: no cover
    """
    Actions to run on application startup.

    This function uses fastAPI app to store data
    in the state, such as db_engine.

    :param app: the fastAPI application.
    :return: function that actually performs actions.
    """

    app.middleware_stack = None
    setup_prometheus(app)
    app.middleware_stack = app.build_middleware_stack()

    app.reranker: FlagReranker = FlagReranker(  # type: ignore
        model_name_or_path=settings.reranker_model_name,
        use_fp16=True,
        trust_remote_code=True,
    )
    app.qdrant_managers: Dict[str, QdrantManager] = get_qdrant_manager()  # type: ignore

    yield
