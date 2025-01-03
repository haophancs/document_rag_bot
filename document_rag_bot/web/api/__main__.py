import os
import shutil
from pathlib import Path

import uvicorn

from document_rag_bot.settings import settings


def set_multiproc_dir() -> None:
    """
    Sets mutiproc_dir env variable.

    This function cleans up the multiprocess directory
    and recreates it. These actions are required by prometheus-client
    to share metrics between processes.

    After cleanup, it sets two variables.
    Uppercase and lowercase because different
    versions of the prometheus-client library
    depend on different environment variables,
    so I've decided to export all needed variables,
    to avoid undefined behaviour.
    """
    shutil.rmtree(settings.prometheus_dir, ignore_errors=True)
    Path(settings.prometheus_dir).mkdir(parents=True)
    os.environ["prometheus_multiproc_dir"] = str(  # noqa: SIM112
        settings.prometheus_dir.expanduser().absolute(),
    )
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = str(
        settings.prometheus_dir.expanduser().absolute(),
    )


def main() -> None:
    """Entrypoint of the application."""
    set_multiproc_dir()
    if settings.api_reload:
        uvicorn.run(
            "document_rag_bot.web.api.application:get_app",
            workers=settings.api_workers_count,
            host=settings.api_host,
            port=settings.api_port,
            reload=settings.api_reload,
            log_level=settings.log_level.value.lower(),
            factory=True,
        )
    else:
        uvicorn.run(
            "document_rag_bot.web.api.application:get_app",
            workers=settings.api_workers_count,
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            log_level=settings.log_level.value.lower(),
            factory=True,
        )
        # TODO: worker killed when using flag embedding
        # We choose gunicorn only if reload
        # option is not used, because reload
        # feature doesn't work with gunicorn workers.
        # GunicornApplication(
        #     "document_rag_bot.web.api.application:get_app",
        #     host=settings.api_host,
        #     port=settings.api_port,
        #     workers=settings.api_workers_count,
        #     timeout=settings.api_worker_timeout,
        #     factory=True,
        #     accesslog="-",
        #     loglevel=settings.log_level.value.lower(),
        #     access_log_format='%r "-" %s "-" %Tf',
        # ).run()


if __name__ == "__main__":
    main()
