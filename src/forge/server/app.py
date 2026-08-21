"""FastAPI application factory.

Creates the Forge API server that:
1. Serves REST endpoints for experiment tracking
2. Provides WebSocket streaming for live training metrics
3. Serves the Next.js static dashboard (if built)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from forge import __version__


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — startup/shutdown hooks."""
    # Startup: initialize DB connection
    from forge.tracking import get_db

    app.state.db = get_db()
    yield
    # Shutdown: cleanup
    from forge.tracking import reset_db

    reset_db()


def create_app(
    dashboard_dir: str | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        dashboard_dir: Path to the built Next.js static files (dashboard/out/).
                       If None, tries to auto-detect relative to the package.
        cors_origins: Allowed CORS origins for development mode.
    """
    app = FastAPI(
        title="Forge API",
        description="Experiment tracking and training dashboard API for Forge.",
        version=__version__,
        lifespan=_lifespan,
    )

    # CORS — permissive in dev, restrictive in production
    if cors_origins is None:
        cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    from forge.server.routes.experiments import router as experiments_router
    from forge.server.routes.system import router as system_router

    app.include_router(experiments_router, prefix="/api")
    app.include_router(system_router, prefix="/api")

    # Register WebSocket handler
    from forge.server.ws import register_websocket

    register_websocket(app)

    # Serve static dashboard files (Next.js static export)
    if dashboard_dir is None:
        # Try to find dashboard relative to the project
        candidates = [
            Path("dashboard/out"),
            Path(__file__).parent.parent.parent.parent / "dashboard" / "out",
        ]
        for candidate in candidates:
            if candidate.is_dir():
                dashboard_dir = str(candidate)
                break

    if dashboard_dir and Path(dashboard_dir).is_dir():
        app.mount(
            "/",
            StaticFiles(directory=dashboard_dir, html=True),
            name="dashboard",
        )

    return app
