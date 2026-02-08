"""FastAPI application for EchoHarvester WebUI."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from echoharvester.config import Config, get_config, load_config
from echoharvester.db import close_db, get_db
from echoharvester.pipeline import get_orchestrator

logger = logging.getLogger(__name__)

# Get the package directory
PACKAGE_DIR = Path(__file__).parent.parent
WEB_DIR = PACKAGE_DIR / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


def create_app(config: Config | None = None) -> FastAPI:
    """Create FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("Starting EchoHarvester API")
        if config:
            app.state.config = config
        else:
            app.state.config = get_config()

        app.state.db = await get_db()
        app.state.orchestrator = await get_orchestrator()

        yield

        # Shutdown
        logger.info("Shutting down EchoHarvester API")
        await close_db()

    app = FastAPI(
        title="EchoHarvester",
        description="Korean ASR Training Data Pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware (for local development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Setup templates
    templates = None
    if TEMPLATES_DIR.exists():
        templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    # Include API routes
    from echoharvester.api.routes import dashboard, pipeline, sources, training, websocket

    app.include_router(sources.router, prefix="/api/sources", tags=["sources"])
    app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
    app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

    # HTML pages (3-page layout: Home, Process, Review)
    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        """Home page (sources + overview)."""
        if templates:
            return templates.TemplateResponse("home.html", {"request": request})
        return HTMLResponse(content=get_fallback_html("Home"), status_code=200)

    @app.get("/process", response_class=HTMLResponse)
    async def process_page(request: Request):
        """Process page (pipeline control)."""
        if templates:
            return templates.TemplateResponse("process.html", {"request": request})
        return HTMLResponse(content=get_fallback_html("Process"), status_code=200)

    @app.get("/review", response_class=HTMLResponse)
    async def review_page(request: Request):
        """Review page (explore + transcribe)."""
        if templates:
            return templates.TemplateResponse("review.html", {"request": request})
        return HTMLResponse(content=get_fallback_html("Review"), status_code=200)

    @app.get("/training", response_class=HTMLResponse)
    async def training_page(request: Request):
        """Training page (setup, monitor, results)."""
        if templates:
            return templates.TemplateResponse("training.html", {"request": request})
        return HTMLResponse(content=get_fallback_html("Training"), status_code=200)

    # Backward-compatible redirects
    @app.get("/sources")
    async def sources_redirect():
        return RedirectResponse(url="/", status_code=301)

    @app.get("/pipeline")
    async def pipeline_redirect():
        return RedirectResponse(url="/process", status_code=301)

    @app.get("/explore")
    async def explore_redirect():
        return RedirectResponse(url="/review", status_code=301)

    @app.get("/transcribe")
    async def transcribe_redirect(request: Request):
        # Preserve query params for deep linking
        query = str(request.url.query)
        target = "/review" + ("?" + query if query else "")
        return RedirectResponse(url=target, status_code=301)

    return app


def get_fallback_html(title: str) -> str:
    """Generate fallback HTML when templates not found."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EchoHarvester - {title}</title>
        <style>
            body {{ font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #333; }}
            .nav {{ margin-bottom: 20px; }}
            .nav a {{ margin-right: 15px; color: #0066cc; }}
        </style>
    </head>
    <body>
        <div class="nav">
            <a href="/">Home</a>
            <a href="/process">Process</a>
            <a href="/review">Review</a>
            <a href="/training">Training</a>
        </div>
        <h1>EchoHarvester - {title}</h1>
        <p>Templates not found. Please create HTML templates in the web/templates directory.</p>
        <p>API is available at <a href="/docs">/docs</a></p>
    </body>
    </html>
    """


# For direct uvicorn usage
app = None


def get_app():
    """Get or create the FastAPI app for uvicorn."""
    global app
    if app is None:
        try:
            config = load_config(Path("config.yaml"))
        except FileNotFoundError:
            from echoharvester.config import Config
            config = Config()
            config.setup()
        app = create_app(config)
    return app
