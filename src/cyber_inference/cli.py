"""
Command-line interface for Cyber-Inference.

Provides commands for:
- Starting the server
- Managing models
- Configuration
- Database operations
"""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from rich.console import Console

from cyber_inference.core.logging import get_logger, log_startup_banner, setup_logging

app = typer.Typer(
    name="cyber-inference",
    help="Cyber-Inference: Edge Inference Server Management",
    add_completion=False,
)

console = Console()
logger = get_logger(__name__)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8337, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
    log_level: str = typer.Option("debug", "--log-level", "-l", help="Log level (debug, info, warning, error)"),
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory path"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", "-m", help="Models directory path"),
) -> None:
    """
    Start the Cyber-Inference server.

    This command initializes and runs the FastAPI server with:
    - Web GUI on /
    - V1 API on /v1/
    - Admin API on /admin/
    """
    import logging

    # Map string log level to logging constant
    log_levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    level = log_levels.get(log_level.lower(), logging.DEBUG)

    # Determine data directory
    if data_dir is None:
        data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Determine models directory
    if models_dir is None:
        models_dir = Path.cwd() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging with file output
    log_dir = data_dir / "logs"
    setup_logging(log_dir=log_dir, level=level)

    # Display startup banner
    log_startup_banner()

    logger.info(f"[highlight]Starting Cyber-Inference server[/highlight]")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Data directory: {data_dir}")
    logger.info(f"  Models directory: {models_dir}")
    logger.info(f"  Log level: {log_level}")
    logger.info(f"  Auto-reload: {reload}")

    # Set environment variables for the app
    import os
    os.environ["CYBER_INFERENCE_DATA_DIR"] = str(data_dir)
    os.environ["CYBER_INFERENCE_MODELS_DIR"] = str(models_dir)

    # Run the server
    uvicorn.run(
        "cyber_inference.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        access_log=True,
    )


@app.command()
def init(
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", "-d", help="Data directory path"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", "-m", help="Models directory path"),
) -> None:
    """
    Initialize Cyber-Inference directories and database.

    Creates necessary directories and initializes the SQLite database.
    """
    setup_logging()
    logger.info("[highlight]Initializing Cyber-Inference[/highlight]")

    # Determine directories
    if data_dir is None:
        data_dir = Path.cwd() / "data"
    if models_dir is None:
        models_dir = Path.cwd() / "models"

    # Create directories
    directories = [
        data_dir,
        data_dir / "logs",
        models_dir,
        Path.cwd() / "bin",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"  Created directory: {directory}")

    # Initialize database
    from cyber_inference.core.database import init_database

    async def _init_db():
        await init_database(data_dir / "cyber-inference.db")

    asyncio.run(_init_db())

    logger.info("[success]Initialization complete![/success]")


@app.command()
def install_llama(
    platform: Optional[str] = typer.Option(None, "--platform", "-p", help="Target platform (auto-detect if not specified)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall even if already installed"),
) -> None:
    """
    Install or update llama.cpp server binary.

    Downloads and installs the appropriate llama.cpp binary for the current platform.
    """
    setup_logging()
    logger.info("[highlight]Installing llama.cpp[/highlight]")

    from cyber_inference.services.llama_installer import LlamaInstaller

    async def _install():
        installer = LlamaInstaller()
        await installer.install(platform=platform, force=force)

    asyncio.run(_install())


@app.command()
def download_model(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')"),
    filename: Optional[str] = typer.Option(None, "--filename", "-f", help="Specific file to download"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", "-m", help="Models directory path"),
) -> None:
    """
    Download a model from HuggingFace.
    """
    setup_logging()
    logger.info(f"[highlight]Downloading model: {model_id}[/highlight]")

    if models_dir is None:
        models_dir = Path.cwd() / "models"

    from cyber_inference.services.model_manager import ModelManager

    async def _download():
        manager = ModelManager(models_dir=models_dir)
        await manager.download_model(model_id, filename=filename)

    asyncio.run(_download())


@app.command()
def list_models(
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", "-m", help="Models directory path"),
) -> None:
    """
    List all downloaded models.
    """
    setup_logging()

    if models_dir is None:
        models_dir = Path.cwd() / "models"

    from cyber_inference.services.model_manager import ModelManager

    async def _list():
        manager = ModelManager(models_dir=models_dir)
        models = await manager.list_models()

        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return

        console.print("\n[bold bright_green]Downloaded Models:[/bold bright_green]\n")
        for model in models:
            size_gb = model.get("size_bytes", 0) / (1024 ** 3)
            console.print(f"  [bright_blue]{model['name']}[/bright_blue]")
            console.print(f"    Path: {model['path']}")
            console.print(f"    Size: {size_gb:.2f} GB")
            console.print()

    asyncio.run(_list())


@app.command()
def version() -> None:
    """
    Display version information.
    """
    from cyber_inference import __version__

    log_startup_banner()
    console.print(f"\n  Version: [bright_green]{__version__}[/bright_green]")
    console.print(f"  Python: [bright_blue]3.12+[/bright_blue]")
    console.print(f"  License: [yellow]GPLv3[/yellow]\n")


if __name__ == "__main__":
    app()

