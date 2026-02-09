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
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level (debug, info, warning, error)"),
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

    # Configure uvicorn loggers to suppress verbose debug messages
    # Always use 'info' for uvicorn to prevent WebSocket/connection debug spam
    # Our application logging is configured separately, so this only affects uvicorn's internal logs
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_access_logger = logging.getLogger("uvicorn.access")

    # Set uvicorn loggers to INFO to suppress DEBUG messages
    uvicorn_logger.setLevel(logging.INFO)
    uvicorn_error_logger.setLevel(logging.INFO)
    uvicorn_access_logger.setLevel(logging.INFO if level <= logging.INFO else logging.WARNING)

    # Run the server with uvicorn log level set to 'info' to suppress debug messages
    uvicorn.run(
        "cyber_inference.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",  # Always use 'info' to suppress uvicorn's verbose debug output
        access_log=(level <= logging.INFO),
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
def install_whisper(
    platform: Optional[str] = typer.Option(None, "--platform", "-p", help="Target platform (auto-detect if not specified)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall even if already installed"),
) -> None:
    """
    Install or update whisper.cpp server binary.

    Downloads and installs the appropriate whisper.cpp binary for audio transcription.
    If whisper-server is already in your PATH, it will be used instead.
    """
    setup_logging()
    logger.info("[highlight]Installing whisper.cpp[/highlight]")

    from cyber_inference.services.whisper_installer import WhisperInstaller

    async def _install():
        installer = WhisperInstaller()
        await installer.install(platform_override=platform, force=force)

    asyncio.run(_install())


@app.command()
def download_model(
    model_id: str = typer.Argument(..., help="HuggingFace model ID (e.g., 'TheBloke/Llama-2-7B-GGUF')"),
    filename: Optional[str] = typer.Option(None, "--filename", "-f", help="Specific file to download"),
    engine: str = typer.Option("auto", "--engine", "-e", help="Engine type: auto, gguf, sglang, transformers"),
    models_dir: Optional[Path] = typer.Option(None, "--models-dir", "-m", help="Models directory path"),
) -> None:
    """
    Download a model from HuggingFace.

    Use --engine sglang to download a full HuggingFace model for SGLang inference.
    Use --engine transformers to download a full HuggingFace model for transformers inference.
    Use --engine gguf (or auto) to download a GGUF file for llama.cpp inference.
    """
    setup_logging()
    logger.info(f"[highlight]Downloading model: {model_id}[/highlight]")

    if models_dir is None:
        models_dir = Path.cwd() / "models"

    from cyber_inference.services.model_manager import ModelManager

    async def _download():
        manager = ModelManager(models_dir=models_dir)

        # Determine engine type
        use_engine = engine
        if engine == "auto":
            model_lower = model_id.lower()
            if "gguf" not in model_lower and "whisper" not in model_lower and not filename:
                # Non-GGUF model: prefer transformers, fallback to sglang
                from cyber_inference.services.sglang_manager import SGLangManager
                if SGLangManager.get_instance().is_available():
                    use_engine = "sglang"
                    console.print(
                        "[yellow]Auto-detected non-GGUF repo, using SGLang. "
                        "Use --engine gguf or --engine transformers to override.[/yellow]"
                    )
                else:
                    use_engine = "transformers"
                    console.print(
                        "[yellow]Auto-detected non-GGUF repo, using transformers. "
                        "Use --engine gguf to force GGUF download.[/yellow]"
                    )
            else:
                use_engine = "gguf"

        if use_engine == "sglang":
            console.print(f"[bright_blue]Downloading SGLang model: {model_id}[/bright_blue]")
            path = await manager.download_sglang_model(model_id)
            console.print(f"[bright_green]SGLang model downloaded to: {path}[/bright_green]")
        elif use_engine == "transformers":
            console.print(f"[bright_yellow]Downloading transformers model: {model_id}[/bright_yellow]")
            path = await manager.download_transformers_model(model_id)
            console.print(f"[bright_green]Transformers model downloaded to: {path}[/bright_green]")
        else:
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
            engine = model.get("engine_type", "llama")
            engine_badge = {
                "llama": "[bright_green]GGUF[/bright_green]",
                "whisper": "[green]Whisper[/green]",
                "sglang": "[bright_magenta]SGLang[/bright_magenta]",
                "transformers": "[bright_yellow]Transformers[/bright_yellow]",
            }.get(engine, f"[dim]{engine}[/dim]")

            console.print(f"  [bright_blue]{model['name']}[/bright_blue]  {engine_badge}")
            console.print(f"    Path: {model['path']}")
            console.print(f"    Size: {size_gb:.2f} GB")
            if model.get("quantization"):
                console.print(f"    Quantization: {model['quantization']}")
            console.print()

    asyncio.run(_list())


@app.command()
def install_sglang(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall even if already installed"),
    cuda_version: str = typer.Option("cu130", "--cuda", help="CUDA version for PyTorch wheels (e.g. cu130, cu128)"),
) -> None:
    """
    Install SGLang for GPU-accelerated inference.

    Installs sglang[all], PyTorch with CUDA support, and sgl-kernel
    from the appropriate wheel indices. Requires NVIDIA GPU with CUDA.
    """
    import platform
    import subprocess
    import sys

    setup_logging()
    logger.info("[highlight]Installing SGLang[/highlight]")

    # Check if already installed
    if not force:
        from cyber_inference.services.sglang_manager import SGLangManager
        mgr = SGLangManager.get_instance()
        if mgr.is_available():
            version = mgr.get_version()
            console.print(f"[bright_green]SGLang is already installed (v{version})[/bright_green]")
            console.print("Use --force to reinstall.")
            return

    console.print("[bright_blue]Installing SGLang with CUDA support...[/bright_blue]")
    console.print(f"[dim]CUDA wheel variant: {cuda_version}[/dim]")
    console.print("[dim]This may take several minutes (PyTorch + CUDA dependencies)[/dim]\n")

    def _run(cmd: list[str], label: str) -> bool:
        console.print(f"[bright_blue]{label}[/bright_blue]")
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            console.print(f"[red]Failed: {label}[/red]")
            return False
        return True

    def _get_installed_version(package: str) -> str | None:
        """Read the installed version of a package, stripping any +cpu/+cu suffix."""
        result = subprocess.run(
            [sys.executable, "-m", "uv", "pip", "show", package],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Version:"):
                ver = line.split(":", 1)[1].strip()
                # Strip local version suffix (+cpu, +cu130, etc.)
                return ver.split("+")[0]
        return None

    def _url_exists(url: str) -> bool:
        """Check if a URL exists by doing a HEAD request (follows redirects)."""
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status < 400
        except Exception:
            return False

    try:
        # Step 1: Install sglang with all extras
        if not _run(
            [sys.executable, "-m", "uv", "pip", "install", "sglang[all]>=0.4.6"],
            "Installing sglang[all]...",
        ):
            raise typer.Exit(1)

        # Step 2: Detect versions that sglang's resolver chose
        torch_ver = _get_installed_version("torch")
        kernel_ver = _get_installed_version("sgl-kernel")
        arch = platform.machine()  # aarch64 or x86_64
        console.print(f"[dim]Detected: torch={torch_ver}, sgl-kernel={kernel_ver}, arch={arch}[/dim]")

        # Step 3: Replace PyTorch with CUDA wheels (exact version sglang resolved)
        pytorch_index = f"https://download.pytorch.org/whl/{cuda_version}"
        if torch_ver:
            console.print(f"\n[bright_blue]Installing PyTorch {torch_ver} with {cuda_version}...[/bright_blue]")
            if not _run(
                [
                    sys.executable, "-m", "uv", "pip", "install",
                    "--reinstall", f"torch=={torch_ver}", "torchvision", "torchaudio",
                    "--index-url", pytorch_index,
                ],
                f"Installing PyTorch {torch_ver} with {cuda_version}...",
            ):
                # Fallback: install latest available CUDA torch (no version pin)
                console.print("[yellow]Exact version failed, trying latest from CUDA index...[/yellow]")
                _run(
                    [
                        sys.executable, "-m", "uv", "pip", "install",
                        "--reinstall", "torch", "torchvision", "torchaudio",
                        "--index-url", pytorch_index,
                    ],
                    f"Installing latest PyTorch with {cuda_version}...",
                )
        else:
            console.print("[yellow]Warning: torch not found after sglang install[/yellow]")

        # Step 4: Install sgl-kernel CUDA wheel (version detected dynamically)
        if kernel_ver:
            sgl_kernel_whl = (
                f"https://github.com/sgl-project/whl/releases/download/"
                f"v{kernel_ver}/sgl_kernel-{kernel_ver}+{cuda_version}"
                f"-cp310-abi3-manylinux2014_{arch}.whl"
            )
            console.print(f"\n[bright_blue]Installing sgl-kernel {kernel_ver} ({cuda_version}, {arch})...[/bright_blue]")

            # Verify the wheel exists before attempting download
            if _url_exists(sgl_kernel_whl):
                console.print(f"[dim]Verified: wheel exists at GitHub releases[/dim]")
                if not _run(
                    [sys.executable, "-m", "uv", "pip", "install", "--reinstall", sgl_kernel_whl],
                    f"Installing sgl-kernel {kernel_ver}+{cuda_version}...",
                ):
                    console.print("[yellow]Direct wheel install failed[/yellow]")
            else:
                console.print(f"[yellow]Wheel not found at GitHub releases, using index fallback...[/yellow]")
                _run(
                    [
                        sys.executable, "-m", "uv", "pip", "install",
                        "--reinstall", "sgl-kernel",
                        "--extra-index-url",
                        f"https://docs.sglang.io/whl/{cuda_version}/sgl-kernel/",
                    ],
                    "Installing sgl-kernel from index...",
                )
        else:
            console.print("[yellow]Warning: sgl-kernel not found after sglang install[/yellow]")

        # Verify installation
        from cyber_inference.services.sglang_manager import SGLangManager
        mgr = SGLangManager.get_instance()
        mgr.reset_cache()

        if mgr.is_available():
            version = mgr.get_version()
            console.print(f"\n[bright_green]SGLang installed successfully! (v{version})[/bright_green]")

            cuda_info = mgr.get_cuda_info()
            if cuda_info["cuda_available"]:
                console.print(f"  CUDA devices: {cuda_info['device_count']}")
                for dev in cuda_info["devices"]:
                    console.print(f"    {dev['name']} ({dev['memory_total_mb']} MB)")
            else:
                console.print("[yellow]  Warning: CUDA not available. SGLang requires a CUDA GPU.[/yellow]")
        else:
            console.print("[red]SGLang installation could not be verified.[/red]")

    except Exception as e:
        console.print(f"[red]Installation error: {e}[/red]")
        raise typer.Exit(1)


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

