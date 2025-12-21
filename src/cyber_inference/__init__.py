"""
Cyber-Inference: A web GUI management tool for v1 compatible inference servers.

This package provides:
- Web GUI for managing llama.cpp inference servers
- OpenAI-compatible V1 API endpoints
- Dynamic model management and resource allocation
- Automatic model loading/unloading
- HuggingFace integration for model downloads
"""

__version__ = "0.1.0"
__author__ = "Matthew Rogers"
__license__ = "GPL-3.0"

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)
logger.info(f"Cyber-Inference v{__version__} initializing...")

