"""
Service modules for Cyber-Inference.

This package contains:
- Process manager for llama.cpp subprocess management
- Model manager for HuggingFace downloads
- Resource monitor for system metrics
"""

from cyber_inference.core.logging import get_logger

logger = get_logger(__name__)

