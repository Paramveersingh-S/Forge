"""Deployment integrations for Forge.

Provides one-click deployment generators and runners for Ollama, vLLM, and Kubernetes.
"""

from .ollama import deploy_to_ollama
from .vllm import generate_k8s_manifests

__all__ = ["deploy_to_ollama", "generate_k8s_manifests"]
