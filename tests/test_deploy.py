from pathlib import Path
from unittest.mock import patch

import pytest

from forge.deploy.ollama import deploy_to_ollama
from forge.deploy.vllm import generate_k8s_manifests
from forge.export.engine import _export_to_onnx, _export_to_safetensors
from forge.export.gguf_converter import export_to_gguf


def test_ollama_deploy_modelfile_generation(tmp_path: Path):
    gguf_file = tmp_path / "model.gguf"
    gguf_file.write_bytes(b"dummy gguf")

    # Mock subprocess.run so it doesn't actually try to call ollama CLI
    with patch("subprocess.run") as mock_run:
        deploy_to_ollama("test-model", gguf_file)

        modelfile = tmp_path / "Modelfile"
        assert modelfile.exists()
        content = modelfile.read_text(encoding="utf-8")
        assert f"FROM {gguf_file.absolute()}" in content
        assert "TEMPLATE" in content

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args == ["ollama", "create", "test-model", "-f", str(modelfile)]


def test_vllm_k8s_manifests(tmp_path: Path):
    adapter = tmp_path / "model.safetensors"
    out_dir = tmp_path / "deployments"

    manifest_path = generate_k8s_manifests("test-model", adapter, out_dir)

    assert Path(manifest_path).exists()

    content = Path(manifest_path).read_text(encoding="utf-8")
    assert "kind: Deployment" in content
    assert "kind: Service" in content
    assert "name: test-model-vllm" in content
    assert "vllm/vllm-openai" in content


def test_export_routing(tmp_path: Path):
    # Test internal mock exporters
    assert _export_to_onnx(tmp_path).endswith("model.onnx")
    assert _export_to_safetensors(tmp_path).endswith("model.safetensors")

    # Test gguf exporter
    gguf_path = export_to_gguf(tmp_path, "q4_k_m")
    assert gguf_path.endswith("model-q4_k_m.gguf")
    assert Path(gguf_path).exists()
