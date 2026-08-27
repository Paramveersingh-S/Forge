import json
from pathlib import Path
import pytest

from forge.governance.bom import MLBOM, generate_bom
from forge.governance.attest import sign_bom, verify_signature

def test_bom_generation(tmp_path: Path):
    # Setup mock adapter directory
    adapter_dir = tmp_path / "test_adapter"
    adapter_dir.mkdir()
    
    # Write a mock config
    config_file = adapter_dir / "forge.yaml"
    config_file.write_text("model:\n  name: my-model\ntraining:\n  method: sft\n")
    
    # Write mock safetensors
    st_file = adapter_dir / "adapter_model.safetensors"
    st_file.write_bytes(b"mock tensors data")
    
    bom = generate_bom(adapter_dir, "base-model/v1")
    
    assert bom.model_name == "test_adapter"
    assert bom.base_model == "base-model/v1"
    assert bom.training_method == "sft"
    assert bom.adapter_hash is not None
    
    json_data = json.loads(bom.to_json())
    assert json_data["version"] == "1.0"
    assert "timestamp" in json_data
    
def test_attestation(tmp_path: Path):
    bom = MLBOM(
        model_name="test",
        base_model="base",
        training_method="test",
        forge_version="test"
    )
    
    key_path = tmp_path / "id_ed25519"
    
    # Sign it
    sig = sign_bom(bom, private_key_path=key_path)
    assert sig.startswith("mock_sig_")
    
    # Verify it
    assert verify_signature(bom, sig, key_path.with_suffix(".pub"))
    
    # Verify bad signature fails
    assert not verify_signature(bom, "mock_sig_badbeef", key_path.with_suffix(".pub"))
