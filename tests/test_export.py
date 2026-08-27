import pytest
from pathlib import Path
from forge.export.engine import export_model

def test_export_model_raises_not_implemented(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    
    with pytest.raises(NotImplementedError, match="Real GGUF export is not yet implemented"):
        export_model(adapter_dir, "gguf")
        
    with pytest.raises(NotImplementedError, match="Real ONNX export is not yet implemented"):
        export_model(adapter_dir, "onnx")

    with pytest.raises(NotImplementedError, match="Real SafeTensors export is not yet implemented"):
        export_model(adapter_dir, "safetensors")

def test_export_model_mock_works(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    
    out_gguf = export_model(adapter_dir, "gguf", mock=True)
    assert Path(out_gguf).exists()
    assert Path(out_gguf).read_bytes().startswith(b"GGUF")
    
    out_onnx = export_model(adapter_dir, "onnx", mock=True)
    assert Path(out_onnx).exists()
    
    out_safetensors = export_model(adapter_dir, "safetensors", mock=True)
    assert Path(out_safetensors).exists()
