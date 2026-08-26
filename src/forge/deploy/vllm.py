from pathlib import Path


def generate_k8s_manifests(
    model_name: str, adapter_path: str | Path, output_dir: str | Path
) -> str:
    """Generate Kubernetes manifests to deploy the model on vLLM.

    Args:
        model_name: Name of the deployment and service.
        adapter_path: Path to the safetensors adapter or merged model.
        output_dir: Where to save the generated YAML files.

    Returns:
        Path to the generated YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simple deployment manifest for vLLM
    # In a real environment, adapter_path would need to be mounted via PVC or downloaded from S3/HF
    manifest = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {model_name}-vllm
  labels:
    app: {model_name}
    component: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {model_name}
  template:
    metadata:
      labels:
        app: {model_name}
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args: [
          "--model", "/data/model",
          "--served-model-name", "{model_name}",
          "--tensor-parallel-size", "1"
        ]
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: model-volume
          mountPath: /data/model
          readOnly: true
      volumes:
      - name: model-volume
        hostPath:
          path: {Path(adapter_path).absolute()}
          type: DirectoryOrCreate
---
apiVersion: v1
kind: Service
metadata:
  name: {model_name}-service
spec:
  selector:
    app: {model_name}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
"""

    manifest_path = output_dir / f"{model_name}-deployment.yaml"
    manifest_path.write_text(manifest, encoding="utf-8")

    return str(manifest_path)
