# Research sources

The dossier was prepared from the linked source projects and current public documentation. The most relevant sources are listed here so implementation teams can re-check details before coding.

## Primary model and world-model repositories

1. **LeWorldModel / le-wm** — official implementation for *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*.  
   URL: https://github.com/lucas-maes/le-wm  
   Key details used: Push-T training/evaluation commands, checkpoint loading/conversion flow, LeWM model primitives (`encode`, `predict`, `rollout`, cost), stated speed/architecture claims.

2. **Hugging Face model: quentinll/lewm-pusht** — official pretrained Push-T model.  
   URL: https://huggingface.co/quentinll/lewm-pusht  
   Key details used: model card, files (`config.json`, `weights.pt`), model config fields, action encoder shape, ViT tiny/patch-14/image-224 configuration.

3. **stable-worldmodel** — robotics/world-model environment management and evaluation platform.  
   URL: https://github.com/galilai-group/stable-worldmodel  
   Key details used: Push-T environment support, datasets, MPC/CEM policy interface, HDF5/LanceDB/LeRobot storage support, Python 3.10/source install guidance.

4. **V-JEPA / V-JEPA 2** — broader JEPA-style world model references.  
   URLs: https://github.com/facebookresearch/jepa, https://github.com/facebookresearch/vjepa2, https://ai.meta.com/vjepa/  
   Key details used: JEPA representation-prediction framing, V-JEPA 2 action-conditioned robot control context, future model compatibility.

## Inference, serving, and deployment frameworks

5. **vLLM**  
   URL: https://github.com/vllm-project/vllm and https://docs.vllm.ai/  
   Key details used: PagedAttention, continuous batching, OpenAI-compatible serving, multi-modal and pooling APIs, IO processor plugins, observability flags, plugin caveats.

6. **NVIDIA Triton Inference Server**  
   URL: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/  
   Key details used: multi-framework serving, HTTP/gRPC, dynamic batching, sequence batching, ensembles, Python backend, model analyzer metrics.

7. **Ray Serve**  
   URL: https://docs.ray.io/en/latest/serve/  
   Key details used: dynamic request batching, async serving, replica autoscaling, request and batching metrics.

8. **KServe**  
   URL: https://kserve.github.io/website/docs/  
   Key details used: Open Inference Protocol/KServe V2 APIs, InferenceService and ServingRuntime CRDs, autoscaling/canary/observability features, custom runtimes.

9. **BentoML**  
   URL: https://docs.bentoml.com/  
   Key details used: adaptive batching and Prometheus-compatible metrics.

10. **OpenTelemetry Collector and Prometheus**  
    URLs: https://opentelemetry.io/docs/collector/configuration/ and https://prometheus.io/docs/guides/opentelemetry/  
    Key details used: trace/metric/log pipelines, Prometheus/OTel integration caveats.

11. **TorchServe**  
    URL: https://github.com/pytorch/serve  
    Key detail used: limited-maintenance/archival status, which makes it unsuitable as the recommended core for a new production service.

## Re-validation checklist before implementation

- Pin exact Git commit hashes for `le-wm`, `stable-worldmodel`, and any V-JEPA dependencies.
- Pin the exact Hugging Face model revision for `quentinll/lewm-pusht`.
- Confirm preprocessing and action scaling used by the Push-T policy/eval code.
- Confirm whether `weights.pt` can be safely converted to `safetensors` or loaded only in a trusted build/init container.
- Benchmark Ray Serve and Triton against the actual rollout/score/plan shapes before committing SLOs.
