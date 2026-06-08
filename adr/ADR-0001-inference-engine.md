# ADR-0001: Choose Ray Serve + PyTorch as the MVP inference engine

## Status

Accepted for MVP. Revisit after Push-T benchmark and Triton/vLLM spikes.

## Context

The project needs to serve an action-conditioned JEPA world model with production-grade API management and observability. The first model is Push-T LeWorldModel, whose runtime primitives are `encode`, `predict`, `rollout`, and goal-conditioned cost evaluation. The workload is a custom tensor-in/tensor-out model with continuous action sequences, image histories, latent embeddings, and planner loops.

Candidate runtimes include vLLM, Ray Serve, Triton Inference Server, KServe, BentoML, TorchServe, ONNX Runtime, and TensorRT.

## Decision

Use **Ray Serve + FastAPI + PyTorch** for the first production-quality demo. Keep the external API WMCP-aligned and isolate runtime execution behind a `WorldModelBackend` interface. Add Triton as a phase-two backend and KServe as the Kubernetes deployment abstraction. Do not use vLLM as the MVP inference core.

## Rationale

1. Ray Serve can run arbitrary Python/PyTorch model code and dynamic batching without forcing the model into a token-generation abstraction.
2. The LeWorldModel service needs custom preprocessing, continuous action tensors, rollout loops, cost functions, and CEM/MPC planning orchestration.
3. Triton is attractive for optimized model operations but may require graph export, explicit shape management, or Python/custom backend work before the model is stable.
4. vLLM is excellent for LLM inference but not directly aligned with world-model rollout/cost APIs.
5. KServe is a deployment/control plane, not the model execution engine.
6. TorchServe is unsuitable for a new greenfield production project due limited maintenance/archival status.

## Consequences

### Positive

- Fastest path to serving the actual Push-T checkpoint.
- Direct use of upstream model code.
- Easy instrumentation around Python call boundaries.
- Clear extension path to distributed GPU serving.
- Keeps WMCP API independent from runtime choice.

### Negative

- Ray Serve may not match Triton’s low-level performance for stable graphs.
- More custom engineering is needed for GPU memory tuning and model analyzer workflows.
- Production deployment requires Ray operational expertise if multi-node scaling is used.

## Follow-up decisions

1. ADR-0002: Model artifact format and safe loading policy.
2. ADR-0003: Binary tensor transport versus JSON/URI transport.
3. ADR-0004: Whether `plan` belongs in core WMCP or an optional extension.
4. ADR-0005: Triton backend adoption criteria.
5. ADR-0006: vLLM plugin spike outcome.
