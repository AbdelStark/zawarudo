# LeWorldModel integration guide

The real Push-T runtime described here is **implemented** as `LeWMRuntime`
(`src/wmcp_jepa_service/runtime_lewm.py`, the `lewm` backend). This guide is the authoritative recipe
for how it was built — and how to repeat the integration for a new checkpoint or world-model backend.

## 1. Pin source dependencies

Pin exact commits for:

- `https://github.com/lucas-maes/le-wm`
- `https://github.com/galilai-group/stable-worldmodel`
- any `stable-pretraining` dependency required by the selected LeWM commit

The upstream LeWorldModel repository indicates Python 3.10 and uses `stable-worldmodel[train,env]`. Keep a dedicated image for this runtime rather than mixing arbitrary research dependencies into a generic server image.

## 2. Prepare model package

The HF checkpoint contains `config.json` and `weights.pt`. Build a trusted model package with:

```text
manifest.json
config.json
weights.pt or converted safe artifact
preprocessing.json
action_space.json
normalizers/action_scaler.json
checksums.txt
```

Do not accept weights or model code from user inference requests.

## 3. Implement `LeWMRuntime`

Create a backend class implementing the `WorldModelBackend` protocol.

Pseudo-flow:

```python
class LeWMRuntime:
    def __init__(self, package_path: str, device: str = "cuda"):
        self.model = load_pinned_lewm_model(package_path)
        self.model.to(device).eval()
        freeze(self.model)
        self.preprocessor = load_preprocessor(package_path)
        self.action_scaler = load_action_scaler(package_path)

    async def score(self, request):
        obs = self.preprocessor.decode_observation(request.inputs["observation_history"])
        goal = self.preprocessor.decode_observation(request.inputs["goal"])
        actions = self.action_scaler.transform(load_tensor(request.inputs["action_candidates"]))
        with torch.inference_mode():
            costs = self.model.get_cost({"pixels": obs, "goal": goal}, actions)
        return format_score_response(costs)
```

## 4. Golden validation

Before exposing the service:

1. Run upstream `eval.py` or a minimal upstream scoring path.
2. Run the service backend with the same tensors.
3. Assert cost shape and numerical tolerance.
4. Verify action normalizer and bounds.
5. Verify device placement and no gradients.

## 5. Performance work

Initial tuning knobs:

- flatten candidate dimension into batch where compatible;
- `torch.inference_mode()`;
- optional AMP/float16 after numerical check;
- `torch.compile` after correctness;
- request/candidate batching;
- reuse goal encodings within a plan request;
- cache static model/projector state;
- avoid returning large inline latent arrays.
