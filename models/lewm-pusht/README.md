# `lewm-pusht` model package (spec)

This directory holds the **reproducible spec** for the Push-T LeWorldModel package — everything except
the weights and the build-time–generated manifest/checksums. The runtime loads only a *trusted,
checksum-verified* package built from these inputs; it never loads weights or model code referenced by
an inference request (PRD non-goal #6).

## Committed (the spec)

| File | What it is |
|---|---|
| `config.json` | Upstream model config (latent_dim 192, action_dim 10, image 224, history 3). |
| `manifest.template.json` | Manifest with everything but `weights.tensors` (filled from the real checkpoint at build). Conforms to `schemas/model-manifest.schema.json`. |
| `preprocessing.json` | Image preprocessing (resize 224, normalization). |
| `action_space.json` | Action dim/layout/bounds. |
| `normalizers/action_scaler.json` | Action normalizer parameters. |
| `sources.lock.json` | Pinned upstream commits + HF revision (run `scripts/pin_sources.py` to fill). |

## Generated at build (git-ignored)

`weights.safetensors`, `manifest.json` (template + real tensor shapes), `checksums.txt`.

## Build

```bash
# 1. Pin upstream sources + HF revision (needs network)
python scripts/pin_sources.py --package models/lewm-pusht

# 2. Download the HF checkpoint (config.json + weights.pt) into a source dir, then:
python scripts/build_model_package.py real \
  --source <hf-download-dir> --out models/lewm-pusht \
  --source-revision <le-wm-commit> --artifact-revision <hf-revision>

# 3. Verify any time
python scripts/build_model_package.py verify --package models/lewm-pusht
```

For CI / loader self-test without real weights, build a tiny synthetic package:

```bash
make package            # -> .artifacts/model-package/lewm-pusht (synthetic, checksum-verified)
```

## RFC-0005 open issues addressed here

- **#1 action scaler/bounds** — `action_scaler.json` defaults to **identity** scaling with explicit
  `[-1, 1]` bounds, flagged as *placeholders pending upstream confirmation*. Replace with the real
  scaler once the upstream Push-T eval normalization is confirmed.
- **#2 safetensors conversion** — the build converts `weights.pt` → `weights.safetensors`; the loader
  **refuses pickled `.pt`** weights unless `allow_pickle=True` is passed explicitly.

> The values in `config.json`, `preprocessing.json`, and the scaler are best-effort until validated
> against the pinned upstream sources (golden validation is issue #5).
