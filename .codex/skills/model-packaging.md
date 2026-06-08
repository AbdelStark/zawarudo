---
name: model-packaging
description: The trusted model-package format and safe-loading policy for real checkpoints (Push-T LeWorldModel and future V-JEPA models) — manifest, config, weights, preprocessing, action normalizers, checksums — plus the hard rule that weights/model code are never accepted from inference requests. Activate when preparing a checkpoint, defining the manifest schema, or reviewing model-loading code for safety.
prerequisites: HF checkpoint access; torch for conversion
---

# Model Packaging & Safe Loading

<purpose>
Define how a checkpoint becomes a trusted, reproducible package the runtime can load — and the security
boundary around model loading.
</purpose>

<context>
- Package contents (`LEWM_INTEGRATION_GUIDE.md` §2): `manifest.json`, `config.json`,
  `weights.pt` (or a converted safe artifact), `preprocessing.json`, `action_space.json`,
  `normalizers/action_scaler.json`, `checksums.txt`.
- Manifest schema: `schemas/model-manifest.schema.json`. `ModelMetadata` (schemas.py) surfaces
  `model_family=jepa`, `model_type=action_conditioned_world_model`, `task=pusht`,
  `latent_space.dimension=192`, input shapes, and `limits` (max_batch 8, max_candidates 1024,
  max_horizon 64, max_planner_iterations 10).
- Security rule (PRD non-goal #6, guide §2): NEVER accept or execute weights/model code from an
  inference request. Only load from a trusted, checksum-verified package.
- Reproducibility: pin upstream commits (le-wm, stable-worldmodel) and the artifact revision.
</context>

<procedure>
1. Obtain the HF checkpoint (`config.json` + `weights.pt`).
2. Convert to a safe artifact where possible (e.g. safetensors); record `checksums.txt`.
3. Author `manifest.json` conforming to `schemas/model-manifest.schema.json` (id, revision, shapes,
   limits, preprocessing + action-space references).
4. Capture `preprocessing.json` + `normalizers/action_scaler.json` from the upstream training config.
5. Verify on load: checksums match, shapes match manifest, device placement, model frozen, no grads.
6. Pin the package revision in deployment config; never float.
</procedure>

<patterns>
<do>
— Verify checksums before loading; fail closed on mismatch.
— Keep preprocessing + action normalization in the package, versioned with the weights.
</do>
<dont>
— Don't load pickled weights from untrusted sources without conversion/verification.
— Don't read model artifacts referenced by a client request payload.
</dont>
</patterns>

<troubleshooting>
| Symptom | Cause | Fix |
|---------|-------|-----|
| Action outputs out of range | wrong/missing action scaler | ship `normalizers/action_scaler.json`, verify bounds |
| Shape mismatch on load | manifest ≠ weights | regenerate manifest from the actual checkpoint |
| Non-reproducible results | floating upstream/artifact rev | pin commits + package revision |
</troubleshooting>

<references>
— schemas/model-manifest.schema.json · LEWM_INTEGRATION_GUIDE.md §2 · rfc/0004-model-packaging-runtime.md
— src/wmcp_jepa_service/schemas.py (ModelMetadata) · PRD.md (non-goals)
</references>
</content>
