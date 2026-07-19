# Model archive

One folder per deployed model version. The models are small (~200 KB
quantized FOMO), so the exported files are committed directly — the git
history of this directory doubles as the deployment record.

```
nicla/models/
  v1/
    trained.tflite       # Edge Impulse OpenMV export
    labels.txt           # Edge Impulse OpenMV export
    model_manifest.json  # written by hand at export time
  v2/
    ...
```

## Manifest format

`model_manifest.json` travels with the model onto the device and is
attached verbatim to every upload's metadata as `model_manifest`:

```json
{
  "model_version": "v1",
  "ei_project_version": 1,
  "trained_at": "2026-07-19",
  "notes": "what changed in this training run"
}
```

The manifest is the human-readable label; the ground-truth identity is
`model_hash`, which the firmware computes at boot by hashing the exact
bytes of `trained.tflite` (first 12 hex chars of SHA-256). If a manifest
is ever wrong or missing, the hash still groups uploads by the model
that actually produced them.

## Deploying a version

Copy all three files from the version folder to the root of the Nicla's
flash filesystem (alongside `main.py` and `wifi_config.py`):

- `trained.tflite`
- `labels.txt`
- `model_manifest.json`

Always copy the manifest and the model together so they cannot drift
apart. On the next boot the firmware logs the hash and manifest it
found, and every upload from then on carries both.
