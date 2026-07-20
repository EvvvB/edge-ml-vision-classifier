# Pi model versioning

The weights file that lives here (`yolo26m.pt` today) is too large to
commit, but every deployed model gets a hand-written manifest committed
next to it, named after the weights file:

```
app/inference/models/
  yolo26m.pt             # gitignored weights
  yolo26m.manifest.json  # committed; the deployment record
```

This mirrors the Nicla convention (see `nicla/models/README.md`): the
manifest is the human-readable label, and the ground-truth identity is a
hash the Pi computes at load time from the exact bytes of the weights
(first 12 hex chars of SHA-256). Both are stamped onto every record the
Pi's model processes, as `yolo_model_hash` and `yolo_model_manifest`,
and travel with the upload to the cloud.

## Manifest format

```json
{
  "model_version": "yolo26m-stock",
  "base_model": "yolo26m",
  "trained_at": null,
  "notes": "what changed in this training run"
}
```

`model_version` is what the dashboard displays and groups by. When a
fine-tuned model replaces the stock weights, drop in the new `.pt`,
write a new manifest with a bumped `model_version` (and a real
`trained_at`), and restart the service. If the manifest is ever wrong or
missing, the hash still groups records by the model that actually
produced them.
