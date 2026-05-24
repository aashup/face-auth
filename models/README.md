# Bundled models

All models here are open-source (Apache-2.0 / MIT). Drop the `.tflite` files in this folder and list
their SHA-256 in `manifest.json` (verified at boot by `src/sync/modelIntegrity.ts`).

| File | Source | License | Purpose |
|------|--------|---------|---------|
| `blazeface.tflite` | MediaPipe | Apache-2.0 | Face detection (bbox) |
| `face_mesh.tflite` | MediaPipe Face Mesh | Apache-2.0 | 468 landmarks (gate + challenge) |
| `minifasnet.tflite` | Silent-Face-Anti-Spoofing | Apache-2.0 | Passive liveness |
| `mobilefacenet.tflite` | MobileFaceNet open weights | Apache/MIT | 128/512-d embedding |

> The same `mobilefacenet.tflite` (and identical preprocessing) MUST be used server-side for enrollment,
> or device embeddings will not be comparable to provisioned templates.

`manifest.json` shape:

```json
{
  "blazeface.tflite":     { "sha256": "...", "version": "1.0.0", "input": [128, 128] },
  "face_mesh.tflite":     { "sha256": "...", "version": "1.0.0", "input": [192, 192] },
  "minifasnet.tflite":    { "sha256": "...", "version": "1.0.0", "input": [80, 80] },
  "mobilefacenet.tflite": { "sha256": "...", "version": "1.0.0", "input": [112, 112], "dims": 512 }
}
```
