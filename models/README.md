# Pose Model Assets

Model files are local runtime inputs and are ignored by Git. Step 2 requires an
explicit MediaPipe `.task` path and never downloads a model during processing.

Use the full Pose Landmarker model recommended for the MVP. MediaPipe's official
Pose Landmarker documentation links the current full model bundle:

1. Open <https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/>.
2. Under **Models**, download **Pose landmarker (Full)**.
3. Place it at `models/pose_landmarker_full.task` without committing it.
4. Record or verify its identity with:

```bash
sha256sum models/pose_landmarker_full.task
stat -c '%s bytes' models/pose_landmarker_full.task
```

Each run records the explicit resolved path identifier, filename, byte size, and
SHA-256 in `pose_metadata.json`. The filename is descriptive, not sufficient
model-version provenance; the hash is authoritative for reproducing the asset.
The file's upstream display name (for example, "Full") is not a semantic model
version. Reproduction should cite the recorded SHA-256, byte size, MediaPipe
library version, and backend configuration rather than inferring a version from
`pose_landmarker_full.task`.

Run from the repository root after the headless installation in `README.md`:

```bash
.venv/bin/python scripts/estimate_pose.py path/to/walk.mp4 \
  --model models/pose_landmarker_full.task --output-root outputs
```
