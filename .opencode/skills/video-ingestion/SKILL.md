---
name: video-ingestion
description: Guidance for reading walking videos reliably while preserving timestamps, frame metadata, orientation, timing, and provenance.
compatibility: opencode
metadata:
  domain: computer-vision
---

# Video ingestion

Use this skill when implementing or reviewing video input.

## Requirements

Preserve:

- original filename or source identifier;
- file hash when experiment provenance requires it;
- codec/container metadata where available;
- nominal frame rate;
- frame count;
- duration;
- image width and height;
- rotation/orientation;
- per-frame timestamps;
- frame index.

Do not assume that:

- frame rate is always exact;
- every frame is readable;
- frame timestamps are perfectly uniform;
- videos have already been rotated correctly;
- metadata and decoded content always agree.

## Pipeline behavior

Video decoding must produce an explicit frame record containing at least:

- frame index;
- timestamp;
- image or image reference;
- decoding status.

Failures and dropped frames must be observable.

Do not silently renumber decoded frames in a way that destroys their
relationship with source timestamps.

Separate video decoding from pose estimation.
