# Architecture

R3DMatch is organized around a staged calibration workflow rather than a single monolithic solve.

## High-Level Flow

```text
source discovery / ingest
-> clip identity and grouping
-> frame decode or mock sample generation
-> shared ROI / neutral sampling
-> gray and chroma measurement
-> strategy comparison
-> trust classification
-> run assessment
-> report + payload generation
-> optional read-only camera verification
-> optional later writeback workflow
```

## Major Boundaries

### Analysis

The analysis layer is responsible for:

- clip discovery
- frame sampling
- neutral-target measurement
- exposure and chromaticity diagnostics
- confidence and stability metrics

### Reporting

The reporting layer converts measurement truth into:

- contact-sheet and lightweight review artifacts
- per-camera trust summaries
- run-level gating
- debug exposure traces
- operator-facing recommendations

### Camera Control

The camera-control layer is intentionally separated from review:

- primary live transport uses RED RCP2 over WebSocket JSON on port `9998`
- read-only state queries and verification are safe default flows
- writeback remains explicit and separate from analysis

### Native / External Tooling

Optional external components are isolated behind clear boundaries:

- RED SDK bridge for real decode when `--backend red` is used
- REDLine for preview rendering
- legacy raw TCP RCP2 fallback for debugging only

The RED SDK is expected to live outside the repo:

- `RED_SDK_ROOT` is the authoritative SDK root
- `RED_SDK_REDISTRIBUTABLE_DIR` may override the runtime redistributable path
- the native bridge should be rebuilt with `scripts/build_red_sdk_bridge.sh` after SDK changes
- repo-local `src/RED_SDK/...` payloads are not part of the runtime contract

## Why The Boundaries Matter

- calibration logic stays testable with the mock backend
- operator trust can be evaluated before any later camera push
- transport and verification logic can evolve without rewriting the measurement pipeline
