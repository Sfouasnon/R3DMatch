# R3DMatch v5

Multi-camera exposure & color alignment for RED KOMODO-X arrays.

R3DMatch measures an 18% gray sphere seen by every camera in a volumetric array,
solves a per-camera correction, and pushes the corrections back to the cameras over RCP2 so
the array matches. Corrections travel with each clip in its R3D
metadata; post-production reproduces them with REDLine's `--useMeta` flag, or with a specialty batching script output during calibration.

- **Exposure** is solved in **scene-linear** and verified prior to output via a closed-loop re-render / verification.
- **White balance** is matched per camera (green/magenta via tint, warm/cool via
  per-camera Kelvin trims toward the working temperature).

**Accuracy:** exposure matched to within **±0.5 IRE** across the array (the
repeatable benchmark), approaching **±0.1 IRE** per camera under ideal lighting —
the floor set by the RED's --exposureAdjust granularity. Every figure in the
report is *measured* from a corrected re-render, not estimated.

---

## Requirements

- **macOS** with **Python 3.9+**
- **REDCINE-X PRO / REDLine** installed (used for all rendering and metadata).
  R3DMatch auto-detects it at the standard install path; set it in
  **Settings → REDLine** if it lives elsewhere.
- Camera network access for the **Capture** and RCP2 **push** steps (cameras
  reachable by IP). Capture's clip ingest uses FTP-over-TLS — set the body
  credentials, a destination folder, and the camera network in **Settings**
  (nothing is assumed; the pull step stays disabled until they're filled).

Python dependencies are pinned in `requirements.txt` / `pyproject.toml`
(PySide6, NumPy, Pillow, OpenCV, scikit-image, SciPy, tifffile, websockets,
Flask).

---

## Setup

Clone the repository, then create a dedicated virtual environment (one time):

```zsh
git clone https://github.com/Sfouasnon/R3DMatch.git
cd R3DMatch
bash setup_venv.command
```

or double-click `setup_venv.command` in Finder. This builds `./.venv`, installs
the app in editable mode, and prints the run command. To pin a specific
interpreter: `PYTHON=/path/to/python3 bash setup_venv.command`.

## Running

From the repository root:

```zsh
source .venv/bin/activate
r3dmatch                 # console entry point
# or:  python -m r3dmatch3.app
```

---

## Workflow

1. **Ingest** — point at the card folder (single frame R3D per camera), or grab the
   frames straight from the array with the **Capture** tab (see *Capture* below).
   Choose the
   matching strategy (Median is the robust default; **18% Gray Anchor** targets an
   absolute gray level — see *Matching strategies* below), the delivery look the
   match is scored through, and the white-balance mode (scene-temp per-camera
   Kelvin is recommended; **Exposure only** skips WB entirely).
2. **Analyze** — renders each clip through REDLine, auto-solves the gray-sphere
   position, and measures chroma + luminance. A scene-linear render is also made
   for the exposure solve.
3. **Assist** — a thumbnail grid of every camera. Auto-solved cameras are green;
   click any camera to open the large solver and place/confirm a sphere. Prev/
   Next cycle the array without returning to the grid.
4. **Match** — review the per-camera corrections and the array-level charts
   (IRE convergence, exposure spread, before/after coherence contact sheet).
5. **Push** — commit the corrections to the cameras over RCP2 (Color Temperature,
   Tint, Exposure Adjust). Configure camera IPs in the Cameras tab first.

The HTML report (written to the output folder) is the chain-of-custody document:
an overview, the IRE-convergence chart, a before/after **Array Coherence**
contact sheet, and a per-camera page with the measured closed-loop result. 
Post renders with --useMeta to inherit the changes.

---

## Capture (optional)

Instead of offloading cards, R3DMatch can acquire the calibration frame directly
from the array over the camera network. The **Capture** tab runs three steps,
driving the same RCP2 protocol (port 9998) used by the push:

1. **Detect** — a TCP probe sweeps the camera network to find the RED KOMODO-X
   bodies. It scans the stage network first, then a `/24` derived from the host's
   own NIC (link-local `/16` ranges are narrowed so a scan never explodes to 65k
   hosts). Detection opens **no** session — it only lists what's reachable.
2. **Synchronized single-frame record** — one persistent RCP2 session per body,
   frame-limit set to 1, sync verified, then `SYNC_RECORD_START` fired at a shared
   timecode a few seconds out so every body grabs the *same* instant. Scales to a
   full array (up to 36 bodies).
3. **Ingest** — the clip just recorded is pulled off each body over **FTP-over-TLS**
   into the destination folder (newest clip only — it never sweeps whole reels),
   staged to `.part` with a size verify.

The ingested frames drop straight into **Ingest → Analyze**, so a full
calibration — capture, solve, and RCP2 push — can run without a card reader.
Credentials, destination, and the camera network are set in **Settings** and are
never assumed.

> **Hardware status:** the capture transport encodes the documented RCP2 sync-record
> flow and the field rules from the proven reference tools, but paths not yet
> confirmed on a live body are marked `# UNVERIFIED` in `capture.py` — validate on
> a camera before relying on them in production.

---

## Matching strategies

The exposure solve offers two ways to choose the array's target level. Both solve
per-camera `exposureAdjust` in scene-linear; they differ only in *what* the array
is driven to.

- **Median** (default) — the target is the robust inlier-median of the measured
  cameras, so the array converges on its own center. Outlier-tolerant and
  independent of any absolute reference; best for general multi-camera matching
  where internal consistency is what matters.
- **18% Gray Anchor** — an *absolute* target. Every camera is driven to a fixed
  scene-linear reflectance rather than the array median. The level is named as a
  **Log3G10 IRE** (default **33.3 IRE = 0.18 scene-linear = an 18% gray**), which
  is the value a DIT reads on a waveform; it is editable on the Setup screen for
  deliberately rating the sphere off 18%. The solve runs entirely in scene-linear,
  so it never touches the display/delivery transform — Log3G10 is only the unit
  the target is entered in. Use it when you need the array pinned to a known,
  repeatable gray level (e.g. a gaffer's reference) rather than to itself.

Because both solve in scene-linear, exposure requires a scene-linear measurement
for every camera; a missing scene-linear render is retried and, if it still can't
be produced, the run stops with a diagnostic rather than silently falling back to
a less accurate display-space solve.

The reported exposure anchor is shown in display IRE (BT.1886) — e.g. an 18% gray
anchor lands near ~44 IRE through IPP2 — with the Log3G10 target noted alongside.

---

## Diagnostic tools (`tools/`)

Run from the same venv. These are read-only and never modify the solver.

- `sphere_doctor.py CLIP.R3D …` — explains why sphere detection passed/failed on
  a clip (every Hough candidate + the gate that rejected it, plus an annotated
  image).
- `sphere_gate_probe.py --profile <id> <footage…>` — validates the detection
  gates against the real ALT candidate stream across footage.
- `scene_linear_match_probe.py --profile <id> <footage…>` — validates
  scene-linear matching (per-channel gains, residuals) before pipeline changes.
- `golden_regression.py` — reference-path identity guard; re-run after any change
  to confirm the proven solve path is byte-identical.

---

## Post-production handover

Each clip's R3D metadata carries the committed correction. Render with
`--useMeta` to reproduce the match:

```zsh
REDline --i /path/to/clip.R3D --useMeta --format 16 --outDir /renders/
```

In an Exposure-only run, only `exposureAdjust` is written; otherwise
`exposureAdjust`, `kelvin`, and `tint`.

---

## Project layout

```
R3DMatch_v5/
├── src/r3dmatch3/        # application package
│   ├── app.py            # PySide6 GUI (entry: r3dmatch / python -m r3dmatch3.app)
│   ├── workflow.py       # render → detect → measure → solve → verify
│   ├── sphere.py         # gray-sphere detection (Hough/ALT + gate pipeline)
│   ├── measure.py        # measurement math (display + scene-linear)
│   ├── solve.py          # exposure (median / 18% gray anchor) + white-balance solve
│   ├── capture.py        # RCP2 synchronized single-frame capture (WebSocket)
│   ├── capture_ftp.py    # single-clip FTP-over-TLS ingest from each body
│   ├── rcp2.py           # RCP2 calibration push (WebSocket)
│   ├── settings.py       # persisted settings (REDLine path, camera net, FTP creds)
│   ├── report.py         # HTML assessment report
│   └── colorpipeline.py  # REDLine color-science configs
├── tools/                # diagnostic + validation scripts
├── requirements.txt      # pinned dependencies
├── pyproject.toml        # package + entry points
└── setup_venv.command    # one-step environment setup
```
