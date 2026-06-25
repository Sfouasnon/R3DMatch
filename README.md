# R3DMatch v4

Multi-camera exposure & color alignment for RED KOMODO-X arrays.

**Related:** [MediaRunner](https://github.com/Sfouasnon/MediaRunner) — R3D-centric transfer & checksum-verification tool (xxHash128, ASC MHL, R3D metadata scraping).

R3DMatch measures a single 18% gray sphere shot by every camera in an array,
solves a per-camera correction, and pushes it back to the cameras over RCP2 so
the array intercuts cleanly. Corrections travel with each clip in its R3D
metadata; post-production reproduces them with REDLine's `--useMeta` flag.

- **Exposure** is solved in **scene-linear** (the only correct space for a
  stop-based correction) and verified at the sensor via a closed-loop re-render.
- **White balance** is matched per camera (green/magenta via tint, warm/cool via
  per-camera Kelvin trims toward the working temperature).
- An **Exposure-only** mode matches exposure alone and omits all WB from the
  report — for when only luminance alignment is required.

**Accuracy:** exposure matched to within **±0.5 IRE** across the array (the
repeatable benchmark), approaching **±0.1 IRE** per camera under ideal lighting —
the floor set by the camera's own exposureAdjust resolution. Every figure in the
report is *measured* from a corrected re-render, not estimated.

---

## Requirements

- **macOS** with **Python 3.9+**
- **REDCINE-X PRO / REDLine** installed (used for all rendering and metadata).
  R3DMatch auto-detects it at the standard install path; set it in
  **Settings → REDLine** if it lives elsewhere.
- Camera network access for the RCP2 push step (cameras reachable by IP).

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

1. **Ingest** — point at the card folder (one R3D frame per camera). Choose the
   matching strategy (Median is the robust default), the delivery look the
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
R3DMatch_v4/
├── src/r3dmatch3/        # application package
│   ├── app.py            # PySide6 GUI (entry: r3dmatch / python -m r3dmatch3.app)
│   ├── workflow.py       # render → detect → measure → solve → verify
│   ├── sphere.py         # gray-sphere detection (Hough/ALT + gate pipeline)
│   ├── measure.py        # measurement math (display + scene-linear)
│   ├── solve.py          # exposure + white-balance solve
│   ├── rcp2.py           # RCP2 camera push (WebSocket)
│   ├── report.py         # HTML assessment report
│   └── colorpipeline.py  # REDLine color-science configs
├── tools/                # diagnostic + validation scripts
├── requirements.txt      # pinned dependencies
├── pyproject.toml        # package + entry points
└── setup_venv.command    # one-step environment setup
```
