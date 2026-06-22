# R3DMatch v2 — Installation & Setup

## Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.9+
- REDCINE-X PRO installed (provides REDLine)
  - Expected: `/Applications/REDCINE-X Professional/REDCINE-X PRO.app/Contents/MacOS/REDline`
- R3D source media

## Install

```bash
cd /Users/sfouasnon/Desktop/R3DMatch_v2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install flask numpy pillow scikit-image scipy

# Install the package
pip install -e .
```

## Start the web server

```bash
cd /Users/sfouasnon/Desktop/R3DMatch_v2
source .venv/bin/activate
python -m r3dmatch2.web_app --host 127.0.0.1 --port 5001
```

Open: http://127.0.0.1:5001

## Run from command line (no web server)

```python
import sys
sys.path.insert(0, 'src')
from r3dmatch2.workflow import run_analysis

result = run_analysis(
    '/path/to/R3D/folder',
    out_dir='/path/to/output',
)
```

## Manual ROI format

If auto-detection fails for a clip, create a `manual_rois.json` file:

```json
{
  "G007_A106_0511R9_001": {"cx": 4055.4, "cy": 1279.3, "r": 209.0},
  "G007_B106_0511C3_001": {"cx": 3820.0, "cy": 1100.0, "r": 215.0}
}
```

Pass it to `run_analysis(manual_roi_file='manual_rois.json')` or via the API.

## Key differences from v1

1. **No RED SDK dependency** — all analysis via REDLine subprocess
2. **Harder sphere detection** — 5 sequential gates, no fallbacks
3. **WB PASS criteria** — GM spread (not absolute residual)
4. **Lens metadata** — shown when present, omitted when absent (no "N/A")
5. **Cleaner codebase** — ~800 lines vs 36,000 lines
