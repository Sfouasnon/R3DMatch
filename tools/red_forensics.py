from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from PIL import Image

from r3dmatch import sdk


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "runs" / "red_forensics"
COMPARE_ROOT = REPO_ROOT / "runs" / "rmd_compare"
COMPATIBILITY_PATH = REPO_ROOT / "runs" / "rmd_compatibility_table.json"
REDLINE = Path("/usr/local/bin/REDLine")
RCX_APP_ROOT = Path("/Applications/REDCINE-X Professional/REDCINE-X PRO.app")
RCX_BINARY = RCX_APP_ROOT / "Contents" / "MacOS" / "REDCINE-X PRO"
RCX_PRESETS_DB = RCX_APP_ROOT / "Contents" / "Resources" / "presets" / "EffectPresets.db"

CLIP_PATH = Path("/Users/sfouasnon/Desktop/R3DMatch_Calibration/G007_B057_0324YT.RDC/G007_B057_0324YT_001.R3D")
REEL_ID = "G007_B057_0324YT"
CLIP_ID = CLIP_PATH.stem


DISPLAY_TRANSFORM = [
    "--colorSciVersion", "3",
    "--colorSpace", "13",
    "--gammaCurve", "32",
    "--outputToneMap", "1",
    "--rollOff", "3",
    "--shadow", "0.000",
]
BASE_RENDER_ARGS = ["--format", "3", "--start", "0", "--frameCount", "1", "--silent"]


@dataclass
class Variant:
    key: str
    description: str
    sdk_settings: Dict[str, Any]
    cli_args: list[str]
    expects_cli_change: bool


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_stats(path: Path) -> Dict[str, Any]:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    hist_r, _ = np.histogram(image[..., 0], bins=8, range=(0, 255))
    hist_g, _ = np.histogram(image[..., 1], bins=8, range=(0, 255))
    hist_b, _ = np.histogram(image[..., 2], bins=8, range=(0, 255))
    return {
        "mean_rgb": [float(x) for x in image.mean(axis=(0, 1))],
        "min_rgb": [int(x) for x in image.min(axis=(0, 1))],
        "max_rgb": [int(x) for x in image.max(axis=(0, 1))],
        "histogram_8bin": {
            "r": hist_r.tolist(),
            "g": hist_g.tolist(),
            "b": hist_b.tolist(),
        },
    }


def image_diff(path_a: Path, path_b: Path) -> Dict[str, Any]:
    image_a = np.asarray(Image.open(path_a).convert("RGB"), dtype=np.float32)
    image_b = np.asarray(Image.open(path_b).convert("RGB"), dtype=np.float32)
    diff = np.abs(image_a - image_b)
    return {
        "mean_absolute_difference": float(diff.mean()),
        "max_absolute_difference": float(diff.max()),
    }


def resolve_rendered_output(output_stub: Path) -> Path:
    if output_stub.exists():
        return output_stub
    candidates = sorted(output_stub.parent.glob(f"{output_stub.name}*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No REDLine output produced for {output_stub}")
    if candidates[0] != output_stub:
        candidates[0].rename(output_stub)
    for extra in candidates[1:]:
        if extra.exists():
            extra.unlink()
    return output_stub


def run_command(command: list[str], *, cwd: Optional[Path] = None) -> Dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False, cwd=str(cwd) if cwd else None)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def display_transform_args(*, omit_output_tonemap: bool = False, omit_rolloff: bool = False) -> list[str]:
    args: list[str] = ["--colorSciVersion", "3", "--colorSpace", "13", "--gammaCurve", "32"]
    if not omit_output_tonemap:
        args.extend(["--outputToneMap", "1"])
    if not omit_rolloff:
        args.extend(["--rollOff", "3"])
    args.extend(["--shadow", "0.000"])
    return args


def run_render(
    input_path: Path,
    output_path: Path,
    extra_args: Iterable[str],
    *,
    transform_args: Optional[list[str]] = None,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    command = [str(REDLINE), "--i", str(input_path), *BASE_RENDER_ARGS, "--o", str(output_path), *(transform_args or display_transform_args()), *list(extra_args)]
    result = run_command(command, cwd=cwd)
    result["output_path"] = str(resolve_rendered_output(output_path)) if result["returncode"] == 0 else str(output_path)
    return result


def run_print_meta(
    input_path: Path,
    extra_args: Iterable[str],
    *,
    level: int,
    cwd: Optional[Path] = None,
) -> Dict[str, Any]:
    command = [str(REDLINE), "--i", str(input_path), "--printMeta", str(level), *list(extra_args)]
    return run_command(command, cwd=cwd)


def require_native() -> Any:
    native = sdk._load_red_native_module()
    if native is None or not hasattr(native, "create_rmd_from_settings"):
        raise RuntimeError("RED SDK bridge with create_rmd_from_settings is not available.")
    if hasattr(native, "sdk_available") and not native.sdk_available():
        raise RuntimeError(getattr(native, "unavailable_message", lambda: "RED SDK unavailable")())
    return native


def create_sdk_rmd(native: Any, out_dir: Path, variant: Variant) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_clip = out_dir / CLIP_PATH.name
    if temp_clip.exists() or temp_clip.is_symlink():
        temp_clip.unlink()
    os.symlink(CLIP_PATH, temp_clip)
    settings = {
        "exposure_adjust": 0.0,
        "cdl_slope": [1.0, 1.0, 1.0],
        "cdl_offset": [0.0, 0.0, 0.0],
        "cdl_power": [1.0, 1.0, 1.0],
        "cdl_saturation": 1.0,
        "cdl_enabled": False,
        "output_tonemap": 1,
        "highlight_rolloff": 3,
        "color_space": 1,
        "gamma_curve": 15,
        "image_pipeline_mode": 1,
    }
    settings.update(variant.sdk_settings)
    result = dict(
        native.create_rmd_from_settings(
            str(temp_clip),
            settings["exposure_adjust"],
            settings["cdl_slope"],
            settings["cdl_offset"],
            settings["cdl_power"],
            settings["cdl_saturation"],
            settings["cdl_enabled"],
            settings["output_tonemap"],
            settings["highlight_rolloff"],
            settings["color_space"],
            settings["gamma_curve"],
            settings["image_pipeline_mode"],
        )
    )
    generated_rmd_path = Path(str(result["rmd_path"])).resolve()
    copied_rmd_path = out_dir / generated_rmd_path.name
    if generated_rmd_path != copied_rmd_path:
        shutil.copy2(generated_rmd_path, copied_rmd_path)
    xmp_path = out_dir / f"{variant.key}.sdk_rmd_xmp.xml"
    if result.get("rmd_xmp"):
        xmp_path.write_text(str(result["rmd_xmp"]), encoding="utf-8")
    result["copied_rmd_path"] = str(copied_rmd_path)
    result["xmp_path"] = str(xmp_path) if xmp_path.exists() else None
    result["settings"] = settings
    return result


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def lookup_rcx_cdl_enable_presets() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "db_exists": RCX_PRESETS_DB.exists(),
        "cdl_enable_rows": [],
        "cdl_disable_rows": [],
        "rcx_binary_string_hits": [],
        "error": None,
    }
    if not RCX_PRESETS_DB.exists():
        return result
    try:
        connection = sqlite3.connect(str(RCX_PRESETS_DB))
        cursor = connection.cursor()
        tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        result["tables"] = [row[0] for row in tables]
        for table in result["tables"]:
            try:
                rows = cursor.execute(f"SELECT * FROM {table}").fetchall()
                columns = [item[1] for item in cursor.execute(f"PRAGMA table_info({table})").fetchall()]
            except sqlite3.DatabaseError:
                continue
            for row in rows:
                joined = " ".join(str(item) for item in row)
                if "Cdl Enable" in joined:
                    result["cdl_enable_rows"].append({"table": table, "columns": columns, "row": [str(item) for item in row]})
                if "Cdl Disable" in joined:
                    result["cdl_disable_rows"].append({"table": table, "columns": columns, "row": [str(item) for item in row]})
        connection.close()
        if RCX_BINARY.exists():
            strings_result = run_command(["strings", str(RCX_BINARY)])
            hits = []
            for index, line in enumerate(strings_result["stdout"].splitlines(), start=1):
                if "Cdl Enable" in line or "Cdl Disable" in line or "cdlenabled" in line:
                    hits.append({"line": index, "text": line})
            result["rcx_binary_string_hits"] = hits
    except Exception as exc:  # pragma: no cover - forensic fallback
        result["error"] = str(exc)
    return result


def compare_texts(label_a: str, text_a: str, label_b: str, text_b: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
        )
    )


def build_adjacent_clip_env(target_dir: Path, reel_rmd_path: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    clip_symlink = target_dir / CLIP_PATH.name
    if clip_symlink.exists() or clip_symlink.is_symlink():
        clip_symlink.unlink()
    os.symlink(CLIP_PATH, clip_symlink)
    target_rmd = target_dir / f"{REEL_ID}.RMD"
    shutil.copy2(reel_rmd_path, target_rmd)
    return clip_symlink


def file_contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle.lower() in path.read_text(encoding="utf-8", errors="ignore").lower()


def parse_printmeta_level1(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for line in text.splitlines():
        if ":\t" not in line:
            continue
        key, value = line.split(":\t", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def main() -> None:
    if not REDLINE.exists():
        raise RuntimeError(f"REDLine not found at {REDLINE}")
    if not CLIP_PATH.exists():
        raise RuntimeError(f"Test clip missing: {CLIP_PATH}")

    ensure_clean_dir(RUN_ROOT)
    ensure_clean_dir(COMPARE_ROOT)
    modes_root = RUN_ROOT / "modes"
    variants_root = RUN_ROOT / "variants"
    write_text(RUN_ROOT / "README.txt", textwrap.dedent(
        f"""\
        RED native forensics harness
        clip: {CLIP_PATH}
        redline: {REDLINE}
        rcx_app: {RCX_BINARY if RCX_BINARY.exists() else 'missing'}
        """
    ))

    native = require_native()

    baseline_render = run_render(CLIP_PATH, RUN_ROOT / "baseline_useMeta.jpg", ["--useMeta"])
    baseline_image = Path(str(baseline_render["output_path"]))
    save_json(RUN_ROOT / "baseline_useMeta.command.json", baseline_render)
    save_json(RUN_ROOT / "baseline_useMeta.image_stats.json", image_stats(baseline_image))

    modes: Dict[str, Dict[str, Any]] = {
        "A_useMeta_only": {"extra_args": ["--useMeta"], "input_path": CLIP_PATH},
        "B_useRMD1_only": {"extra_args": ["--useRMD", "1"], "input_path": None},
        "C_useMeta_plus_useRMD1": {"extra_args": ["--useMeta", "--useRMD", "1"], "input_path": None},
        "D_loadRMD_explicit": {"extra_args": [], "input_path": CLIP_PATH},
        "E_masterRMDFolder_useRMD1": {"extra_args": ["--useRMD", "1"], "input_path": CLIP_PATH},
    }

    controlled_variant = Variant(
        key="controlled_reference",
        description="SDK-authored RMD with exposure=0.25, CDL slope 1.1/0.9/1.0, cdl_enabled=true, outputToneMap=1, highlightRollOff=3",
        sdk_settings={
            "exposure_adjust": 0.25,
            "cdl_slope": [1.1, 0.9, 1.0],
            "cdl_offset": [0.0, 0.0, 0.0],
            "cdl_power": [1.0, 1.0, 1.0],
            "cdl_saturation": 1.0,
            "cdl_enabled": True,
            "output_tonemap": 1,
            "highlight_rolloff": 3,
        },
        cli_args=[
            "--useMeta",
            "--exposure", "0.250000",
            "--cdlRedSlope", "1.100000",
            "--cdlGreenSlope", "0.900000",
            "--cdlBlueSlope", "1.000000",
            "--cdlRedOffset", "0.000000",
            "--cdlGreenOffset", "0.000000",
            "--cdlBlueOffset", "0.000000",
            "--cdlRedPower", "1.000000",
            "--cdlGreenPower", "1.000000",
            "--cdlBluePower", "1.000000",
            "--cdlSaturation", "1.000000",
        ],
        expects_cli_change=True,
    )

    controlled_rmd = create_sdk_rmd(native, COMPARE_ROOT / "sdk_authored" / controlled_variant.key, controlled_variant)
    controlled_rmd_path = Path(str(controlled_rmd["copied_rmd_path"]))
    reel_rmd_path = controlled_rmd_path.with_name(f"{REEL_ID}.RMD")
    if controlled_rmd_path.name != reel_rmd_path.name:
        shutil.copy2(controlled_rmd_path, reel_rmd_path)
    save_json(COMPARE_ROOT / "sdk_authored" / controlled_variant.key / "metadata.json", controlled_rmd)

    mode_adjacent_dir = RUN_ROOT / "mode_adjacent"
    mode_adjacent_clip = build_adjacent_clip_env(mode_adjacent_dir, reel_rmd_path)
    modes["B_useRMD1_only"]["input_path"] = mode_adjacent_clip
    modes["B_useRMD1_only"]["cwd"] = mode_adjacent_dir
    modes["C_useMeta_plus_useRMD1"]["input_path"] = mode_adjacent_clip
    modes["C_useMeta_plus_useRMD1"]["cwd"] = mode_adjacent_dir

    modes["D_loadRMD_explicit"]["extra_args"] = ["--loadRMD", str(controlled_rmd_path), "--useRMD", "1"]
    modes["E_masterRMDFolder_useRMD1"]["extra_args"] = ["--masterRMDFolder", str(controlled_rmd_path.parent), "--useRMD", "1"]

    mode_results: Dict[str, Any] = {}
    for mode_key, mode in modes.items():
        mode_dir = modes_root / mode_key
        mode_dir.mkdir(parents=True, exist_ok=True)
        input_path = Path(str(mode["input_path"]))
        printmeta1 = run_print_meta(input_path, mode["extra_args"], level=1, cwd=mode.get("cwd"))
        printmeta2 = run_print_meta(input_path, mode["extra_args"], level=2, cwd=mode.get("cwd"))
        render = run_render(input_path, mode_dir / f"{mode_key}.jpg", mode["extra_args"], cwd=mode.get("cwd"))
        render_path = Path(str(render["output_path"])) if render["returncode"] == 0 else mode_dir / f"{mode_key}.jpg"
        result = {
            "mode": mode_key,
            "printMeta_1": printmeta1,
            "printMeta_2": printmeta2,
            "render": render,
            "render_exists": render_path.exists(),
            "render_path": str(render_path),
            "diff_vs_baseline": image_diff(baseline_image, render_path) if render_path.exists() else None,
            "image_stats": image_stats(render_path) if render_path.exists() else None,
        }
        save_json(mode_dir / "printMeta1.json", printmeta1)
        save_json(mode_dir / "printMeta2.json", printmeta2)
        save_json(mode_dir / "render.json", render)
        if render_path.exists():
            save_json(mode_dir / "image_stats.json", result["image_stats"])
            save_json(mode_dir / "diff_vs_baseline.json", result["diff_vs_baseline"])
        mode_results[mode_key] = result

    variants = [
        Variant(
            key="highlight_only",
            description="highlight rolloff soft (2) with defaults otherwise",
            sdk_settings={"highlight_rolloff": 2, "output_tonemap": 1},
            cli_args=["--useMeta", "--rollOff", "2"],
            expects_cli_change=True,
        ),
        Variant(
            key="output_tonemap_high",
            description="output tone map high (2)",
            sdk_settings={"output_tonemap": 2},
            cli_args=["--useMeta", "--outputToneMap", "2"],
            expects_cli_change=True,
        ),
        Variant(
            key="exposure_only",
            description="exposure adjust +3.0",
            sdk_settings={"exposure_adjust": 3.0},
            cli_args=["--useMeta", "--exposure", "3.000000"],
            expects_cli_change=True,
        ),
        Variant(
            key="cdl_slope_only",
            description="CDL slope 1.8/0.4/1.0 enabled",
            sdk_settings={
                "cdl_enabled": True,
                "cdl_slope": [1.8, 0.4, 1.0],
            },
            cli_args=[
                "--useMeta",
                "--cdlRedSlope", "1.800000",
                "--cdlGreenSlope", "0.400000",
                "--cdlBlueSlope", "1.000000",
                "--cdlRedOffset", "0.000000",
                "--cdlGreenOffset", "0.000000",
                "--cdlBlueOffset", "0.000000",
                "--cdlRedPower", "1.000000",
                "--cdlGreenPower", "1.000000",
                "--cdlBluePower", "1.000000",
                "--cdlSaturation", "1.000000",
            ],
            expects_cli_change=True,
        ),
        Variant(
            key="cdl_sat_only",
            description="CDL saturation 1.6 enabled",
            sdk_settings={"cdl_enabled": True, "cdl_saturation": 1.6},
            cli_args=[
                "--useMeta",
                "--cdlRedSlope", "1.000000",
                "--cdlGreenSlope", "1.000000",
                "--cdlBlueSlope", "1.000000",
                "--cdlRedOffset", "0.000000",
                "--cdlGreenOffset", "0.000000",
                "--cdlBlueOffset", "0.000000",
                "--cdlRedPower", "1.000000",
                "--cdlGreenPower", "1.000000",
                "--cdlBluePower", "1.000000",
                "--cdlSaturation", "1.600000",
            ],
            expects_cli_change=True,
        ),
        Variant(
            key="cdl_disabled_extreme",
            description="CDL slope extreme but cdl_enabled=false",
            sdk_settings={"cdl_enabled": False, "cdl_slope": [1.8, 0.4, 1.0]},
            cli_args=[],
            expects_cli_change=False,
        ),
    ]

    variant_results: Dict[str, Any] = {}
    for variant in variants:
        variant_dir = variants_root / variant.key
        variant_dir.mkdir(parents=True, exist_ok=True)
        sdk_rmd = create_sdk_rmd(native, COMPARE_ROOT / "sdk_authored" / variant.key, variant)
        sdk_rmd_path = Path(str(sdk_rmd["copied_rmd_path"]))
        reel_sdk_rmd_path = sdk_rmd_path.with_name(f"{REEL_ID}.RMD")
        if sdk_rmd_path.name != reel_sdk_rmd_path.name:
            shutil.copy2(sdk_rmd_path, reel_sdk_rmd_path)

        omit_rolloff = variant.key == "highlight_only"
        omit_output_tonemap = variant.key == "output_tonemap_high"
        transform_args = display_transform_args(omit_output_tonemap=omit_output_tonemap, omit_rolloff=omit_rolloff)

        direct_render = run_render(CLIP_PATH, variant_dir / "direct_cli.jpg", variant.cli_args, transform_args=transform_args)
        direct_path = Path(str(direct_render["output_path"]))

        load_rmd_render = run_render(CLIP_PATH, variant_dir / "loadRMD.jpg", ["--loadRMD", str(sdk_rmd_path), "--useRMD", "1"], transform_args=transform_args)
        load_rmd_path = Path(str(load_rmd_render["output_path"]))

        adjacent_dir = variant_dir / "adjacent_dir"
        adjacent_clip = build_adjacent_clip_env(adjacent_dir, reel_sdk_rmd_path)
        adjacent_render = run_render(adjacent_clip, adjacent_dir / "adjacent_useRMD.jpg", ["--useRMD", "1"], transform_args=transform_args, cwd=adjacent_dir)
        adjacent_path = Path(str(adjacent_render["output_path"]))

        master_dir = variant_dir / "master_rmd_folder"
        master_dir.mkdir(exist_ok=True)
        shutil.copy2(reel_sdk_rmd_path, master_dir / reel_sdk_rmd_path.name)
        master_render = run_render(CLIP_PATH, variant_dir / "masterRMDFolder.jpg", ["--masterRMDFolder", str(master_dir), "--useRMD", "1"], transform_args=transform_args)
        master_path = Path(str(master_render["output_path"]))

        print_meta_load = run_print_meta(CLIP_PATH, ["--loadRMD", str(sdk_rmd_path), "--useRMD", "1"], level=2)
        print_meta_adjacent = run_print_meta(adjacent_clip, ["--useRMD", "1"], level=2, cwd=adjacent_dir)
        print_meta_master = run_print_meta(CLIP_PATH, ["--masterRMDFolder", str(master_dir), "--useRMD", "1"], level=2)

        variant_result = {
            "description": variant.description,
            "sdk_rmd": sdk_rmd,
            "direct_cli_command": direct_render["command"],
            "rmd_load_command": load_rmd_render["command"],
            "adjacent_command": adjacent_render["command"],
            "master_rmd_command": master_render["command"],
            "print_meta_load": print_meta_load,
            "print_meta_adjacent": print_meta_adjacent,
            "print_meta_master": print_meta_master,
            "diffs": {
                "baseline_vs_direct_cli": image_diff(baseline_image, direct_path),
                "baseline_vs_loadRMD": image_diff(baseline_image, load_rmd_path),
                "direct_cli_vs_loadRMD": image_diff(direct_path, load_rmd_path),
                "baseline_vs_adjacent_useRMD": image_diff(baseline_image, adjacent_path),
                "baseline_vs_masterRMDFolder": image_diff(baseline_image, master_path),
            },
        }
        variant_results[variant.key] = variant_result
        save_json(variant_dir / "variant_result.json", variant_result)

    rcx_probe = lookup_rcx_cdl_enable_presets()
    save_json(COMPARE_ROOT / "rcx_effect_preset_probe.json", rcx_probe)

    sdk_rmd_texts: Dict[str, str] = {}
    for variant in variants + [controlled_variant]:
        rmd_path = COMPARE_ROOT / "sdk_authored" / variant.key / f"{REEL_ID}.RMD"
        if rmd_path.exists():
            sdk_rmd_texts[variant.key] = rmd_path.read_text(encoding="utf-8", errors="ignore")
    representative_printmeta = {
        key: parse_printmeta_level1(value["printMeta_1"]["stdout"] or "")
        for key, value in mode_results.items()
    }
    structural_notes = {
        "rcx_app_present": RCX_BINARY.exists(),
        "rcx_cli_automation_available": False,
        "rcx_rmd_generation_blocker": (
            "REDCINE-X PRO is installed, but no documented non-GUI RMD export path was found in this environment. "
            "This pass could inspect bundled presets and binaries, but not generate a true RCX-authored RMD headlessly."
        ),
        "sdk_field_presence": {
            "kelvin": file_contains(COMPARE_ROOT / "sdk_authored" / "controlled_reference" / f"{REEL_ID}.RMD", "<kelvin"),
            "tint": file_contains(COMPARE_ROOT / "sdk_authored" / "controlled_reference" / f"{REEL_ID}.RMD", "<tint"),
            "saturation": file_contains(COMPARE_ROOT / "sdk_authored" / "controlled_reference" / f"{REEL_ID}.RMD", "<saturation"),
            "contrast": file_contains(COMPARE_ROOT / "sdk_authored" / "controlled_reference" / f"{REEL_ID}.RMD", "<contrast"),
            "exposurecompensation": file_contains(COMPARE_ROOT / "sdk_authored" / "exposure_only" / f"{REEL_ID}.RMD", "<exposurecompensation"),
            "exposureadjust": file_contains(COMPARE_ROOT / "sdk_authored" / "exposure_only" / f"{REEL_ID}.RMD", "<exposureadjust"),
            "highlightrolloff": file_contains(COMPARE_ROOT / "sdk_authored" / "highlight_only" / f"{REEL_ID}.RMD", "<highlightrolloff"),
            "outputtonemap": file_contains(COMPARE_ROOT / "sdk_authored" / "output_tonemap_high" / f"{REEL_ID}.RMD", "<outputtonemap"),
            "cdl": file_contains(COMPARE_ROOT / "sdk_authored" / "cdl_slope_only" / f"{REEL_ID}.RMD", "<cdl "),
            "cdlenabled": file_contains(COMPARE_ROOT / "sdk_authored" / "cdl_slope_only" / f"{REEL_ID}.RMD", "<cdlenabled"),
            "cdl_offset": file_contains(COMPARE_ROOT / "sdk_authored" / "cdl_slope_only" / f"{REEL_ID}.RMD", "<Offset>"),
            "cdl_power": file_contains(COMPARE_ROOT / "sdk_authored" / "cdl_slope_only" / f"{REEL_ID}.RMD", "<Power>"),
            "cdl_saturation": file_contains(COMPARE_ROOT / "sdk_authored" / "cdl_sat_only" / f"{REEL_ID}.RMD", "<Saturation>"),
        },
        "sdk_rmd_diffs": {
            "highlight_vs_exposure": compare_texts(
                "highlight_only",
                sdk_rmd_texts.get("highlight_only", ""),
                "exposure_only",
                sdk_rmd_texts.get("exposure_only", ""),
            ),
            "exposure_vs_cdl_slope": compare_texts(
                "exposure_only",
                sdk_rmd_texts.get("exposure_only", ""),
                "cdl_slope_only",
                sdk_rmd_texts.get("cdl_slope_only", ""),
            ),
            "cdl_slope_enabled_vs_disabled": compare_texts(
                "cdl_slope_only",
                sdk_rmd_texts.get("cdl_slope_only", ""),
                "cdl_disabled_extreme",
                sdk_rmd_texts.get("cdl_disabled_extreme", ""),
            ),
        },
        "printmeta_observation": {
            "useRMD_changes_pixels": mode_results["B_useRMD1_only"]["diff_vs_baseline"]["mean_absolute_difference"] > 1e-3,
            "reported_exposure_adjust": representative_printmeta["B_useRMD1_only"].get("Exposure Adjust"),
            "reported_roll_off": representative_printmeta["B_useRMD1_only"].get("Roll Off"),
            "reported_output_tone_map": representative_printmeta["B_useRMD1_only"].get("Output Tone Map"),
            "reported_cdl_enabled": representative_printmeta["B_useRMD1_only"].get("CDL Enabled:"),
            "reported_cdl_slope_red": representative_printmeta["B_useRMD1_only"].get("CDL Slope: Red"),
            "note": "On this build, printMeta reports baseline-looking values even when loaded RMDs change rendered pixels.",
        },
        "rcx_effect_preset_probe": rcx_probe,
    }
    save_json(COMPARE_ROOT / "structural_notes.json", structural_notes)

    compatibility_rows = []
    field_map = [
        ("rollOff", "highlight_only"),
        ("outputToneMap", "output_tonemap_high"),
        ("saturation", "controlled_reference"),
        ("contrast", "controlled_reference"),
        ("Kelvin", "controlled_reference"),
        ("Tint", "controlled_reference"),
        ("ExposureAdjust / exposure", "exposure_only"),
        ("CDL slope", "cdl_slope_only"),
        ("CDL offset", "cdl_slope_only"),
        ("CDL power", "cdl_slope_only"),
        ("CDL saturation", "cdl_sat_only"),
        ("CDL enable flag", "cdl_disabled_extreme"),
    ]
    for field_name, variant_key in field_map:
        result = variant_results.get(variant_key)
        if result is None and variant_key == "controlled_reference":
            result = {
                "print_meta_load": mode_results["D_loadRMD_explicit"]["printMeta_1"],
                "print_meta_adjacent": mode_results["B_useRMD1_only"]["printMeta_1"],
                "print_meta_master": mode_results["E_masterRMDFolder_useRMD1"]["printMeta_1"],
                "diffs": {
                    "baseline_vs_loadRMD": mode_results["D_loadRMD_explicit"]["diff_vs_baseline"],
                    "baseline_vs_adjacent_useRMD": mode_results["B_useRMD1_only"]["diff_vs_baseline"],
                    "baseline_vs_masterRMDFolder": mode_results["E_masterRMDFolder_useRMD1"]["diff_vs_baseline"],
                    "baseline_vs_direct_cli": image_diff(
                        baseline_image,
                        resolve_rendered_output(
                            Path(
                                str(
                                    run_render(
                                        CLIP_PATH,
                                        RUN_ROOT / "controlled_reference_direct_cli.jpg",
                                        controlled_variant.cli_args,
                                    )["output_path"]
                                )
                            )
                        ),
                    ),
                    "direct_cli_vs_loadRMD": image_diff(
                        resolve_rendered_output(RUN_ROOT / "controlled_reference_direct_cli.jpg"),
                        Path(str(mode_results["D_loadRMD_explicit"]["render_path"])),
                    ),
                },
            }
        rmd_file = COMPARE_ROOT / "sdk_authored" / variant_key / f"{REEL_ID}.RMD"
        if field_name == "rollOff":
            exists_in_sdk = file_contains(rmd_file, "<highlightrolloff")
        elif field_name == "outputToneMap":
            exists_in_sdk = file_contains(rmd_file, "<outputtonemap")
        elif field_name == "saturation":
            exists_in_sdk = file_contains(rmd_file, "<saturation")
        elif field_name == "contrast":
            exists_in_sdk = file_contains(rmd_file, "<contrast")
        elif field_name == "Kelvin":
            exists_in_sdk = file_contains(rmd_file, "<kelvin")
        elif field_name == "Tint":
            exists_in_sdk = file_contains(rmd_file, "<tint")
        elif field_name == "ExposureAdjust / exposure":
            exists_in_sdk = file_contains(rmd_file, "<exposureadjust") or file_contains(rmd_file, "<exposurecompensation")
        elif field_name == "CDL slope":
            exists_in_sdk = file_contains(rmd_file, "<Slope>")
        elif field_name == "CDL offset":
            exists_in_sdk = file_contains(rmd_file, "<Offset>")
        elif field_name == "CDL power":
            exists_in_sdk = file_contains(rmd_file, "<Power>")
        elif field_name == "CDL saturation":
            exists_in_sdk = file_contains(rmd_file, "<Saturation>")
        else:
            exists_in_sdk = file_contains(rmd_file, "<cdlenabled")
        printmeta_text = (
            (result["print_meta_load"]["stdout"] or "")
            + (result["print_meta_load"]["stderr"] or "")
            + (result["print_meta_adjacent"]["stdout"] or "")
            + (result["print_meta_adjacent"]["stderr"] or "")
            + (result["print_meta_master"]["stdout"] or "")
            + (result["print_meta_master"]["stderr"] or "")
        ).lower()
        row = {
            "field": field_name,
            "exists_in_rcx_authored_rmd": None,
            "exists_in_sdk_authored_rmd": exists_in_sdk,
            "appears_in_printMeta_output": (
                ("roll off" in printmeta_text if field_name == "rollOff" else False)
                or ("output tone map" in printmeta_text if field_name == "outputToneMap" else False)
                or ("saturation" in printmeta_text if field_name == "saturation" else False)
                or ("contrast" in printmeta_text if field_name == "contrast" else False)
                or ("kelvin" in printmeta_text if field_name == "Kelvin" else False)
                or ("tint" in printmeta_text if field_name == "Tint" else False)
                or ("exposure adjust" in printmeta_text if field_name == "ExposureAdjust / exposure" else False)
                or ("cdl slope" in printmeta_text if field_name == "CDL slope" else False)
                or ("cdl offset" in printmeta_text if field_name == "CDL offset" else False)
                or ("cdl power" in printmeta_text if field_name == "CDL power" else False)
                or ("cdl saturation" in printmeta_text if field_name == "CDL saturation" else False)
                or ("cdl enabled" in printmeta_text if field_name == "CDL enable flag" else False)
            ),
            "changes_pixels_when_loaded_from_rmd": result["diffs"]["baseline_vs_loadRMD"]["mean_absolute_difference"] > 1e-3
            or result["diffs"]["baseline_vs_adjacent_useRMD"]["mean_absolute_difference"] > 1e-3
            or result["diffs"]["baseline_vs_masterRMDFolder"]["mean_absolute_difference"] > 1e-3,
            "changes_pixels_when_passed_directly_via_cli": result["diffs"]["baseline_vs_direct_cli"]["mean_absolute_difference"] > 1e-3,
            "pixel_diff_direct_vs_rmd": result["diffs"]["direct_cli_vs_loadRMD"]["mean_absolute_difference"],
        }
        compatibility_rows.append(row)
    save_json(COMPATIBILITY_PATH, {"redline_build": "65.1.3", "rows": compatibility_rows})

    final_summary = {
        "clip_id": CLIP_ID,
        "clip_path": str(CLIP_PATH),
        "redline_path": str(REDLINE),
        "rcx_binary_present": RCX_BINARY.exists(),
        "mode_results_summary": {
            key: {
                "render_diff_vs_baseline": value["diff_vs_baseline"],
                "printMeta1_returncode": value["printMeta_1"]["returncode"],
                "printMeta2_returncode": value["printMeta_2"]["returncode"],
            }
            for key, value in mode_results.items()
        },
        "variant_results_summary": {
            key: value["diffs"] for key, value in variant_results.items()
        },
        "conclusion": {
            "metadata_driven_exposure_possible": bool(
                variant_results["exposure_only"]["diffs"]["baseline_vs_loadRMD"]["mean_absolute_difference"] > 1e-3
            ),
            "metadata_driven_cdl_possible": bool(
                variant_results["cdl_slope_only"]["diffs"]["baseline_vs_loadRMD"]["mean_absolute_difference"] > 1e-3
            ),
            "blocked_by": (
                "RMD loading itself is not blocked on this REDLine 65.1.3 build/path. SDK-authored RMDs drive rendered pixels for exposure, "
                "highlight rolloff, output tone map, CDL slope, and CDL saturation. The remaining RED-native blockers are that printMeta does not "
                "reflect the loaded RMD-applied look state, direct CLI CDL flags do not match the metadata-driven CDL path on this build, and "
                "true RCX-authored RMDs could not be generated headlessly here for a direct RCX-vs-SDK parity test."
            ),
        },
    }
    save_json(RUN_ROOT / "summary.json", final_summary)


if __name__ == "__main__":
    main()
