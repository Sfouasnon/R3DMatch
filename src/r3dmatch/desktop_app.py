from __future__ import annotations

import argparse
import queue
import shlex
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional

from PIL import Image, ImageTk

from .identity import clip_id_from_path
from .matching import discover_clips


DEFAULT_LOGO_PATH = Path(__file__).resolve().parent / "static" / "r3dmatch_logo.png"
MAX_LOGO_WIDTH = 220
MAX_LOGO_HEIGHT = 88


def build_review_command(
    *,
    repo_root: str,
    input_path: str,
    output_path: str,
    backend: str,
    target_type: str,
    processing_mode: str,
    roi_x: Optional[str],
    roi_y: Optional[str],
    roi_w: Optional[str],
    roi_h: Optional[str],
    target_strategies: List[str],
    reference_clip_id: Optional[str],
    preview_mode: str,
    preview_lut: Optional[str],
) -> List[str]:
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "review-calibration",
        input_path,
        "--out",
        output_path,
        "--backend",
        backend,
        "--target-type",
        target_type,
        "--processing-mode",
        processing_mode,
        "--preview-mode",
        preview_mode,
    ]
    if all(value not in (None, "") for value in (roi_x, roi_y, roi_w, roi_h)):
        args.extend(["--roi-x", str(roi_x), "--roi-y", str(roi_y), "--roi-w", str(roi_w), "--roi-h", str(roi_h)])
    for strategy in target_strategies:
        args.extend(["--target-strategy", strategy])
    if reference_clip_id:
        args.extend(["--reference-clip-id", reference_clip_id])
    if preview_lut:
        args.extend(["--preview-lut", preview_lut])
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def build_approve_command(*, repo_root: str, analysis_dir: str, target_strategy: str, reference_clip_id: Optional[str]) -> List[str]:
    args = [
        "python3",
        "-m",
        "r3dmatch.cli",
        "approve-master-rmd",
        analysis_dir,
        "--target-strategy",
        target_strategy,
    ]
    if reference_clip_id:
        args.extend(["--reference-clip-id", reference_clip_id])
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def build_clear_cache_command(*, repo_root: str, analysis_dir: str) -> List[str]:
    args = ["python3", "-m", "r3dmatch.cli", "clear-preview-cache", analysis_dir]
    shell_command = f'cd "{repo_root}"; setenv PYTHONPATH "$PWD/src"; ' + " ".join(shlex.quote(item) for item in args)
    return ["/bin/tcsh", "-c", shell_command]


def scan_calibration_sources(input_path: str) -> Dict[str, object]:
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        return {
            "input_path": str(root),
            "exists": False,
            "clip_count": 0,
            "clip_ids": [],
            "sample_clip_ids": [],
            "rdc_count": 0,
            "r3d_count": 0,
            "warning": "Selected calibration folder does not exist.",
        }
    clips = discover_clips(str(root))
    clip_ids = [clip_id_from_path(str(path)) for path in clips]
    rdc_count = len({path.parent for path in clips if path.parent.suffix.lower() == ".rdc"})
    return {
        "input_path": str(root),
        "exists": True,
        "clip_count": len(clips),
        "clip_ids": clip_ids,
        "sample_clip_ids": clip_ids[:12],
        "rdc_count": rdc_count,
        "r3d_count": len(clips),
        "warning": None if clips else "No valid RED .R3D clips were found in the selected folder.",
    }


def run_ui_self_check(repo_root: str, *, minimal_mode: bool = False) -> List[str]:
    repo_root = str(Path(repo_root).expanduser().resolve())
    lines = [
        f"repo_root={repo_root}",
        "header section created",
        "calibration folder section created",
        "output folder section created",
        "basic settings section created",
        "roi section created",
        "strategies section created",
        "preview section created",
        "source summary section created",
        "actions section created",
        "log section created",
        f"minimal_mode={minimal_mode}",
    ]
    return lines


class R3DMatchDesktopApp:
    def __init__(self, root: tk.Tk, *, repo_root: str, minimal_mode: bool = False) -> None:
        self.root = root
        self.repo_root = str(Path(repo_root).expanduser().resolve())
        self.minimal_mode = minimal_mode
        self.root.title("R3DMatch Internal Review")
        self.root.geometry("1100x800")

        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self.active_process: Optional[subprocess.Popen[str]] = None
        self.source_summary: Dict[str, object] = {"clip_count": 0, "sample_clip_ids": []}

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.backend = tk.StringVar(value="red")
        self.target_type = tk.StringVar(value="gray_sphere")
        self.processing_mode = tk.StringVar(value="both")
        self.roi_x = tk.StringVar()
        self.roi_y = tk.StringVar()
        self.roi_w = tk.StringVar()
        self.roi_h = tk.StringVar()
        self.reference_clip_id = tk.StringVar()
        self.preview_mode = tk.StringVar(value="calibration")
        self.preview_lut = tk.StringVar()
        self.status_text = tk.StringVar(value="Select a calibration folder containing RED clips.")
        self.source_summary_text = tk.StringVar(value="Found 0 RED clips")
        self.command_preview = tk.StringVar(value="")

        self.strategy_vars = {
            "median": tk.BooleanVar(value=True),
            "optimal-exposure": tk.BooleanVar(value=True),
            "manual": tk.BooleanVar(value=False),
        }

        self.run_button: Optional[ttk.Button] = None
        self.approve_button: Optional[ttk.Button] = None
        self.open_report_button: Optional[ttk.Button] = None
        self.clear_cache_button: Optional[ttk.Button] = None
        self.reference_entry: Optional[ttk.Entry] = None
        self.clip_list: Optional[tk.Listbox] = None
        self.log_widget: Optional[scrolledtext.ScrolledText] = None
        self.logo_image: Optional[ImageTk.PhotoImage] = None

        self._build_layout()
        self._bind_updates()
        self._refresh_ui_state()
        self.root.after(150, self._drain_logs)

    def _section(self, parent: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.pack(fill="x", padx=0, pady=6)
        return frame

    def _path_row(self, parent: ttk.Frame, label_text: str, variable: tk.StringVar, browse_command: object) -> None:
        ttk.Label(parent, text=label_text).pack(anchor="w")
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=browse_command).pack(side="left", padx=(8, 0))

    def _build_layout(self) -> None:
        content = ttk.Frame(self.root, padding=14)
        content.pack(fill="both", expand=True)

        header = ttk.Frame(content)
        header.pack(fill="x", pady=(0, 8))
        self._build_header(header)

        calibration = self._section(content, "Calibration Folder")
        self._path_row(calibration, "Calibration Folder", self.input_path, self._pick_input)
        ttk.Label(calibration, textvariable=self.source_summary_text).pack(anchor="w", pady=(8, 4))
        self.clip_list = tk.Listbox(calibration, height=6)
        self.clip_list.pack(fill="x", expand=False)
        self._populate_clip_list([])

        output = self._section(content, "Output Folder")
        self._path_row(output, "Output Folder", self.output_path, self._pick_output)

        if self.minimal_mode:
            actions = self._section(content, "Actions")
            self.run_button = ttk.Button(actions, text="Run Review", command=self.run_review)
            self.run_button.pack(anchor="w")
            self._build_log_section(content)
            return

        basic = self._section(content, "Basic Settings")
        self._labeled_combobox(basic, "Backend", self.backend, ["red", "mock"])
        self._labeled_combobox(basic, "Target Type", self.target_type, ["gray_sphere", "gray_card", "color_chart"])
        self._labeled_combobox(basic, "Processing Mode", self.processing_mode, ["exposure", "color", "both"])

        roi = self._section(content, "ROI")
        roi_row = ttk.Frame(roi)
        roi_row.pack(fill="x")
        for label, variable in (("x", self.roi_x), ("y", self.roi_y), ("w", self.roi_w), ("h", self.roi_h)):
            cell = ttk.Frame(roi_row)
            cell.pack(side="left", padx=(0, 10))
            ttk.Label(cell, text=label).pack(anchor="w")
            ttk.Entry(cell, textvariable=variable, width=8).pack(anchor="w")

        strategies = self._section(content, "Strategies")
        checks = ttk.Frame(strategies)
        checks.pack(fill="x")
        ttk.Checkbutton(checks, text="median", variable=self.strategy_vars["median"]).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(checks, text="Optimal Exposure (Best Match to Gray)", variable=self.strategy_vars["optimal-exposure"]).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(checks, text="manual", variable=self.strategy_vars["manual"]).pack(side="left")
        ttk.Label(strategies, text="Reference Clip ID").pack(anchor="w", pady=(8, 0))
        self.reference_entry = ttk.Entry(strategies, textvariable=self.reference_clip_id)
        self.reference_entry.pack(fill="x")

        preview = self._section(content, "Preview")
        self._labeled_combobox(preview, "Preview Mode", self.preview_mode, ["calibration", "monitoring"])
        ttk.Label(preview, text="LUT (.cube)").pack(anchor="w", pady=(8, 0))
        lut_row = ttk.Frame(preview)
        lut_row.pack(fill="x", pady=(4, 0))
        ttk.Entry(lut_row, textvariable=self.preview_lut).pack(side="left", fill="x", expand=True)
        ttk.Button(lut_row, text="Browse", command=self._pick_lut).pack(side="left", padx=(8, 0))

        actions = self._section(content, "Actions")
        action_row = ttk.Frame(actions)
        action_row.pack(fill="x")
        self.run_button = ttk.Button(action_row, text="Run Review", command=self.run_review)
        self.run_button.pack(side="left", padx=(0, 8))
        self.approve_button = ttk.Button(action_row, text="Approve Master RMD", command=self.approve_master)
        self.approve_button.pack(side="left", padx=(0, 8))
        self.open_report_button = ttk.Button(action_row, text="Open Report", command=self.open_report)
        self.open_report_button.pack(side="left", padx=(0, 8))
        self.clear_cache_button = ttk.Button(action_row, text="Clear Preview Cache", command=self.clear_preview_cache)
        self.clear_cache_button.pack(side="left")

        self._build_log_section(content)

    def _labeled_combobox(self, parent: ttk.Frame, label_text: str, variable: tk.StringVar, values: List[str]) -> None:
        ttk.Label(parent, text=label_text).pack(anchor="w", pady=(0, 0))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").pack(fill="x", pady=(4, 0))

    def _build_log_section(self, parent: ttk.Frame) -> None:
        log = self._section(parent, "Log")
        ttk.Label(log, textvariable=self.command_preview, wraplength=1000, justify="left").pack(anchor="w", pady=(0, 6))
        self.log_widget = scrolledtext.ScrolledText(log, height=12, wrap="word")
        self.log_widget.pack(fill="both", expand=True)
        self.log_widget.insert("end", "Ready.\n")
        self.log_widget.configure(state="disabled")

    def _build_header(self, parent: ttk.Frame) -> None:
        logo_widget = self._try_build_logo(parent)
        if logo_widget is not None:
            logo_widget.pack(anchor="w", pady=(0, 8))
        ttk.Label(parent, text="R3DMatch", font=("Helvetica", 18, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Internal Review", font=("Helvetica", 11)).pack(anchor="w")
        ttk.Label(
            parent,
            text="Select a calibration folder containing RED clips, choose review settings, then run review.",
            wraplength=1000,
            justify="left",
        ).pack(anchor="w", pady=(6, 0))

    def _try_build_logo(self, parent: ttk.Frame) -> Optional[tk.Label]:
        try:
            logo_path = DEFAULT_LOGO_PATH.expanduser().resolve()
            if not logo_path.exists():
                return None
            image = Image.open(logo_path)
            image.thumbnail((MAX_LOGO_WIDTH, MAX_LOGO_HEIGHT), Image.Resampling.LANCZOS)
            self.logo_image = ImageTk.PhotoImage(image)
            return tk.Label(parent, image=self.logo_image, borderwidth=0)
        except Exception:
            self.logo_image = None
            return None

    def _bind_updates(self) -> None:
        self.input_path.trace_add("write", lambda *_: self._on_input_changed())
        self.output_path.trace_add("write", lambda *_: self._refresh_ui_state())
        self.reference_clip_id.trace_add("write", lambda *_: self._refresh_ui_state())
        self.preview_mode.trace_add("write", lambda *_: self._refresh_ui_state())
        for var in self.strategy_vars.values():
            var.trace_add("write", lambda *_: self._refresh_ui_state())

    def _append_log(self, text: str) -> None:
        if self.log_widget is None:
            return
        self.log_widget.configure(state="normal")
        self.log_widget.insert("end", text)
        self.log_widget.see("end")
        self.log_widget.configure(state="disabled")

    def _pick_input(self) -> None:
        selected = filedialog.askdirectory(title="Select Calibration Folder")
        if selected:
            self.input_path.set(selected)

    def _pick_output(self) -> None:
        selected = filedialog.askdirectory(title="Select Output Folder")
        if selected:
            self.output_path.set(selected)

    def _pick_lut(self) -> None:
        selected = filedialog.askopenfilename(title="Select Monitoring LUT", filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")])
        if selected:
            self.preview_lut.set(selected)

    def _selected_strategies(self) -> List[str]:
        return [name for name, variable in self.strategy_vars.items() if variable.get()]

    def _populate_clip_list(self, clip_ids: List[str]) -> None:
        if self.clip_list is None:
            return
        self.clip_list.delete(0, "end")
        if not clip_ids:
            self.clip_list.insert("end", "No clips discovered yet.")
            return
        for clip_id in clip_ids:
            self.clip_list.insert("end", clip_id)
        remaining = int(self.source_summary.get("clip_count", 0)) - len(clip_ids)
        if remaining > 0:
            self.clip_list.insert("end", f"... and {remaining} more")

    def _on_input_changed(self) -> None:
        candidate = self.input_path.get().strip()
        if not candidate:
            self.source_summary = {"clip_count": 0, "sample_clip_ids": [], "warning": "Found 0 RED clips"}
            self.source_summary_text.set("Found 0 RED clips")
            self._populate_clip_list([])
            self._refresh_ui_state()
            return
        self.source_summary = scan_calibration_sources(candidate)
        warning = self.source_summary.get("warning")
        if warning:
            self.source_summary_text.set(str(warning))
        else:
            self.source_summary_text.set(f"Found {self.source_summary['clip_count']} RED clips")
        self._populate_clip_list(self.source_summary.get("sample_clip_ids", []))
        self._refresh_ui_state()

    def _refresh_ui_state(self) -> None:
        has_source = bool(self.input_path.get().strip()) and int(self.source_summary.get("clip_count", 0)) > 0
        has_output = bool(self.output_path.get().strip())
        needs_reference = self.strategy_vars["manual"].get()
        if self.reference_entry is not None:
            state = "normal" if needs_reference else "disabled"
            self.reference_entry.configure(state=state)
        if self.active_process is None:
            if has_source and has_output:
                self.status_text.set("Ready")
            else:
                missing = []
                if not has_source:
                    missing.append("calibration folder with RED clips")
                if not has_output:
                    missing.append("output folder")
                self.status_text.set("Missing: " + ", ".join(missing))
        if self.run_button is not None:
            run_enabled = has_source and has_output and self._selected_strategies() and (not needs_reference or bool(self.reference_clip_id.get().strip()))
            self.run_button.configure(state=("normal" if run_enabled else "disabled"))
        if self.approve_button is not None:
            self.approve_button.configure(state=("normal" if has_output else "disabled"))
        if self.clear_cache_button is not None:
            self.clear_cache_button.configure(state=("normal" if has_output else "disabled"))
        if self.open_report_button is not None:
            report_pdf = Path(self.output_path.get().strip()).expanduser().resolve() / "report" / "preview_contact_sheet.pdf" if has_output else None
            self.open_report_button.configure(state=("normal" if report_pdf and report_pdf.exists() else "disabled"))

    def _validate(self) -> bool:
        if not self.input_path.get().strip():
            messagebox.showerror("Missing calibration folder", "Select a calibration folder containing RED clips.")
            return False
        if int(self.source_summary.get("clip_count", 0)) <= 0:
            messagebox.showerror("No RED media found", "No valid RED .R3D clips were discovered in the selected calibration folder.")
            return False
        if not self.output_path.get().strip():
            messagebox.showerror("Missing output folder", "Select an output folder.")
            return False
        if not self._selected_strategies():
            messagebox.showerror("Missing strategy", "Select at least one strategy.")
            return False
        if self.strategy_vars["manual"].get() and not self.reference_clip_id.get().strip():
            messagebox.showerror("Missing manual reference", "Manual strategy requires a reference clip ID.")
            return False
        return True

    def _run_subprocess(self, command: List[str]) -> None:
        self.command_preview.set(" ".join(command))
        self._append_log(f"$ {' '.join(command)}\n")
        self.status_text.set("Running...")

        def worker() -> None:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            self.active_process = process
            assert process.stdout is not None
            for line in process.stdout:
                self.log_queue.put(line)
            returncode = process.wait()
            self.log_queue.put(f"\n[exit] {returncode}\n")
            self.log_queue.put("__PROCESS_DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def _drain_logs(self) -> None:
        while not self.log_queue.empty():
            item = self.log_queue.get()
            if item == "__PROCESS_DONE__":
                self.active_process = None
                self._refresh_ui_state()
            else:
                self._append_log(item)
        self.root.after(150, self._drain_logs)

    def run_review(self) -> None:
        if not self._validate():
            return
        command = build_review_command(
            repo_root=self.repo_root,
            input_path=self.input_path.get().strip(),
            output_path=self.output_path.get().strip(),
            backend=self.backend.get(),
            target_type=self.target_type.get(),
            processing_mode=self.processing_mode.get(),
            roi_x=self.roi_x.get().strip() or None,
            roi_y=self.roi_y.get().strip() or None,
            roi_w=self.roi_w.get().strip() or None,
            roi_h=self.roi_h.get().strip() or None,
            target_strategies=self._selected_strategies(),
            reference_clip_id=self.reference_clip_id.get().strip() or None,
            preview_mode=self.preview_mode.get(),
            preview_lut=self.preview_lut.get().strip() or None,
        )
        self._run_subprocess(command)

    def approve_master(self) -> None:
        if not self.output_path.get().strip():
            messagebox.showerror("Missing output folder", "Select the review output folder first.")
            return
        selected = self._selected_strategies()
        command = build_approve_command(
            repo_root=self.repo_root,
            analysis_dir=self.output_path.get().strip(),
            target_strategy=selected[0] if selected else "median",
            reference_clip_id=self.reference_clip_id.get().strip() or None,
        )
        self._run_subprocess(command)

    def clear_preview_cache(self) -> None:
        if not self.output_path.get().strip():
            messagebox.showerror("Missing output folder", "Select the review output folder first.")
            return
        self._run_subprocess(build_clear_cache_command(repo_root=self.repo_root, analysis_dir=self.output_path.get().strip()))

    def open_report(self) -> None:
        report_pdf = Path(self.output_path.get().strip()).expanduser().resolve() / "report" / "preview_contact_sheet.pdf"
        if report_pdf.exists():
            subprocess.Popen(["open", str(report_pdf)])
        else:
            messagebox.showerror("Missing report", f"No report found at {report_pdf}")


def launch_desktop_ui(repo_root: str, *, minimal_mode: bool = False) -> None:
    root = tk.Tk()
    R3DMatchDesktopApp(root, repo_root=repo_root, minimal_mode=minimal_mode)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--check", action="store_true", help="Validate imports without launching the window")
    parser.add_argument("--self-check", action="store_true", help="Print structural UI section checks")
    parser.add_argument("--minimal", action="store_true", help="Launch the minimal fallback UI")
    args = parser.parse_args()
    if args.check:
        print("R3DMatch desktop UI imports OK")
        return
    if args.self_check:
        for line in run_ui_self_check(args.repo_root, minimal_mode=args.minimal):
            print(line)
        return
    launch_desktop_ui(args.repo_root, minimal_mode=args.minimal)


if __name__ == "__main__":
    main()
