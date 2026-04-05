from __future__ import annotations

import os
import sys


def pick_existing_file(*, title: str, directory: str = "", filter: str = "") -> str:
    if str(os.environ.get("R3DMATCH_PICK_FILE_ERROR") or "").strip():
        raise RuntimeError(str(os.environ.get("R3DMATCH_PICK_FILE_ERROR")))

    scripted_cancel = str(os.environ.get("R3DMATCH_PICK_FILE_CANCEL") or "").strip().lower()
    if scripted_cancel in {"1", "true", "yes", "on"}:
        return ""
    scripted_response = os.environ.get("R3DMATCH_PICK_FILE_RESPONSE")
    if scripted_response is not None:
        return str(scripted_response)

    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([sys.argv[0]])

    selected, _ = QFileDialog.getOpenFileName(
        None,
        title,
        directory,
        filter or "Executables (*)",
    )
    if owns_app:
        app.quit()
    return str(selected or "")


def pick_existing_directory(*, title: str, directory: str = "") -> str:
    if str(os.environ.get("R3DMATCH_PICK_FOLDER_ERROR") or "").strip():
        raise RuntimeError(str(os.environ.get("R3DMATCH_PICK_FOLDER_ERROR")))

    scripted_response = os.environ.get("R3DMATCH_PICK_FOLDER_RESPONSE")
    scripted_cancel = str(os.environ.get("R3DMATCH_PICK_FOLDER_CANCEL") or "").strip().lower()
    if scripted_cancel in {"1", "true", "yes", "on"}:
        return ""
    if scripted_response is not None:
        return str(scripted_response)

    from PySide6.QtWidgets import QApplication, QFileDialog

    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QApplication([sys.argv[0]])

    selected = QFileDialog.getExistingDirectory(
        None,
        title,
        directory,
        QFileDialog.Option.ShowDirsOnly | QFileDialog.Option.DontResolveSymlinks,
    )
    if owns_app:
        app.quit()
    return str(selected or "")
