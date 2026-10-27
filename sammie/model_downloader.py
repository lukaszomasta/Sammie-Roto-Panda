"""
model_downloader.py

A reusable dialog for downloading model files with:
- Progress bar per file + overall progress
- MD5 checksum verification
- .part extension during download, renamed on success
- On-demand or batch usage via a central MODEL_REGISTRY

All models are registered once in MODEL_REGISTRY at the bottom of this file.

Usage (on-demand, single model by key):
    from model_downloader import ensure_models

    if not ensure_models("matanyone", parent=self):
        return  # user cancelled or download failed

Usage (multiple models by key):
    if not ensure_models(["sam2_large", "sam2_base_plus"], parent=self):
        return

Usage (all models at once, e.g. on first install):
    if not ensure_models("all", parent=self, title="Downloading Models"):
        return

Usage (ad-hoc spec, bypassing the registry):
    from model_downloader import ensure_models, DownloadSpec

    if not ensure_models(DownloadSpec(url=..., md5=..., dest_dir=...), parent=self):
        return
"""

from __future__ import annotations

import hashlib
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
)
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)


# The portable Python's OpenSSL fails its first TLS setup when that happens off the main
# thread (SSLError "unknown error"), which is where downloads run. One context made here,
# at import on the main thread, initialises it.
ssl.create_default_context()


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class DownloadSpec:
    """Describes a single file to download.

    gated:   model page of a gated Hugging Face repository. The user's Hugging Face token
             (HF_TOKEN or `hf auth login`) is sent, and a refused download explains what to do.
    size_mb: approximate size, shown when ensure_models() asks before a large download.
    extract: for a zip archive (md5 may be None): (prefix inside the zip, destination folder)
             pairs. The matching members are unpacked and the archive is deleted.
    present: files whose presence means this spec is done (for archives).
    name:    saved file name, if it differs from the last part of the URL.
    """
    url: str
    md5: "str | None"
    dest_dir: str
    gated: "str | None" = None
    size_mb: int = 0
    extract: "tuple | None" = None
    present: "tuple | None" = None
    name: "str | None" = None

    @property
    def filename(self) -> str:
        return self.name or self.url.split("/")[-1]

    @property
    def final_path(self) -> Path:
        return Path(self.dest_dir) / self.filename

    @property
    def part_path(self) -> Path:
        return Path(self.dest_dir) / (self.filename + ".part")

    def already_downloaded(self) -> bool:
        if self.present:
            return all(Path(p).exists() for p in self.present)
        return self.final_path.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _hf_token() -> "str | None":
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return os.environ.get("HF_TOKEN")


def _extract(spec: DownloadSpec) -> None:
    """Unpack the members of a downloaded zip under each prefix into its destination."""
    import shutil
    import zipfile

    with zipfile.ZipFile(spec.part_path) as zf:
        for prefix, dest in spec.extract:
            members = [m for m in zf.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
            if not members:
                raise RuntimeError(f"{prefix} not found in {spec.filename}")
            for m in members:
                target = Path(dest) / m.filename[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    spec.part_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Worker (runs in a background QThread)
# ---------------------------------------------------------------------------

class _DownloadWorker(QObject):
    # file-level signals
    file_started = Signal(int, str)                 # (index, filename)
    file_progress = Signal(int, float, float, int)  # (index, mb_done, mb_total, percent)
    file_done = Signal(int)                         # (index,)
    file_skipped = Signal(int, str)                 # (index, filename)
    file_error = Signal(int, str)                   # (index, error_message)

    # overall
    overall_progress = Signal(int, int)      # (files_done, files_total)
    all_done = Signal()
    cancelled = Signal()

    def __init__(self, specs: List[DownloadSpec], parent: QObject | None = None):
        super().__init__(parent)
        self._specs = specs
        self._abort = False
        self._current_response = None  # set during active download
    
        self._session = requests.Session()
        self._session.headers.update({
            "Accept-Encoding": "identity"
        })

    @Slot()
    def run(self) -> None:
        total = len(self._specs)
        done = 0

        for idx, spec in enumerate(self._specs):
            if self._abort:
                self.cancelled.emit()
                return

            # Already on disk with correct checksum?
            if spec.already_downloaded():
                self.file_skipped.emit(idx, spec.filename)
                done += 1
                self.overall_progress.emit(done, total)
                continue

            # Ensure destination directory exists
            Path(spec.dest_dir).mkdir(parents=True, exist_ok=True)

            self.file_started.emit(idx, spec.filename)

            try:
                self._download_one(idx, spec)
            except Exception as exc:
                # Clean up partial file
                if spec.part_path.exists():
                    spec.part_path.unlink(missing_ok=True)
                if self._abort:
                    self.cancelled.emit()
                else:
                    self.file_error.emit(idx, str(exc))
                return  # stop processing further files on error

            done += 1
            self.overall_progress.emit(done, total)
            self.file_done.emit(idx)

        self.all_done.emit()

    def _download_one(self, idx: int, spec: DownloadSpec) -> None:
        headers = {}
        token = _hf_token() if spec.url.startswith("https://huggingface.co/") else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._current_response = self._session.get(spec.url, stream=True, timeout=30, headers=headers)
        response = self._current_response
        if spec.gated and response.status_code in (401, 403):
            raise RuntimeError(
                f"{spec.filename} is in a gated repository.\n"
                f"Request access at {spec.gated} and log in to Hugging Face on this computer "
                f"(run 'hf auth login', or set the HF_TOKEN environment variable), then try again."
                + ("" if token else "\nNo Hugging Face token was found.")
            )
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        last_emit = 0
        emit_interval = 1 * 1024 * 1024  # emit every 1 MB

        with open(spec.part_path, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if self._abort:
                    raise RuntimeError("Download cancelled by user.")
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if downloaded - last_emit >= emit_interval or downloaded == total_size:
                        mb_done = downloaded / 1_048_576
                        mb_total = total_size / 1_048_576 if total_size else 0
                        pct = int(downloaded / total_size * 100) if total_size else 0

                        self.file_progress.emit(idx, mb_done, mb_total, pct)
                        last_emit = downloaded

        # download finished — no active network socket anymore
        self._current_response = None

        if spec.extract:
            _extract(spec)
            return

        # Verify checksum
        actual_md5 = _md5(spec.part_path)
        if actual_md5 != spec.md5:
            spec.part_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum mismatch for {spec.filename}.\n"
                f"  Expected : {spec.md5}\n"
                f"  Got      : {actual_md5}"
            )

        # Atomic rename
        spec.part_path.rename(spec.final_path)

    def abort(self) -> None:
        self._abort = True
        # Close the socket so iter_content() unblocks immediately
        if self._current_response is not None:
            try:
                self._current_response.raw.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class ModelDownloadDialog(QDialog):
    """
    Modal dialog that downloads one or more model files in the background.

    Parameters
    ----------
    specs : list of DownloadSpec
        Files to download (already-present files are silently skipped).
    parent : QWidget, optional
    title : str
        Window title.
    """

    def __init__(
        self,
        specs: List[DownloadSpec],
        parent: QWidget | None = None,
        title: str = "Downloading Models",
    ):
        super().__init__(parent)
        self._specs = specs
        self._worker: _DownloadWorker | None = None
        self._thread: QThread | None = None
        self._success = False

        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setModal(True)

        self._build_ui()
        self._start_downloads()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(20, 20, 20, 20)

        # Status label (current file name)
        self._status_label = QLabel("Preparing…")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Per-file progress bar
        self._file_bar = QProgressBar()
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(0)
        self._file_bar.setTextVisible(True)
        layout.addWidget(self._file_bar)

        # Overall label + bar (only shown for multi-file downloads)
        self._overall_label = QLabel("Overall progress:")
        self._overall_bar = QProgressBar()
        self._overall_bar.setRange(0, len(self._specs))
        self._overall_bar.setValue(0)
        self._overall_bar.setFormat("%v / %m files")

        if len(self._specs) > 1:
            layout.addWidget(self._overall_label)
            layout.addWidget(self._overall_bar)
        else:
            self._overall_label.hide()
            self._overall_bar.hide()

        # Error label (hidden until needed)
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: red;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        # Button box: Cancel only while downloading; Close after
        self._button_box = QDialogButtonBox()
        self._cancel_btn = self._button_box.addButton(QDialogButtonBox.Cancel)
        self._cancel_btn.clicked.connect(self._on_cancel)
        layout.addWidget(self._button_box)

    # ------------------------------------------------------------------
    # Download orchestration
    # ------------------------------------------------------------------

    def _start_downloads(self) -> None:
        self._worker = _DownloadWorker(self._specs)
        self._thread = QThread(self)

        self._worker.moveToThread(self._thread)

        # Wire signals
        self._thread.started.connect(self._worker.run)

        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_skipped.connect(self._on_file_skipped)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_error.connect(self._on_file_error)
        self._worker.overall_progress.connect(self._on_overall_progress)
        self._worker.all_done.connect(self._on_all_done)
        self._worker.cancelled.connect(self._on_cancelled)

        self._thread.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot(int, str)
    def _on_file_started(self, idx: int, filename: str) -> None:
        self._status_label.setText(f"Downloading  {filename}…")
        self._file_bar.setValue(0)
        self._file_bar.setFormat("%p%")

    @Slot(int, float, float, int)
    def _on_file_progress(self, idx: int, mb_done: float, mb_total: float, pct: int) -> None:
        if mb_total > 0:
            self._file_bar.setValue(pct)
            self._file_bar.setFormat(f"{pct}%  ({mb_done:.1f} / {mb_total:.1f} MB)")
        else:
            self._file_bar.setRange(0, 0)

    @Slot(int, str)
    def _on_file_skipped(self, idx: int, filename: str) -> None:
        self._status_label.setText(f"✓  {filename}  (already downloaded)")
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(100)
        self._file_bar.setFormat("Already downloaded")

    @Slot(int)
    def _on_file_done(self, idx: int) -> None:
        self._file_bar.setRange(0, 100)
        self._file_bar.setValue(100)
        self._file_bar.setFormat("Verified ✓")

    @Slot(int, str)
    def _on_file_error(self, idx: int, message: str) -> None:
        self._thread.quit()
        spec = self._specs[idx]
        self._status_label.setText(f"Failed to download  {spec.filename}")
        self._error_label.setText(f"Error: {message}")
        self._error_label.show()
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

    @Slot(int, int)
    def _on_overall_progress(self, done: int, total: int) -> None:
        self._overall_bar.setValue(done)

    @Slot()
    def _on_all_done(self) -> None:
        self._thread.quit()
        self._success = True
        self.accept() # auto-close
        #self._status_label.setText("All models downloaded successfully.")
        #self._cancel_btn.setText("Close")
        #self._cancel_btn.clicked.disconnect()
        #self._cancel_btn.clicked.connect(self.accept)

    @Slot()
    def _on_cancelled(self) -> None:
        self._thread.quit()
        self.reject()

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.abort()
        # Dialog will close when the worker emits cancelled or we force-close
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelling…")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802
        if self._thread and self._thread.isRunning():
            if self._worker:
                self._worker.abort()
            self._thread.quit()
            self._thread.wait(3000)
        super().closeEvent(event)

    @property
    def succeeded(self) -> bool:
        """True if all downloads completed successfully."""
        return self._success


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def ensure_models(
    models: "str | DownloadSpec | List[str | DownloadSpec]",
    parent: "QWidget | None" = None,
    title: str = "Downloading Models",
    confirm: bool = False,
) -> bool:
    """
    Ensure one or more models are present, downloading any that are missing.

    ``models`` can be:
    - A registry key string:           "matanyone"
    - The special string "all":        downloads every entry in MODEL_REGISTRY
    - A list of registry key strings:  ["sam2_large", "matanyone"]
    - A DownloadSpec instance:         for ad-hoc specs not in the registry
    - A list mixing keys and specs
    - A MODEL_GROUPS key:              "corridorkey" (every file that model needs)

    Files already on disk are never downloaded again, so an offline copy with all models in
    place never touches the network. With confirm=True the user is asked first, with the
    total size, before anything is downloaded.

    Returns True if all models are present (or were successfully downloaded),
    False if the user cancelled or a download failed.
    """
    # Normalise to a flat list of DownloadSpec
    if isinstance(models, str) and models == "all":
        specs = list(MODEL_REGISTRY.values())
    else:
        if not isinstance(models, list):
            models = [models]
        specs = []
        for item in models:
            if isinstance(item, str) and item in MODEL_GROUPS:
                specs.extend(MODEL_GROUPS[item])
            elif isinstance(item, str):
                if item not in MODEL_REGISTRY:
                    raise KeyError(
                        f"Unknown model key {item!r}. "
                        f"Available keys: {list(MODEL_REGISTRY)}"
                    )
                specs.append(MODEL_REGISTRY[item])
            elif isinstance(item, DownloadSpec):
                specs.append(item)
            else:
                raise TypeError(f"Expected str or DownloadSpec, got {type(item)}")

    needed = [s for s in specs if not s.already_downloaded()]
    if not needed:
        return True

    if confirm:
        total_gb = sum(s.size_mb for s in needed) / 1024
        gated = sorted({s.gated for s in needed if s.gated})
        names = "\n".join(f"  {s.filename}" for s in needed[:12])
        if len(needed) > 12:
            names += f"\n  ... and {len(needed) - 12} more"
        message = f"{title}: {len(needed)} file(s) are missing (about {total_gb:.1f} GB):\n\n{names}\n\nDownload them now?"
        if gated:
            message += ("\n\nSome files are in gated Hugging Face repositories. You need access to\n"
                        + "\n".join(gated) + "\nand a Hugging Face login on this computer ('hf auth login').")
        answer = QMessageBox.question(parent, title, message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer != QMessageBox.Yes:
            return False

    dlg = ModelDownloadDialog(needed, parent=parent, title=title)
    accepted = dlg.exec() == QDialog.Accepted
    return accepted and dlg.succeeded

def ensure_local_model(
    key: str,
    parent: "QWidget | None" = None,
) -> bool:
    """
    Like ensure_models(), but for checkpoints with no direct download source
    (e.g. Google Drive-only files). Checks for local presence and shows a
    plain instructional message box if missing, instead of attempting a
    download.
    """
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model key {key!r}. Available keys: {list(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[key]
    if spec.already_downloaded():
        return True

    QMessageBox.warning(
        parent,
        "Model File Missing",
        f"'{spec.filename}' was not found.\n\n"
        f"This checkpoint has no automatic download source and must be placed manually:\n\n"
        f"{spec.final_path}",
    )
    return False

# ---------------------------------------------------------------------------
# Model registry  ← edit this to add / remove models
# ---------------------------------------------------------------------------

MODEL_REGISTRY: "dict[str, DownloadSpec]" = {
    "SAM2 Large": DownloadSpec(
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        md5="2b30654b6112c42a115563c638d238d9",
        dest_dir="checkpoints",
    ),
    "SAM2 Base": DownloadSpec(
        url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        md5="ec7bd7d23d280d5e3cfa45984c02eda5",
        dest_dir="checkpoints",
    ),
    "SAM2 Efficient": DownloadSpec(
        url="https://huggingface.co/yunyangx/efficient-track-anything/resolve/main/efficienttam_s_512x512.pt",
        md5="962e151a9dca3b75d8228a16e5264010",
        dest_dir="checkpoints",
    ),
    "matanyone": DownloadSpec(
        url="https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth",
        md5="a50eeaa149a37509feb45e3d6b06f41d",
        dest_dir="checkpoints",
    ),
    "matanyone2": DownloadSpec(
        url="https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth",
        md5="b1d3cfbb7596ecf3b88391198427ca95",
        dest_dir="checkpoints",
    ),
    "minimax_transformer": DownloadSpec(
        url="https://huggingface.co/zibojia/minimax-remover/resolve/main/transformer/diffusion_pytorch_model.safetensors",
        md5="183c7a631e831f73f8da64c5c4d83e2f",
        dest_dir="checkpoints/minimax/transformer",
    ),
    "minimax_vae": DownloadSpec(
        url="https://huggingface.co/zibojia/minimax-remover/resolve/main/vae/diffusion_pytorch_model.safetensors",
        md5="3f80444947443d8f36c0ed2497c20c8d",
        dest_dir="checkpoints/minimax/vae",
    ),
    "videomama": DownloadSpec(
        url="https://huggingface.co/SammyLim/VideoMaMa/resolve/main/unet/diffusion_pytorch_model.safetensors",
        md5="c8d457d4d5eb90f274bd441df60c8e47",
        dest_dir="checkpoints/videomama/unet",
    ),
    "svd_vae": DownloadSpec(
        url="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.fp16.safetensors",
        md5="46a0af9a794fb405221988a7e2b1396b",
        dest_dir="checkpoints/videomama/vae",
    ),
    "vitmatte_small": DownloadSpec(
        url="https://huggingface.co/nielsr/vitmatte-checkpoints/resolve/main/ViTMatte_S_DIS.pth",
        md5="64cffda7b7abb3b836ed075651e03607",
        dest_dir="checkpoints/vitmatte",
    ),
    "vitmatte_base": DownloadSpec(
        url="https://huggingface.co/nielsr/vitmatte-checkpoints/resolve/main/ViTMatte_B_DIS.pth",
        md5="374d711633d05a419c15f78674eaa816",
        dest_dir="checkpoints/vitmatte",
    ),
    "mematte_vits": DownloadSpec(
        url="https://drive.google.com/uc?id=PLACEHOLDER/MEMatte_ViTS_DIM.pth",  # Google Drive only, no direct HTTPS — place file manually
        md5="ebb5e96a6e12d9f83f0f652974e75785",
        dest_dir="checkpoints/mematte",
    ),
    "mematte_vitb": DownloadSpec(
        url="https://drive.google.com/uc?id=PLACEHOLDER/MEMatte_ViTB_DIM.pth",  # Google Drive only, no direct HTTPS — place file manually
        md5="8bf7f3d54d1bbc542714764d59b9acf4",
        dest_dir="checkpoints/mematte",
    ),
    "mematte_robust": DownloadSpec(
        url="https://drive.google.com/uc?id=PLACEHOLDER/MEMatte.pth",  # Google Drive only, no direct HTTPS — place file manually, real-world/AIM-500 mixed-data model
        md5="1e1bc19026dc5202a9d7e4f7bd12441c",
        dest_dir="checkpoints/mematte",
    ),
}


# Multi-file models, by the name the loaders pass to ensure_models(). Each file is checked
# on its own, so an offline copy with everything in place never downloads anything.
MODEL_GROUPS: "dict[str, list[DownloadSpec]]" = {
    "SAM3": [
    DownloadSpec(
        url="https://huggingface.co/facebook/sam3/resolve/main/config.json",
        md5="760c4d906c4ce85a56b4fb21347d6912",
        dest_dir="checkpoints/sam3",
        gated="https://huggingface.co/facebook/sam3",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/facebook/sam3/resolve/main/processor_config.json",
        md5="85f75dabc7f5d4c4865d21639c819cfa",
        dest_dir="checkpoints/sam3",
        gated="https://huggingface.co/facebook/sam3",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/facebook/sam3/resolve/main/merges.txt",
        md5="d2bc10797e70c920c14c69729ecdab34",
        dest_dir="checkpoints/sam3",
        gated="https://huggingface.co/facebook/sam3",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/facebook/sam3/resolve/main/model.safetensors",
        md5="f3d84beaa5fe179d5b2808fa533e9954",
        dest_dir="checkpoints/sam3",
        gated="https://huggingface.co/facebook/sam3",
        size_mb=3281,
    ),
    ],
    # LTX Alpha on NVIDIA: the prebuilt fp8 transformer with the Alpha-Gen LoRA fused in, the
    # encoded empty prompt, and Lightricks' VAEs. Apple Silicon needs the bf16 build (README).
    "ltx_alpha_fp8": [
    DownloadSpec(
        url="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused/resolve/main/ltx-2.5-22b-distilled-alpha-gen-0.9-fp8-fused.safetensors",
        md5="68c231c443c04b5ae375e18e7bc2273e",
        dest_dir="checkpoints/ltx_alpha",
        gated="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused",
        size_mb=22404,
    ),
    DownloadSpec(
        url="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused/resolve/main/empty_prompt_embeddings.safetensors",
        md5="b7dd1153d6a183dfd8aae8dc4e188429",
        dest_dir="checkpoints/ltx_alpha",
        gated="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused",
        size_mb=12,
    ),
    DownloadSpec(
        url="https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors",
        md5="f79dd38c32cca9b421614be53428ed5a",
        dest_dir="checkpoints/ltx_alpha",
        gated="https://huggingface.co/Lightricks/LTX-2.5",
        size_mb=1404,
    ),
    DownloadSpec(
        url="https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors",
        md5="fdc1459b18ea23ce7829e7a91a51fc09",
        dest_dir="checkpoints/ltx_alpha",
        gated="https://huggingface.co/Lightricks/LTX-2.5",
        size_mb=348,
    ),
    DownloadSpec(
        url="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused/resolve/main/LICENSE-2_x",
        md5="fe460f42798062bc9047895d62a165f7",
        dest_dir="checkpoints/ltx_alpha",
        gated="https://huggingface.co/lomasta/LTX-2.5-22b-Alpha-Gen-fp8-fused",
    ),
    ],
    # Lightricks' LTX-2 code at the tested commit, unpacked next to sammie_main.py.
    "ltx_code": [
        DownloadSpec(
            url="https://github.com/Lightricks/LTX-2/archive/be6d6d0a69d11a0a399c462ed92606692e5f8697.zip",
            md5=None,
            dest_dir="temp",
            name="LTX-2-be6d6d0.zip",
            size_mb=10,
            extract=(
                ("LTX-2-be6d6d0a69d11a0a399c462ed92606692e5f8697/packages/ltx-core/src/ltx_core/", "ltx_core"),
                ("LTX-2-be6d6d0a69d11a0a399c462ed92606692e5f8697/packages/ltx-pipelines/src/ltx_pipelines/", "ltx_pipelines"),
            ),
            present=("ltx_core/__init__.py", "ltx_pipelines/__init__.py"),
        ),
    ],
    "corridorkey": [
    DownloadSpec(
        url="https://huggingface.co/nikopueringer/CorridorKey2.0/resolve/main/corridorkey_v2.safetensors",
        md5="e3f2d366de74622f44b53bbc124bf2bf",
        dest_dir="checkpoints/corridorkey/weights",
        size_mb=2538,
    ),
    DownloadSpec(
        url="https://huggingface.co/nikopueringer/CorridorKey2.0/resolve/main/foundation_stats.safetensors",
        md5="c0b09287bd506e6503bbd81a037e44bb",
        dest_dir="checkpoints/corridorkey/weights",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/README.md",
        md5="77bc0e846f3a8d69467bcb1a9d3930c9",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_attn.py",
        md5="906b33b56acce669af1a94adba4a1eab",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_base.py",
        md5="879c7f7898d6d11956753e5161bfe60e",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_generic.py",
        md5="95e7d04520a829390379fcbded6befee",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_mlp.py",
        md5="e2800bffaa0f5a3f1e92bb0d72d9a79e",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_module_factory.py",
        md5="e9400511a7c623d41f781a0bdaccfd22",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/adaptor_registry.py",
        md5="c635e68de6ebb85e74e1831d66c5bc94",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/cls_token.py",
        md5="0647e7b8a5d05b51cd313228d01f2364",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/common.py",
        md5="eeb75c36549b54fc946732c2b5be8425",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/config.json",
        md5="616543e1c6f51a8aff65f5c2ecfa88b6",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/dinov2_arch.py",
        md5="90ccfd4395432d2824505d21ce18e541",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/dual_hybrid_vit.py",
        md5="f5313117292a55b1d1314ef506deaed5",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/enable_cpe_support.py",
        md5="2115ad6048083729c21e00603a736034",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/enable_damp.py",
        md5="8385628da69c0a3a715ae91f04a099a9",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/enable_spectral_reparam.py",
        md5="0e209a1c0589c93635bde9a3f5f509f2",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/eradio_model.py",
        md5="835f3f1f99b87636f7530b251c369e2c",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/extra_models.py",
        md5="8f039e8329e50de2eaab698d9c854eb2",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/extra_timm_models.py",
        md5="8b3d253017c3998e33259d464e16e365",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/feature_normalizer.py",
        md5="e3b445bf84cd42b6ebb4f07df88b20f6",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/forward_intermediates.py",
        md5="27ce2cd05a15dff25586ce6eb0e96332",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/hf_model.py",
        md5="d4bf654d240647e6a5d93228f03b6550",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/input_conditioner.py",
        md5="cc6da64c4601f1f1fa5065a44523f577",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/open_clip_adaptor.py",
        md5="f2eb1553f1df562fe4c0cbdae37a0e21",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/preprocessor_config.json",
        md5="b0ebf9fbdc2853dc7b299868cdd4d2e2",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/radio_model.py",
        md5="ffb59624143bbf426469cb9eedb0fa33",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/siglip2_adaptor.py",
        md5="bc7df1bfc758bc5a8580f59629f34511",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/utils.py",
        md5="54334a4caca6979e05d02c4cce1c3b47",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/vit_patch_generator.py",
        md5="dacdf116ee373a99f0ca0ca6a777c666",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/vitdet.py",
        md5="86275c0aa1d32a01794ca49c88daeafe",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1,
    ),
    DownloadSpec(
        url="https://huggingface.co/nvidia/C-RADIOv4-SO400M/resolve/c0457f5dc26ca145f954cd4fc5bb6114e5705ad8/model.safetensors",
        md5="3fdff667bcfcf5b3a527572fd4258c47",
        dest_dir="checkpoints/corridorkey/vfms/cradio/C-RADIOv4-SO400M",
        size_mb=1645,
    ),
    DownloadSpec(
        url="https://huggingface.co/Ruicheng/moge-2-vitb-normal/resolve/ca5f0e07ff01d3e5a364c1d954ed12ee1814b368/model.pt",
        md5="1f3d11493af7b2099216f21e3b865700",
        dest_dir="checkpoints/corridorkey/vfms/moge",
        size_mb=400,
    ),
    DownloadSpec(
        url="https://storage.googleapis.com/representations4d/checkpoints/pretrain_rvm_small16_256_204031069.npz",
        md5="6496fa923f3eb5ab2225b55ea655eeed",
        dest_dir="checkpoints/corridorkey/vfms/rvm",
        size_mb=258,
    ),
    ],
    # CorridorKey's code at the tested commit, unpacked next to sammie_main.py.
    "corridorkey_code": [
        DownloadSpec(
            url="https://github.com/nikopueringer/CorridorKey2.0/archive/485f5e9ab759ead6138cc990776184b59d6ea971.zip",
            md5=None,
            dest_dir="temp",
            name="CorridorKey2.0-485f5e9.zip",
            size_mb=1,
            extract=(("CorridorKey2.0-485f5e9ab759ead6138cc990776184b59d6ea971/", "CorridorKey2.0"),),
            present=("CorridorKey2.0/corridorkey/runtime.py",),
        ),
    ],
    # utils3d for CorridorKey's bundled MoGe; only fetched when it is not installed.
    "utils3d": [
        DownloadSpec(
            url="https://github.com/EasternJournalist/utils3d/archive/83bfa5bda008803f6521ce9547f94023b8afeddd.zip",
            md5=None,
            dest_dir="temp",
            name="utils3d-83bfa5b.zip",
            size_mb=1,
            extract=(("utils3d-83bfa5bda008803f6521ce9547f94023b8afeddd/utils3d/", "CorridorKey2.0/third_party/utils3d/utils3d"),),
            present=("CorridorKey2.0/third_party/utils3d/utils3d/__init__.py",),
        ),
    ],
}


# ---------------------------------------------------------------------------
# Quick test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

    app = QApplication(sys.argv)

    win = QMainWindow()
    btn = QPushButton("Download SAM2 Models")

    def launch():
        ok = ensure_models(["Base", "Large"], parent=win, title="Downloading SAM2 Models")
        print("All models ready:", ok)

    btn.clicked.connect(launch)
    win.setCentralWidget(btn)
    win.resize(300, 100)
    win.show()

    sys.exit(app.exec())
