# sammie/matting.py
import gc
import os

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QProgressDialog
from sammie.model_downloader import ensure_models, ensure_local_model, MODEL_GROUPS, MODEL_REGISTRY
from sammie.settings_manager import get_settings_manager
from tqdm import tqdm
from sammie import core


# ---------------------------------------------------------------------------
# General Helpers
# ---------------------------------------------------------------------------

def _import_detectron2(parent_window):
    """detectron2 (ViTMatte / MEMatte only) is optional; it is hard to build on Windows."""
    try:
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate
    except ImportError as e:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.warning(
            parent_window,
            "detectron2 Missing",
            "ViTMatte and MEMatte need detectron2, which is not installed:\n\n"
            f"{e}\n\nSee \"Installation\" in README.md. The other matting models work without it.",
        )
        return None
    return LazyConfig, instantiate, DetectionCheckpointer


def generate_trimap(mask: np.ndarray, erode_pct: float = 2.5, dilate_pct: float = 2.5,
                     min_kernel: int = 3) -> np.ndarray:
    """
    Convert a binary SAM mask into a trimap for trimap-based matting models.

    Kernel sizes are computed as a percentage of the mask's shorter side,
    so results stay proportionally consistent across resolutions.

    Args:
        mask: uint8 array, values 0 or 255 (as loaded from core.mask_dir)
        erode_pct: erosion kernel size as % of min(height, width)
        dilate_pct: dilation kernel size as % of min(height, width)
        min_kernel: floor on kernel size in pixels, so tiny frames don't
                    collapse to a 0-1px (i.e. no-op) kernel

    Returns:
        uint8 trimap array: 0 = background, 128 = unknown, 255 = foreground
    """
    h, w = mask.shape[:2]
    min_side = min(h, w)

    def _pct_to_kernel_size(pct):
        size = int(round(min_side * (pct / 100.0)))
        size = max(size, min_kernel)
        if size % 2 == 0: 
            size += 1
        return size

    erode_size = _pct_to_kernel_size(erode_pct)
    dilate_size = _pct_to_kernel_size(dilate_pct)

    binary = (mask > 127).astype(np.uint8) * 255

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))

    definite_fg = cv2.erode(binary, erode_kernel)
    definite_bg = 255 - cv2.dilate(binary, dilate_kernel)

    trimap = np.full(mask.shape, 128, dtype=np.uint8)
    trimap[definite_fg > 0] = 255
    trimap[definite_bg > 0] = 0

    return trimap

class MattingManager:
    """
    Shared base class for matting managers.
    Provides common infrastructure: callbacks, image resize/restore, mask loading,
    progress dialog helpers, and matting directory management.
    Subclasses must implement load_matting_model() and run_matting().
    """

    def __init__(self):
        self.processor = None
        self.propagated = False  # whether we have propagated the mattes
        self.callbacks = []

    def add_callback(self, callback):
        """Add callback for matting events"""
        self.callbacks.append(callback)

    def _notify(self, action, **kwargs):
        """Notify callbacks of changes"""
        for callback in self.callbacks:
            try:
                callback(action, **kwargs)
            except RuntimeError as e:
                # Allow cancellation to propagate
                if str(e) == "USER_CANCELLED":
                    raise
                print(f"Callback error: {e}")
            except Exception as e:
                print(f"Callback error: {e}")

    def _prepare_device(self, load_to_cpu=False):
        """Return the appropriate torch device and clear cache"""
        core.DeviceManager.clear_cache()
        if load_to_cpu:
            return torch.device("cpu")
        return core.DeviceManager.get_device()

    def unload_matting_model(self):
        """Unload the matting model and clear cache"""
        self.processor = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded Matting model")

    def _resize_image(self, image):
        """Resize image and ensure dimensions are multiples of 8 for the model."""
        settings_mgr = get_settings_manager()
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        h, w = image.shape[:2]

        # 1. Determine scaling factor
        scale = 1.0
        if max_size > 0:
            min_side = min(h, w)
            if min_side > max_size:
                scale = max_size / min_side

        # 2. ALWAYS round to a multiple of 8
        # This ensures [3, 1384, 600] instead of [3, 1390, 602]
        new_h = (int(h * scale) // 8) * 8
        new_w = (int(w * scale) // 8) * 8

        # 3. Always resize, even if scale is 1.0, to catch those extra pixels
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _restore_image_size(self, image, original_size):
        """Restore image to original size. original_size must be (w, h) as expected by cv2."""
        original_w, original_h = original_size
        restored_image = cv2.resize(
            image, (original_w, original_h), interpolation=cv2.INTER_LINEAR
        )
        return restored_image

    def _load_mask_for_matting(self, object_id, frame_number, device, combine_ids=None):
        """
        Load and validate a mask for matting processing.

        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory and returned as a single mask, without touching any files on disk.
        object_id is used only as the label for error messages in that case.

        Args:
            object_id: ID of the object (or output label when combining)
            frame_number: Frame number
            device: Processing device
            combine_ids: Optional list of object IDs to union into a single mask

        Returns:
            tuple: (mask_tensor, original_size) or (None, None) if failed
        """
        if combine_ids:
            union_mask = None
            original_size = None
            for oid in combine_ids:
                mask_filename = os.path.join(
                    core.mask_dir, f"{frame_number:05d}", f"{oid}.png"
                )
                if not os.path.exists(mask_filename):
                    continue
                m = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                if original_size is None:
                    original_size = m.shape[1::-1]
                union_mask = m if union_mask is None else np.maximum(union_mask, m)
            if union_mask is None or not np.any(union_mask):
                print(f"Combined mask is blank or missing for frame {frame_number}")
                return None, None
            mask = self._resize_image(union_mask)
            mask = torch.tensor(mask, dtype=torch.float32, device=device)
            return mask, original_size

        mask_filename = os.path.join(
            core.mask_dir, f"{frame_number:05d}", f"{object_id}.png"
        )
        if not os.path.exists(mask_filename):
            print(
                f"Mask not found for object {object_id} at frame {frame_number}: {mask_filename}"
            )
            return None, None

        mask = cv2.imread(mask_filename, cv2.IMREAD_GRAYSCALE)
        if mask is None or not np.any(mask):
            print(
                f"Mask is blank or invalid for object {object_id} at frame {frame_number}"
            )
            return None, None

        original_size = mask.shape[1::-1]
        mask = core.apply_mask_postprocessing(mask)
        mask = self._resize_image(mask)
        mask = torch.tensor(mask, dtype=torch.float32, device=device)

        return mask, original_size

    def _make_progress_dialog(self, parent_window, total_operations, unit="frame"):
        """Create and show the Qt progress dialog and tqdm bar"""
        progress_dialog = QProgressDialog(
            "Running matting...", "Cancel", 0, 100, parent_window
        )
        progress_dialog.setWindowTitle("Matting Progress")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setAutoClose(True)
        progress_dialog.show()
        pbar = tqdm(total=total_operations, desc="Matting Progress", unit=unit)
        return progress_dialog, pbar

    def _get_frame_range(self):
        """
        Read in/out points from settings and return (start_frame, end_frame, frames_to_process).
        """
        settings_mgr = get_settings_manager()
        frame_count = core.VideoInfo.total_frames
        in_point = settings_mgr.get_session_setting("in_point", None)
        out_point = settings_mgr.get_session_setting("out_point", None)
        start_frame = in_point if in_point is not None else 0
        end_frame = out_point if out_point is not None else frame_count - 1
        frames_to_process = end_frame - start_frame + 1
        return start_frame, end_frame, frames_to_process

    def _collect_image_paths(self, start_frame, end_frame):
        """Return a list of existing frame image paths in [start_frame, end_frame]."""
        extension = core.get_frame_extension()
        images = []
        for frame_number in range(start_frame, end_frame + 1):
            image_filename = os.path.join(
                core.frames_dir, f"{frame_number:05d}.{extension}"
            )
            if os.path.exists(image_filename):
                images.append(image_filename)
        return images

    def clear_matting(self):
        """Clear matting data"""
        import shutil

        if os.path.exists(core.matting_dir):
            shutil.rmtree(core.matting_dir)
        os.makedirs(core.matting_dir)
        core.clear_matting_fg()
        self.propagated = False
        print("Matting data cleared")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        raise NotImplementedError("Subclasses must implement load_matting_model()")

    def run_matting(self, points_list, parent_window, combined=False):
        raise NotImplementedError("Subclasses must implement run_matting()")


# ---------------------------------------------------------------------------
# MatAnyone backend
# ---------------------------------------------------------------------------


class MatAnyManager(MattingManager):
    """Matting manager that uses the MatAnyone / MatAnyone2 model."""

    BACKEND = "matanyone"

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """Load the MatAnyone model and return processor"""
        from matanyone.inference.inference_core import InferenceCore
        from matanyone.utils.get_default_model import get_matanyone_model

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone2")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if matting_model == "MatAnyone2":
            checkpoint = "./checkpoints/matanyone2.pth"
            if not ensure_models("matanyone2", parent=parent_window):
                return False  # user cancelled or download failed
        else:
            checkpoint = "./checkpoints/matanyone.pth"
            if not ensure_models("matanyone", parent=parent_window):
                return False  # user cancelled or download failed

        matanyone = get_matanyone_model(checkpoint, device=device)
        print(
            f"Loaded {matting_model} model to {device} with max size {max_size} and combined={combined}"
        )

        # Initialize inference processor
        self.processor = InferenceCore(matanyone, cfg=matanyone.cfg, device=device)
        return self.processor

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using multiple keyframes for each object.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """
        if self.processor is None:
            print("Matting model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()
        frame_count = core.VideoInfo.total_frames

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(
            f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)"
        )

        # Get unique object IDs from points list
        object_ids = sorted(
            list(
                set(point["object_id"] for point in points_list if "object_id" in point)
            )
        )
        if not object_ids:
            print("No objects found for matting")
            return 0

        # Find all keyframes for each object (within processing range)
        object_keyframes = {}
        for object_id in object_ids:
            keyframes = sorted(
                list(
                    set(
                        point["frame"]
                        for point in points_list
                        if point.get("object_id") == object_id
                        and start_frame <= point["frame"] <= end_frame
                    )
                )
            )
            if keyframes:
                object_keyframes[object_id] = keyframes
            else:
                print(
                    f"No frames found for object {object_id} in range {start_frame}-{end_frame}"
                )

        if not object_keyframes:
            print("No valid keyframes found for any objects in the specified range")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            earliest_keyframe = min(kf[0] for kf in object_keyframes.values())
            object_ids = [0]
            object_keyframes = {0: [earliest_keyframe]}
        else:
            combine_ids = None

        # Calculate total operations for progress tracking
        total_operations = 0
        for object_id, keyframes in object_keyframes.items():
            first_keyframe = keyframes[0]

            # Operations before first keyframe (backward propagation to start_frame)
            total_operations += first_keyframe - start_frame

            # Operations between keyframes and after last keyframe to end_frame
            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    total_operations += end_frame - keyframes[i] + 1
                else:
                    total_operations += keyframes[i + 1] - keyframes[i]

        progress_dialog, pbar = self._make_progress_dialog(
            parent_window, total_operations
        )

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        images = self._collect_image_paths(start_frame, end_frame)

        operations_completed = 0

        # Process each object with its keyframes
        for object_id, keyframes in object_keyframes.items():
            if progress_dialog.wasCanceled():
                break

            pbar.set_description(f"Object {object_id}")

            # Process segments for this object
            success = self._process_object_with_keyframes(
                images,
                object_id,
                keyframes,
                end_frame + 1,
                device,
                progress_dialog,
                operations_completed,
                total_operations,
                pbar,
                parent_window,
                start_frame=start_frame,
                combine_ids=combine_ids,
            )

            if not success:
                break

            # Update operations completed for this object
            first_keyframe = keyframes[0]
            operations_completed += (
                first_keyframe - start_frame
            )  # backward from first keyframe

            for i in range(len(keyframes)):
                if i == len(keyframes) - 1:
                    operations_completed += (
                        end_frame - keyframes[i] + 1
                    )  # last keyframe to end
                else:
                    operations_completed += (
                        keyframes[i + 1] - keyframes[i]
                    )  # between keyframes

        # Close tqdm progress bar
        pbar.close()

        # Final cleanup
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify("matting_complete")
            return 1

    def _process_object_with_keyframes(
        self,
        images,
        object_id,
        keyframes,
        frame_count,
        device,
        progress_dialog,
        operations_completed,
        total_operations,
        pbar,
        parent_window,
        start_frame=0,
        combine_ids=None,
    ):
        """
        Process a single object using multiple keyframes.

        Args:
            images: List of image paths
            object_id: ID of the object to process (also the output file label)
            keyframes: Sorted list of keyframe indices for this object
            frame_count: Total number of frames
            device: Processing device
            progress_dialog: Progress dialog for user feedback
            operations_completed: Number of operations completed so far
            total_operations: Total operations for all objects
            pbar: tqdm progress bar
            parent_window: Parent window
            start_frame: Starting frame for the processing range
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        first_keyframe = keyframes[0]

        # Load and validate the first keyframe mask
        mask, original_size = self._load_mask_for_matting(
            object_id, first_keyframe, device, combine_ids=combine_ids
        )
        if mask is None:
            return False

        # Special case for single frame
        if len(images) == 1:
            return self._process_single_frame(
                images[0], mask, object_id, original_size, device
            )

        current_operations = operations_completed

        # 1. Process backward from first keyframe to start_frame
        if first_keyframe > start_frame:
            success = self._process_backward(
                images,
                mask,
                object_id,
                first_keyframe,
                original_size,
                device,
                progress_dialog,
                current_operations,
                total_operations,
                parent_window,
                pbar,
                start_frame_offset=start_frame,
            )
            if not success:
                return False
            current_operations += first_keyframe - start_frame

        # 2. Process forward segments between keyframes
        for i in range(len(keyframes)):
            if progress_dialog.wasCanceled():
                return False

            current_keyframe = keyframes[i]

            # Load mask for current keyframe (refresh for each segment)
            mask, original_size = self._load_mask_for_matting(
                object_id, current_keyframe, device, combine_ids=combine_ids
            )
            if mask is None:
                print(
                    f"Failed to load mask for object {object_id} at keyframe {current_keyframe}"
                )
                return False

            # Determine end frame for this segment
            if i == len(keyframes) - 1:
                end_frame = frame_count  # Last keyframe - process to end of range
            else:
                end_frame = keyframes[i + 1]  # Process to next keyframe (exclusive)

            # Process this forward segment
            if end_frame > current_keyframe:
                success = self._process_forward(
                    images,
                    mask,
                    object_id,
                    current_keyframe,
                    original_size,
                    device,
                    progress_dialog,
                    current_operations,
                    total_operations,
                    parent_window,
                    end_frame,
                    pbar,
                    start_frame_offset=start_frame,
                )
                if not success:
                    return False
                current_operations += end_frame - current_keyframe

        return True

    def _process_single_frame(self, frame_path, mask, object_id, original_size, device):
        """Process a single frame for matting"""
        try:
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self._resize_image(img)
            img = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(
                2, 0, 1
            )

            output_prob = self.processor.step(img, mask, objects=[1])
            for i in range(10):  # Warmup iterations
                output_prob = self.processor.step(img, first_frame_pred=True)
                core.DeviceManager.clear_cache()

            mat = self.processor.output_prob_to_mask(output_prob)
            mat = mat.detach().cpu().numpy()
            mat = (mat * 255).astype(np.uint8)
            mat = self._restore_image_size(mat, original_size)

            mat_filename = os.path.join(core.matting_dir, f"00000", f"{object_id}.png")
            os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
            cv2.imwrite(mat_filename, mat)
            return True

        except Exception as e:
            print(f"Error processing single frame: {e}")
            return False

    def _process_forward(
        self,
        images,
        mask,
        object_id,
        start_frame,
        original_size,
        device,
        progress_dialog,
        operations_completed,
        total_operations,
        parent_window,
        end_frame=None,
        pbar=None,
        start_frame_offset=0,
    ):
        """
        Process frames forward from start_frame.

        Args:
            images: List of image paths
            mask: Initial mask tensor
            object_id: Object ID
            start_frame: Starting frame (inclusive)
            original_size: Original image size
            device: Processing device
            progress_dialog: Progress dialog
            operations_completed: Operations completed before this segment
            total_operations: Total operations
            parent_window: Parent window
            end_frame: Ending frame (exclusive). If None, process to end of images.
            pbar: tqdm progress bar
            start_frame_offset: Offset for mapping array indices to absolute frame numbers

        Returns:
            bool: True if successful, False if cancelled or failed
        """
        if end_frame is None:
            end_frame = start_frame + len(images)

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting(
            "display_update_frequency", 5
        )

        try:
            for frame_number in range(start_frame, end_frame):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(
                        f"Warning: Frame {frame_number} out of range for images array"
                    )
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(
                    img / 255.0, dtype=torch.float32, device=device
                ).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(
                    core.matting_dir, f"{frame_number:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                current_progress = int(
                    ((operations_completed + (frame_number - start_frame) + 1) * 100)
                    / total_operations
                )
                progress_dialog.setValue(current_progress)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in forward processing: {e}")
            return False

    def _process_backward(
        self,
        images,
        mask,
        object_id,
        start_frame,
        original_size,
        device,
        progress_dialog,
        operations_completed,
        total_operations,
        parent_window,
        pbar=None,
        start_frame_offset=0,
    ):
        """Process frames backward from start_frame"""

        # Get display update frequency from settings
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting(
            "display_update_frequency", 5
        )

        try:
            for frame_number in range(start_frame, start_frame_offset - 1, -1):
                if progress_dialog.wasCanceled():
                    return False

                # Map absolute frame number to array index
                array_idx = frame_number - start_frame_offset
                if array_idx < 0 or array_idx >= len(images):
                    print(
                        f"Warning: Frame {frame_number} out of range for images array"
                    )
                    continue

                frame_path = images[array_idx]
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = self._resize_image(img)
                img = torch.tensor(
                    img / 255.0, dtype=torch.float32, device=device
                ).permute(2, 0, 1)

                if frame_number == start_frame:
                    # First frame - initialize with mask
                    output_prob = self.processor.step(img, mask, objects=[1])
                    for i in range(10):  # Warmup iterations
                        output_prob = self.processor.step(img, first_frame_pred=True)
                        core.DeviceManager.clear_cache()
                else:
                    # Subsequent frames - propagate
                    output_prob = self.processor.step(img)

                # Convert to matte
                mat = self.processor.output_prob_to_mask(output_prob)
                mat = mat.detach().cpu().numpy()
                mat = (mat * 255).astype(np.uint8)
                mat = self._restore_image_size(mat, original_size)

                # Save matte
                mat_filename = os.path.join(
                    core.matting_dir, f"{frame_number:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, mat)
                core.DeviceManager.clear_cache()

                # Update display at the specified frequency
                if frame_number % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_number)
                    except Exception as e:
                        print(f"Error updating display: {e}")

                # Update progress
                if pbar is not None:
                    pbar.update(1)
                operations_completed += 1
                progress_dialog.setValue(operations_completed * 100 // total_operations)
                QApplication.processEvents()

            return True

        except Exception as e:
            print(f"Error in backward processing: {e}")
            return False


# ---------------------------------------------------------------------------
# VideoMaMa backend
# ---------------------------------------------------------------------------


class VideoMaMaManager(MattingManager):
    """
    Matting manager that uses the VideoMaMa model.

    """

    BACKEND = "videomama"

    def unload_matting_model(self):
        """Unload the VideoMaMa pipeline and free VRAM"""
        if self.pipeline is not None:
            try:
                if hasattr(self.pipeline, "unet"):
                    self.pipeline.unet = None
                if hasattr(self.pipeline, "vae"):
                    self.pipeline.vae = None
            except Exception as e:
                print(f"Warning during pipeline teardown: {e}")
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded VideoMaMa model")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        """
        Load the VideoMaMa pipeline.
        """
        from videomama.pipeline_svd_mask_numpy import VideoInferencePipeline

        device = self._prepare_device(load_to_cpu)
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "videomama")
        max_size = settings_mgr.get_session_setting("matany_res", 0)
        overlap = settings_mgr.get_session_setting("matany_overlap", 2)
        batch_size = settings_mgr.get_session_setting("matany_chunk", 16)
        combined = settings_mgr.get_session_setting("matany_combined", False)

        if not ensure_models(["videomama", "svd_vae"], parent=parent_window):
            return False  # user cancelled or download failed

        try:
            self.pipeline = VideoInferencePipeline(
                base_model_path=os.path.join("checkpoints", "videomama"),
                unet_checkpoint_path=os.path.join("checkpoints", "videomama", "unet"),
                weight_dtype=torch.float16,
                device=str(device),
                enable_model_cpu_offload=False,  # Not much benefit here, since the vae is a small model
                vae_encode_chunk_size=1,  # Process VAE in small chunks, increasing doesnt help anything
                attention_mode="auto",  # Use xformers if available, else SDPA
                enable_vae_tiling=False,  # Tiling VAE is not worth it
                enable_vae_slicing=True,  # Process VAE one image at a time
            )
            print(
                f"Loaded {matting_model} model to {device} with max size {max_size}, overlap={overlap}, batch={batch_size}, and combined={combined}"
            )
            return True

        except Exception as e:
            print(f"Error loading VideoMaMa pipeline: {e}")
            self.pipeline = None
            return False

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        """
        Run matting on all frames, using tracked segmentation data.

        Args:
            points_list (list): List of point dictionaries containing object_id and frame information
            parent_window: Parent window for progress dialog
            combined (bool): If True, union all object masks into a single combined mask
                and run matting in a single pass instead of once per object.

        Returns:
            int: 1 if successful, 0 if cancelled/failed
        """

        if self.pipeline is None:
            print("VideoMaMa model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        extension = core.get_frame_extension()
        settings_mgr = get_settings_manager()

        # --- VideoMaMa batch settings ---
        batch_size = settings_mgr.get_session_setting(
            "matany_chunk", 16
        )  # frames per chunk sent to the model
        overlap = settings_mgr.get_session_setting(
            "matany_overlap", 2
        )  # frames re-processed at each boundary for continuity
        if overlap == 0:
            enable_boundary_blend = False
        else:
            enable_boundary_blend = (
                True  # blend overlap frames linearly at chunk boundaries
            )

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(
            f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)"
        )

        # Get unique object IDs from points list
        object_ids = sorted(
            list(
                set(point["object_id"] for point in points_list if "object_id" in point)
            )
        )
        if not object_ids:
            print("No objects found for matting")
            return 0

        # When combined mode is requested, run a single pass using the union of all
        # object masks loaded in memory — no files are written to disk.
        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        # Create matting directory if it doesn't exist
        os.makedirs(core.matting_dir, exist_ok=True)

        # If combined mode is selected, delete any existing matting files except object 0.
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        # Calculate total operations for progress tracking
        total_operations = len(
            self._generate_windows(frames_to_process, batch_size, overlap)
        ) * len(object_ids)

        progress_dialog, pbar = self._make_progress_dialog(
            parent_window, total_operations, unit="batch"
        )
        operations_completed = 0

        try:
            # Process each object separately
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break

                pbar.set_description(f"Object {object_id}")
                print(f"Processing object {object_id}...")

                batches_completed = self._process_object(
                    object_id,
                    start_frame,
                    end_frame,
                    overlap,
                    batch_size,
                    extension,
                    progress_dialog,
                    pbar,
                    operations_completed,
                    total_operations,
                    parent_window,
                    combine_ids=combine_ids,
                    enable_boundary_blend=enable_boundary_blend,
                )

                if batches_completed is None:  # cancelled or failed
                    break

                operations_completed += batches_completed

        except Exception as e:
            pbar.close()
            progress_dialog.close()
            raise

        # Final cleanup
        pbar.close()
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        else:
            progress_dialog.setValue(100)
            if frame_count == frames_to_process:
                self.propagated = True  # only set propagated to True if the entire video was processed
            else:
                self.propagated = False
            print("Matting completed")
            self._notify("matting_complete")
            return 1

    def _load_batch(
        self, abs_start, abs_end, object_id, extension, combine_ids=None, crop_rect=None
    ):
        """
        Load a single batch window of frames and masks from disk, resized using
        the same proportional downscaling as MatAnyone (matany_res setting).

        If combine_ids is provided, masks for all IDs in that list are unioned
        in memory per frame rather than loading a single object mask from disk.

        If crop_rect is provided, both frames and masks are cropped to that region
        before being passed through _resize_image, so the model only processes the
        relevant portion of the frame.

        Args:
            abs_start:   Absolute start frame (inclusive)
            abs_end:     Absolute end frame (exclusive)
            object_id:   Object ID for mask lookup (ignored when combine_ids is set)
            extension:   Frame file extension
            combine_ids: Optional list of object IDs to union into a single mask
            crop_rect:   Optional (x1, y1, x2, y2) from core.compute_mask_bounding_box.
                         When set, frames and masks are cropped before resizing.

        Returns:
            tuple: (cond_frames, mask_frames, valid)
                - cond_frames: list of np.ndarray RGB uint8 (H, W, 3)
                - mask_frames: list of np.ndarray grayscale uint8 (H, W), binarized
                - valid: False if any source frame is missing
        """
        cond_frames = []
        mask_frames = []

        for frame_num in range(abs_start, abs_end):
            frame_path = os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}")
            if not os.path.exists(frame_path):
                print(f"Warning: Frame not found: {frame_path}")
                return [], [], False

            # Load and convert frame to RGB, then proportionally downscale via matany_res
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_rect is not None:
                frame = core.apply_crop(frame, crop_rect)
            frame = self._resize_image(frame)
            resized_h, resized_w = frame.shape[:2]

            # Load mask — union across all combine_ids in memory, or single object from disk
            if combine_ids:
                union_mask = None
                for oid in combine_ids:
                    mask_path = os.path.join(
                        core.mask_dir, f"{frame_num:05d}", f"{oid}.png"
                    )
                    if not os.path.exists(mask_path):
                        continue
                    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if m is None:
                        continue
                    union_mask = m if union_mask is None else np.maximum(union_mask, m)
                if union_mask is not None:
                    union_mask = core.apply_mask_postprocessing(union_mask)
                    if crop_rect is not None:
                        union_mask = core.apply_crop(union_mask, crop_rect)
                    mask = cv2.resize(
                        union_mask,
                        (resized_w, resized_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)
            else:
                mask_path = os.path.join(
                    core.mask_dir, f"{frame_num:05d}", f"{object_id}.png"
                )
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    mask = core.apply_mask_postprocessing(mask)
                    if crop_rect is not None:
                        mask = core.apply_crop(mask, crop_rect)
                    mask = cv2.resize(
                        mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask = np.zeros((resized_h, resized_w), dtype=np.uint8)

            # Binarize mask
            mask = (mask > 127).astype(np.uint8) * 255

            cond_frames.append(frame)
            mask_frames.append(mask)

        return cond_frames, mask_frames, True

    def _generate_windows(self, total_frames, batch_size, overlap):
        """
        Generate sliding window (start, end) pairs covering total_frames.
        end is exclusive. Overlap must be less than batch_size.

        Args:
            total_frames: Total frames to cover
            batch_size: Frames per batch
            overlap: Overlap frames between consecutive batches

        Returns:
            list of (start, end) tuples (relative indices, end exclusive)
        """
        step = batch_size - overlap
        if step <= 0:
            print(
                f"Warning: overlap ({overlap}) >= batch_size ({batch_size}), clamping to batch_size - 1"
            )
            overlap = batch_size - 1
            step = 1

        if total_frames <= batch_size:
            return [(0, total_frames)]

        windows = []
        pos = 0
        while pos < total_frames:
            end = min(pos + batch_size, total_frames)
            windows.append((pos, end))
            if end >= total_frames:
                break
            pos += step

        return windows

    def _process_object(
        self,
        object_id,
        start_frame,
        end_frame,
        overlap,
        batch_size,
        extension,
        progress_dialog,
        pbar,
        operations_completed,
        total_operations,
        parent_window,
        combine_ids=None,
        enable_boundary_blend=True,
    ):
        """
        Process a single object across all frames using windowed batching.

        Args:
            object_id: Object ID to process (also the output file label)
            start_frame: First frame (inclusive)
            end_frame: Last frame (inclusive)
            overlap: Overlap frames between batches
            batch_size: Number of frames per chunk sent to the model
            extension: Frame file extension
            progress_dialog: Qt progress dialog
            pbar: tqdm progress bar
            operations_completed: Batches completed before this object (for progress math)
            total_operations: Total batch count across all objects
            parent_window: Parent window for display updates
            combine_ids: If set, union masks for these IDs in memory rather than
                         loading a single object mask from disk
            enable_boundary_blend: Linearly blend overlap frames at chunk boundaries

        Returns:
            int: Number of batches completed, or None if cancelled/failed
        """
        frames_to_process = end_frame - start_frame + 1
        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting(
            "display_update_frequency", 5
        )

        # Original frame dimensions for restoring output
        first_frame_path = os.path.join(
            core.frames_dir, f"{start_frame:05d}.{extension}"
        )
        first_frame_img = cv2.imread(first_frame_path)
        if first_frame_img is not None:
            original_h, original_w = first_frame_img.shape[:2]
        else:
            original_w = core.VideoInfo.width
            original_h = core.VideoInfo.height

        # Compute a single crop rect covering all mask extents across the full frame range.
        # This is done once before any batch processing so every batch uses an identical
        # spatial region, keeping output consistent across chunk boundaries.
        object_ids_for_bbox = combine_ids if combine_ids is not None else [object_id]
        progress_dialog.setLabelText(
            f"Object {object_id}: Finding region of interest..."
        )
        QApplication.processEvents()

        crop_rect = core.compute_mask_bounding_box(
            frame_range=range(start_frame, end_frame + 1),
            object_ids=object_ids_for_bbox,
            combine_ids=combine_ids,
        )
        if crop_rect is not None:
            cx1, cy1, cx2, cy2 = crop_rect
            print(
                f"Object {object_id}: crop rect ({cx1}, {cy1}) -> ({cx2}, {cy2})  "
                f"[{cx2 - cx1 + 1}x{cy2 - cy1 + 1} of {original_w}x{original_h}]"
            )
        else:
            print(f"Object {object_id}: no mask content found, processing full frame")

        windows = self._generate_windows(frames_to_process, batch_size, overlap)
        batches_completed = 0

        # Stores soft alpha mattes (model working resolution, grayscale uint8) for the
        # last `overlap` frames of each batch.  Fed back as mask_frames for the overlap
        # frames of the next batch, giving the model continuity across chunk boundaries.
        # Kept soft (not binarized) so the model sees graduated edge information.
        previous_overlap_masks = None

        # Stores the last `overlap` committed alpha mattes at original resolution,
        # used for linear blending across the boundary after each non-first batch.
        prev_boundary_alphas = None

        for batch_idx, (window_start, window_end) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return None

            abs_start = start_frame + window_start
            abs_end = start_frame + window_end  # exclusive

            # For non-first batches, the leading overlap frames are warm-up context only —
            # their output is discarded in favour of the already-committed result from the
            # previous batch. Only the non-overlap tail is saved.
            is_first_batch = batch_idx == 0
            output_start_offset = 0 if is_first_batch else overlap

            cond_frames, mask_frames, valid = self._load_batch(
                abs_start,
                abs_end,
                object_id,
                extension,
                combine_ids=combine_ids,
                crop_rect=crop_rect,
            )

            if not valid:
                print(
                    f"Warning: Skipping batch {batch_idx} for object {object_id} "
                    f"(frames {abs_start}-{abs_end - 1}) — missing data"
                )
                pbar.update(1)
                batches_completed += 1
                # Reset both carry-overs so stale data is never injected after a gap
                previous_overlap_masks = None
                prev_boundary_alphas = None
                continue

            # --- FEEDBACK LOOP INJECTION ---
            # Replace the SAM2 masks for the overlap frames with the soft alpha mattes
            # predicted by the previous batch.  The model sees its own prior output as
            # guidance, encouraging consistent alpha values across the chunk boundary.
            if not is_first_batch and previous_overlap_masks is not None:
                for i in range(
                    min(overlap, len(mask_frames), len(previous_overlap_masks))
                ):
                    mask_frames[i] = previous_overlap_masks[i]
            # -------------------------------

            def _on_pipeline_progress(step, total, desc):
                progress_dialog.setLabelText(
                    f"Batch {batch_idx + 1}/{len(windows)} — {desc}"
                )
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("USER_CANCELLED")

            try:
                with torch.amp.autocast("cuda", enabled=False):
                    output_frames = self.pipeline.run(
                        cond_frames=cond_frames,
                        mask_frames=mask_frames,
                        seed=42,
                        progress_callback=_on_pipeline_progress,
                    )
            except RuntimeError as e:
                if str(e) == "USER_CANCELLED":
                    return None  # signals cancel upstream
                raise
            except Exception as e:
                print(f"Error in VideoMaMa inference for batch {batch_idx}: {e}")
                raise

            # --- CAPTURE SOFT MASKS FOR NEXT BATCH ---
            # Store the last `overlap` frames of the current output at model resolution
            # (before _restore_image_size) as soft grayscale — NOT binarized, so the
            # model receives graduated edge values rather than a hard binary boundary.
            previous_overlap_masks = []
            for frame_out in output_frames[-overlap:]:
                alpha_out = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)
                previous_overlap_masks.append(alpha_out)
            # -----------------------------------------

            # Discard leading overlap output on non-first batches
            committed_output = output_frames[output_start_offset:]

            # --- LINEAR BLEND AT CHUNK BOUNDARIES ---
            # The overlap frames were re-processed by the new batch, giving us a second
            # prediction for those already-committed frames (output_frames[0:overlap]).
            # We linearly blend the original committed alpha (prev_boundary_alphas) with
            # the new prediction across the overlap window and re-write those frames.
            if (
                enable_boundary_blend
                and not is_first_batch
                and prev_boundary_alphas is not None
            ):
                for i in range(
                    min(overlap, len(prev_boundary_alphas), len(output_frames))
                ):
                    new_alpha = cv2.cvtColor(output_frames[i], cv2.COLOR_RGB2GRAY)
                    new_alpha = self._restore_image_size(
                        new_alpha,
                        (cx2 - cx1 + 1, cy2 - cy1 + 1)
                        if crop_rect
                        else (original_w, original_h),
                    )
                    if crop_rect is not None:
                        new_alpha = core.expand_to_full(
                            new_alpha, crop_rect, original_w, original_h
                        )
                    new_weight = (i + 1) / (overlap + 1)
                    blended = (
                        (
                            (1.0 - new_weight)
                            * prev_boundary_alphas[i].astype(np.float32)
                            + new_weight * new_alpha.astype(np.float32)
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    abs_blend_frame = abs_start + i
                    mat_filename = os.path.join(
                        core.matting_dir, f"{abs_blend_frame:05d}", f"{object_id}.png"
                    )
                    os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                    cv2.imwrite(mat_filename, blended)
            # -----------------------------------------

            # Collect the last `overlap` committed alphas (original resolution) for the
            # next batch's boundary blend, then write all committed frames to disk.
            current_boundary_alphas = []
            for i, frame_out in enumerate(committed_output):
                abs_frame = abs_start + output_start_offset + i
                alpha = cv2.cvtColor(frame_out, cv2.COLOR_RGB2GRAY)

                # Restore to the cropped region's pixel dimensions first, then expand
                # back into the full frame canvas so the saved matte is always full-size.
                if crop_rect is not None:
                    crop_w = cx2 - cx1 + 1
                    crop_h = cy2 - cy1 + 1
                    alpha = self._restore_image_size(alpha, (crop_w, crop_h))
                    final_alpha = core.expand_to_full(
                        alpha, crop_rect, original_w, original_h
                    )
                else:
                    final_alpha = self._restore_image_size(
                        alpha, (original_w, original_h)
                    )

                if i >= len(committed_output) - overlap:
                    current_boundary_alphas.append(final_alpha.copy())

                mat_filename = os.path.join(
                    core.matting_dir, f"{abs_frame:05d}", f"{object_id}.png"
                )
                os.makedirs(os.path.dirname(mat_filename), exist_ok=True)
                cv2.imwrite(mat_filename, final_alpha)

                if abs_frame % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(abs_frame)
                        QApplication.processEvents()
                    except Exception:
                        pass

            prev_boundary_alphas = (
                current_boundary_alphas
                if len(current_boundary_alphas) == overlap
                else None
            )

            # Update progress
            pbar.update(1)
            batches_completed += 1
            current_progress = int(
                ((operations_completed + batches_completed) * 100)
                / max(total_operations, 1)
            )
            progress_dialog.setValue(min(current_progress, 99))
            QApplication.processEvents()

            core.DeviceManager.clear_cache()

        return batches_completed


# ---------------------------------------------------------------------------
# LTX-2.5 Alpha-Gen backend
# ---------------------------------------------------------------------------


class LTXAlphaManager(MattingManager):
    """
    Matting manager that uses LTX-2.5 IC-LoRA Alpha-Gen.

    The model has no mask input, so each SAM2 mask is grown and feathered, then used to
    premultiply the frame onto black; the model mattes what is left. Frames are cropped
    to the subject, scaled so the crop's short side matches ltx_res, padded to multiples
    of 32 and processed in 8n+1-frame batches with a linear crossfade between batches.
    """

    BACKEND = "ltx_alpha"
    GROW_PCT = 15.0     # mask dilation, % of the subject's short side
    FEATHER_PCT = 2.0   # gaussian feather sigma, % of the subject's short side

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def unload_matting_model(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded LTX Alpha model")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        from PySide6.QtWidgets import QMessageBox
        from sammie.ltx_alpha import LTX_DIR, LTXAlphaPipeline, fp8_supported, missing_code, missing_files

        missing = missing_code()
        if missing and ensure_models("ltx_code", parent=parent_window, title="LTX-2 code", confirm=True):
            import importlib

            importlib.invalidate_caches()
            missing = missing_code()
        if missing:
            QMessageBox.warning(
                parent_window,
                "LTX-2 Code Missing",
                "LTX Alpha needs Lightricks' LTX-2 packages, which are installed separately:\n\n"
                + "\n".join(missing)
                + "\n\nSee \"LTX Alpha setup\" in README.md.",
            )
            return False
        device = self._prepare_device(load_to_cpu)
        missing = missing_files(fp8=fp8_supported(device))
        if missing and fp8_supported(device):
            # The prebuilt fp8 files exist only for NVIDIA; other GPUs need the bf16 build (README)
            specs = [s for s in MODEL_GROUPS["ltx_alpha_fp8"]
                     if any(s.filename in m for m in missing) or s.filename == "LICENSE-2_x"]
            if not ensure_models(specs, parent=parent_window, title="LTX Alpha models", confirm=True):
                return False
            missing = missing_files(fp8=True)
        if missing:
            QMessageBox.warning(
                parent_window,
                "Model Files Missing",
                "LTX Alpha needs these files in\n"
                f"{os.path.abspath(LTX_DIR)}:\n\n" + "\n".join(missing)
                + "\n\nSee \"LTX Alpha setup\" in README.md.",
            )
            return False

        try:
            self.pipeline = LTXAlphaPipeline(device)
        except Exception as e:
            print(f"Error loading LTX Alpha pipeline: {e}")
            self.pipeline = None
            return False

        settings_mgr = get_settings_manager()
        print(
            f"Loaded LTX Alpha to {device} (offload={self.pipeline.offload.name}, "
            f"cache_weights={self.pipeline.cache_weights}) with res={settings_mgr.get_session_setting('ltx_res', 736)}, "
            f"batch={settings_mgr.get_session_setting('ltx_chunk', 97)}, "
            f"overlap={settings_mgr.get_session_setting('ltx_overlap', 8)}"
        )
        return True

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        if self.pipeline is None:
            print("LTX Alpha model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        settings_mgr = get_settings_manager()
        batch_size = settings_mgr.get_session_setting("ltx_chunk", 97)
        overlap = settings_mgr.get_session_setting("ltx_overlap", 8)

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        object_ids = sorted(set(point["object_id"] for point in points_list if "object_id" in point))
        if not object_ids:
            print("No objects found for matting")
            return 0

        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        windows = self._generate_windows(frames_to_process, batch_size, overlap)
        total_operations = len(windows) * len(object_ids)
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations, unit="batch")
        operations_completed = 0

        try:
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break
                pbar.set_description(f"Object {object_id}")
                done = self._process_object(
                    object_id, start_frame, end_frame, windows, overlap, progress_dialog, pbar,
                    operations_completed, total_operations, parent_window, combine_ids=combine_ids,
                )
                if done is None:
                    break
                operations_completed += done
        except Exception:
            pbar.close()
            progress_dialog.close()
            raise

        pbar.close()
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        progress_dialog.setValue(100)
        self.propagated = frame_count == frames_to_process
        print("Matting completed")
        self._notify("matting_complete")
        return 1

    @staticmethod
    def _generate_windows(total_frames, batch_size, overlap):
        """(start, end) windows, end exclusive, relative to the range start.

        Full windows are batch_size frames. Consecutive windows overlap by at least
        `overlap`. The last window is the shortest 8n+1 length that covers the remaining
        frames plus the overlap, aligned to the end, so it needs no padding frames and
        recomputes as little as possible. Only a range shorter than one batch can come
        out off the 8n+1 grid; that single window gets padded at run time.
        """
        from sammie.ltx_alpha import snap_up_8n1

        if total_frames <= batch_size:
            return [(0, total_frames)]
        step = max(1, batch_size - overlap)
        windows = [(0, batch_size)]
        while windows[-1][1] < total_frames:
            pos = windows[-1][0] + step
            if pos + batch_size >= total_frames:
                needed = total_frames - (windows[-1][1] - overlap)
                length = min(batch_size, snap_up_8n1(needed))
                windows.append((total_frames - length, total_frames))
            else:
                windows.append((pos, pos + batch_size))
        return windows

    def _load_mask(self, frame_num, object_id, combine_ids, shape):
        """Binary SAM2 mask (with segmentation postprocessing), union for combined mode."""
        ids = combine_ids if combine_ids else [object_id]
        union = None
        for oid in ids:
            m = cv2.imread(os.path.join(core.mask_dir, f"{frame_num:05d}", f"{oid}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                union = m if union is None else np.maximum(union, m)
        if union is None:
            return np.zeros(shape, np.uint8)
        return core.apply_mask_postprocessing(union)

    def _subject_region(self, start_frame, end_frame, object_id, combine_ids, full_w, full_h):
        """Grow/feather sizes and the crop rect, from the union of masks over the range."""
        x1 = y1 = None
        for frame_num in range(start_frame, end_frame + 1):
            m = self._load_mask(frame_num, object_id, combine_ids, (full_h, full_w))
            ys, xs = np.nonzero(m > 127)
            if len(xs) == 0:
                continue
            if x1 is None:
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            else:
                x1, y1 = min(x1, xs.min()), min(y1, ys.min())
                x2, y2 = max(x2, xs.max() + 1), max(y2, ys.max() + 1)
        if x1 is None:
            return None
        subject = min(x2 - x1, y2 - y1)
        grow_px = max(1, int(round(subject * self.GROW_PCT / 100)))
        sigma = max(0.5, subject * self.FEATHER_PCT / 100)
        pad = grow_px + int(np.ceil(3 * sigma)) + 8
        crop = (max(0, x1 - pad), max(0, y1 - pad), min(full_w, x2 + pad), min(full_h, y2 + pad))
        return grow_px, sigma, crop

    @staticmethod
    def _working_size(crop_w, crop_h, res):
        """Scale the crop's short side to res (0 = native, never upscale), then pad to /32."""
        scale = 1.0 if res <= 0 else min(1.0, res / min(crop_w, crop_h))
        w = max(32, int(round(crop_w * scale)))
        h = max(32, int(round(crop_h * scale)))
        return w, h, ((w + 31) // 32) * 32, ((h + 31) // 32) * 32

    @staticmethod
    def _soft_grown(mask, grow_px, sigma):
        """Grow a binary mask by grow_px (distance transform, O(N)), then feather it."""
        dist = cv2.distanceTransform((mask <= 127).astype(np.uint8), cv2.DIST_L2, 5)
        return cv2.GaussianBlur((dist <= grow_px).astype(np.float32), (0, 0), sigma)

    def _process_object(self, object_id, start_frame, end_frame, windows, overlap, progress_dialog,
                        pbar, operations_completed, total_operations, parent_window, combine_ids=None):
        from sammie.ltx_alpha import snap_up_8n1

        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        res = settings_mgr.get_session_setting("ltx_res", 736)
        extension = core.get_frame_extension()
        fps = core.VideoInfo.fps or 24.0

        first = cv2.imread(os.path.join(core.frames_dir, f"{start_frame:05d}.{extension}"))
        full_h, full_w = first.shape[:2] if first is not None else (core.VideoInfo.height, core.VideoInfo.width)

        progress_dialog.setLabelText(f"Object {object_id}: Finding region of interest...")
        QApplication.processEvents()
        region = self._subject_region(start_frame, end_frame, object_id, combine_ids, full_w, full_h)
        if region is None:
            print(f"Object {object_id}: no mask content in range, skipping")
            pbar.update(len(windows))
            return len(windows)
        grow_px, sigma, (cx1, cy1, cx2, cy2) = region
        cw, ch = cx2 - cx1, cy2 - cy1
        w, h, pw, ph = self._working_size(cw, ch, res)
        print(f"Object {object_id}: crop ({cx1},{cy1})-({cx2},{cy2}) {cw}x{ch} -> {w}x{h} padded {pw}x{ph}, "
              f"grow {grow_px}px, feather {sigma:.1f}px")

        prev_end = None
        prev_tail = {}  # frame -> float alpha (crop res) for the last `overlap` frames of the previous batch
        batches_completed = 0

        for batch_idx, (ws, we) in enumerate(windows):
            if progress_dialog.wasCanceled():
                return None
            abs_start, abs_end = start_frame + ws, start_frame + we
            progress_dialog.setLabelText(f"Object {object_id}: batch {batch_idx + 1}/{len(windows)} - preparing")
            QApplication.processEvents()

            # --- premultiplied, cropped, scaled, padded reference frames ---
            soft_masks, reference = [], []
            for frame_num in range(abs_start, abs_end):
                frame = cv2.imread(os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}"))
                if frame is None:
                    raise FileNotFoundError(f"Frame not found: {frame_num}")
                frame = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
                mask = self._load_mask(frame_num, object_id, combine_ids, (full_h, full_w))[cy1:cy2, cx1:cx2]
                soft = self._soft_grown(mask, grow_px, sigma)
                premult = (frame * soft[..., None]).round().astype(np.uint8)
                padded = np.zeros((ph, pw, 3), np.uint8)
                padded[:h, :w] = cv2.resize(premult, (w, h), interpolation=cv2.INTER_AREA)
                soft_masks.append(soft)
                reference.append(padded)
            # Only a range shorter than one batch can miss the 8n+1 grid: repeat the last frame.
            while len(reference) < snap_up_8n1(len(reference)):
                reference.append(reference[-1])

            def _on_step(step, total, _batch=batch_idx):
                progress_dialog.setLabelText(
                    f"Object {object_id}: batch {_batch + 1}/{len(windows)} - denoising step {step}/{total}"
                )
                done = operations_completed + batches_completed + (step / max(total, 1))
                progress_dialog.setValue(min(int(done * 100 / max(total_operations, 1)), 99))
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    raise RuntimeError("USER_CANCELLED")

            try:
                alphas = self.pipeline.run(reference, fps, core.temp_dir, step_callback=_on_step)
            except RuntimeError as e:
                if str(e) == "USER_CANCELLED":
                    return None
                raise

            # --- back to crop resolution, clamped to the grown selection ---
            new = {}
            for k, frame_num in enumerate(range(abs_start, abs_end)):
                a = cv2.resize(alphas[k][:h, :w], (cw, ch), interpolation=cv2.INTER_LINEAR)
                new[frame_num] = a * soft_masks[k]

            # Frames already committed by the previous batch keep their value, except the
            # last `overlap` of them, which crossfade linearly into this batch's result.
            if prev_end is None:
                commit_from = abs_start
            else:
                commit_from = prev_end
                blend_start = max(abs_start, prev_end - overlap)
                span = prev_end - blend_start
                for i, frame_num in enumerate(range(blend_start, prev_end)):
                    if frame_num in prev_tail:
                        wt = (i + 1) / (span + 1)
                        self._write_matte((1 - wt) * prev_tail[frame_num] + wt * new[frame_num],
                                          frame_num, object_id, (cx1, cy1, cx2, cy2), full_w, full_h)
            for frame_num in range(commit_from, abs_end):
                self._write_matte(new[frame_num], frame_num, object_id, (cx1, cy1, cx2, cy2), full_w, full_h)
                if frame_num % display_update_frequency == 0:
                    try:
                        parent_window.frame_slider.setValue(frame_num)
                    except Exception:
                        pass

            prev_tail = {f: new[f] for f in range(max(abs_start, abs_end - overlap), abs_end)}
            prev_end = abs_end

            pbar.update(1)
            batches_completed += 1
            progress_dialog.setValue(
                min(int((operations_completed + batches_completed) * 100 / max(total_operations, 1)), 99)
            )
            QApplication.processEvents()
            core.DeviceManager.clear_cache()

        return batches_completed

    @staticmethod
    def _write_matte(alpha, frame_num, object_id, crop, full_w, full_h):
        x1, y1, x2, y2 = crop
        full = np.zeros((full_h, full_w), np.uint8)
        full[y1:y2, x1:x2] = (alpha * 255).round().clip(0, 255).astype(np.uint8)
        path = os.path.join(core.matting_dir, f"{frame_num:05d}", f"{object_id}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cv2.imwrite(path, full)


# ---------------------------------------------------------------------------
# CorridorKey backend
# ---------------------------------------------------------------------------

class CorridorKeyManager(MattingManager):
    """
    Matting manager that uses CorridorKey 2.0.

    The tracked SAM masks are passed as CorridorKey's full per-frame hint, so SAM picks the
    subject and CorridorKey keys it. Besides the matte it writes the predicted foreground
    colour to core.matting_fg_dir, used by the CorridorKey views and exports.
    """

    BACKEND = "corridorkey"

    def __init__(self):
        super().__init__()
        self.pipeline = None

    def unload_matting_model(self):
        if self.pipeline is not None:
            self.pipeline.close()
            self.pipeline = None
        gc.collect()
        core.DeviceManager.clear_cache()
        print("Unloaded CorridorKey model")

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        from PySide6.QtWidgets import QMessageBox
        from sammie.corridorkey import CK_DIR, CorridorKeyPipeline, missing_code, missing_files

        # Check the GPU first, so nothing is downloaded on a machine that cannot run it
        device = self._prepare_device(load_to_cpu)
        if device.type != "cuda":
            QMessageBox.warning(
                parent_window,
                "CorridorKey Not Supported",
                "CorridorKey needs an NVIDIA GPU (CUDA). It does not run on Apple Silicon or on the CPU.",
            )
            return False

        # Download whatever is missing (nothing, on an offline copy with everything in place)
        missing = missing_code()
        if missing:
            if not ensure_models(missing, parent=parent_window, title="CorridorKey code", confirm=True):
                return False
            missing = missing_code()
            if missing:
                QMessageBox.warning(
                    parent_window,
                    "CorridorKey Code Missing",
                    "CorridorKey's code could not be set up:\n\n" + "\n".join(missing)
                    + "\n\nSee \"CorridorKey setup\" in README.md.",
                )
                return False
        if missing_files() and not ensure_models("corridorkey", parent=parent_window,
                                                 title="CorridorKey models", confirm=True):
            return False
        missing = missing_files()
        if missing:
            QMessageBox.warning(
                parent_window,
                "Model Files Missing",
                "CorridorKey needs these files in\n"
                f"{os.path.abspath(CK_DIR)}:\n\n" + "\n".join(missing)
                + "\n\nSee \"CorridorKey setup\" in README.md.",
            )
            return False

        try:
            self.pipeline = CorridorKeyPipeline(device)
        except Exception as e:
            print(f"Error loading CorridorKey: {e}")
            self.pipeline = None
            return False

        settings_mgr = get_settings_manager()
        print(
            f"Loaded CorridorKey to {device} with hint trust={settings_mgr.get_session_setting('ck_hint_quality', 0.75)}, "
            f"window={settings_mgr.get_session_setting('ck_chunk', 0) or 'Auto'}, "
            f"carry={settings_mgr.get_session_setting('ck_carry', True)}"
        )
        return True

    def run_matting(self, points_list, parent_window, combined=False):
        if self.pipeline is None:
            print("CorridorKey model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        frame_count = core.VideoInfo.total_frames
        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        object_ids = sorted(set(point["object_id"] for point in points_list if "object_id" in point))
        if not object_ids:
            print("No objects found for matting")
            return 0

        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))
            core.clear_matting_fg()

        total_operations = frames_to_process * len(object_ids)
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)
        operations_completed = 0

        try:
            for object_id in object_ids:
                if progress_dialog.wasCanceled():
                    break
                pbar.set_description(f"Object {object_id}")
                done = self._process_object(
                    object_id, start_frame, end_frame, progress_dialog, pbar,
                    operations_completed, total_operations, parent_window, combine_ids=combine_ids,
                )
                if done is None:
                    break
                operations_completed += done
        except Exception:
            pbar.close()
            progress_dialog.close()
            raise

        pbar.close()
        core.DeviceManager.clear_cache()

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            self.propagated = False
            progress_dialog.close()
            return 0
        progress_dialog.setValue(100)
        self.propagated = frame_count == frames_to_process
        print("Matting completed")
        self._notify("matting_complete")
        return 1

    def _load_hint(self, frame_num, object_id, combine_ids, shape):
        """Binary SAM2 mask (with segmentation postprocessing), union for combined mode."""
        ids = combine_ids if combine_ids else [object_id]
        union = None
        for oid in ids:
            m = cv2.imread(os.path.join(core.mask_dir, f"{frame_num:05d}", f"{oid}.png"), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                union = m if union is None else np.maximum(union, m)
        if union is None:
            return np.zeros(shape, np.uint8)
        return core.apply_mask_postprocessing(union)

    def _process_object(self, object_id, start_frame, end_frame, progress_dialog, pbar,
                        operations_completed, total_operations, parent_window, combine_ids=None):
        import shutil

        settings_mgr = get_settings_manager()
        display_update_frequency = settings_mgr.get_app_setting("display_update_frequency", 5)
        res = settings_mgr.get_session_setting("matany_res", 0)
        extension = core.get_frame_extension()
        frames_to_process = end_frame - start_frame + 1

        # The runtime reads a folder of frames and a matching folder of hints, so give it
        # just the processing range: hard links where the filesystem allows, or frames scaled
        # down to the internal resolution (the runtime itself only renders at native size).
        work_dir = os.path.join(core.temp_dir, "corridorkey_work")
        shutil.rmtree(work_dir, ignore_errors=True)
        frame_dir = os.path.join(work_dir, "frames")
        hint_dir = os.path.join(work_dir, "hints")
        os.makedirs(frame_dir)
        os.makedirs(hint_dir)

        progress_dialog.setLabelText(f"Object {object_id}: Preparing frames and hints...")
        QApplication.processEvents()
        first = cv2.imread(os.path.join(core.frames_dir, f"{start_frame:05d}.{extension}"))
        full_h, full_w = first.shape[:2]
        scale = 1.0 if res <= 0 else min(1.0, res / min(full_h, full_w))
        work_size = (max(64, round(full_w * scale)), max(64, round(full_h * scale)))
        has_hint = False
        for frame_num in range(start_frame, end_frame + 1):
            src = os.path.join(core.frames_dir, f"{frame_num:05d}.{extension}")
            dst = os.path.join(frame_dir, f"{frame_num:05d}.{extension}")
            hint = self._load_hint(frame_num, object_id, combine_ids, (full_h, full_w))
            has_hint = has_hint or bool(hint.any())
            if scale < 1.0:
                cv2.imwrite(dst, cv2.resize(cv2.imread(src), work_size, interpolation=cv2.INTER_AREA))
                hint = cv2.resize(hint, work_size, interpolation=cv2.INTER_AREA)
            else:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            cv2.imwrite(os.path.join(hint_dir, f"{frame_num:05d}.png"), hint)
            if progress_dialog.wasCanceled():
                shutil.rmtree(work_dir, ignore_errors=True)
                return None
        if not has_hint:
            print(f"Object {object_id}: no mask content in range, skipping")
            shutil.rmtree(work_dir, ignore_errors=True)
            pbar.update(frames_to_process)
            return frames_to_process

        written = {"alpha": 0, "fg": 0}

        def _on_frames(kind, frames):
            for frame in frames:
                frame_num = start_frame + written[kind]
                written[kind] += 1
                out = frame[..., 0] if kind == "alpha" else frame
                if scale < 1.0:
                    out = cv2.resize(out, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
                out = (out.astype(np.float32) / 257.0).round().astype(np.uint8)
                if kind == "alpha":
                    path = os.path.join(core.matting_dir, f"{frame_num:05d}", f"{object_id}.png")
                else:
                    path = os.path.join(core.matting_fg_dir, f"{frame_num:05d}", f"{object_id}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                cv2.imwrite(path, out)
                if kind == "alpha":
                    pbar.update(1)
                    if frame_num % display_update_frequency == 0:
                        try:
                            parent_window.frame_slider.setValue(frame_num)
                        except Exception:
                            pass
            progress_dialog.setLabelText(
                f"Object {object_id}: frame {written['alpha']}/{frames_to_process}")
            done = operations_completed + written["alpha"]
            progress_dialog.setValue(min(int(done * 100 / max(total_operations, 1)), 99))
            QApplication.processEvents()
            if progress_dialog.wasCanceled():
                raise RuntimeError("USER_CANCELLED")

        progress_dialog.setLabelText(f"Object {object_id}: Extracting features...")
        QApplication.processEvents()
        try:
            self.pipeline.run(
                frame_dir, hint_dir, os.path.join(work_dir, "run"), _on_frames,
                fps=core.VideoInfo.fps or 24.0,
                hint_quality=float(settings_mgr.get_session_setting("ck_hint_quality", 0.75)),
                carry_hint=bool(settings_mgr.get_session_setting("ck_carry", True)),
                chunk=int(settings_mgr.get_session_setting("ck_chunk", 0)),
            )
        except RuntimeError as e:
            if str(e) == "USER_CANCELLED":
                return None
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        return frames_to_process


# ---------------------------------------------------------------------------
# ViTMatte backend
# ---------------------------------------------------------------------------

class ViTMatteManager(MattingManager):
    """
    Matting manager stub for ViTMatte.
    Trimap-based single-image model — not yet implemented.
    """

    BACKEND = "vitmatte"

    _KEY_MAP = {
        "ViTMatte Small": "vitmatte_small",
        "ViTMatte Base": "vitmatte_base",
    }

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone")
        registry_key = self._KEY_MAP.get(matting_model, "vitmatte_base")

        if not ensure_models(registry_key, parent=parent_window):
            return False
        d2 = _import_detectron2(parent_window)
        if d2 is None:
            return False
        LazyConfig, instantiate, DetectionCheckpointer = d2

        device = 'cpu' if load_to_cpu else core.DeviceManager.get_device()

        config_path = os.path.join(os.path.dirname(__file__), "vitmatte", "configs", "common", "model.py")
        cfg = LazyConfig.load(config_path)

        if registry_key == "vitmatte_base":
            cfg.model.backbone.embed_dim = 768
            cfg.model.backbone.num_heads = 12
            cfg.model.decoder.in_chans = 768

        self.model = instantiate(cfg.model)
        self.model.to(device)
        self.model.eval()

        checkpoint_path = str(MODEL_REGISTRY[registry_key].final_path)
        DetectionCheckpointer(self.model).load(checkpoint_path)

        print(f"Loaded ViTMatte ({registry_key}) model to {device}")
        return True

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        if self.model is None:
            print("Matting model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        total_operations = len(object_ids) * frames_to_process
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        images = self._collect_image_paths(start_frame, end_frame)
        operations_completed = 0

        for object_id in object_ids:
            if progress_dialog.wasCanceled():
                break
            pbar.set_description(f"Object {object_id}")

            for frame_path in images:
                if progress_dialog.wasCanceled():
                    break

                frame_number = int(os.path.splitext(os.path.basename(frame_path))[0])
                mask, original_size = self._load_mask_for_matting(object_id, frame_number, device, combine_ids=combine_ids)

                if mask is not None:
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self._resize_image(image)  # match the mask's working resolution

                    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
                    trimap = generate_trimap(mask_np)

                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
                    trimap_tensor = torch.from_numpy(trimap).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

                    output = self.model({"image": image_tensor, "trimap": trimap_tensor})["phas"]
                    alpha = output.squeeze().clamp(0, 1).mul(255).byte().cpu().numpy()
                    alpha = self._restore_image_size(alpha, original_size)

                    frame_dir = os.path.join(core.matting_dir, f"{frame_number:05d}")
                    os.makedirs(frame_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(frame_dir, f"{object_id}.png"), alpha)

                operations_completed += 1
                pbar.update(1)
                progress_dialog.setValue(min(int(operations_completed / total_operations * 100), 99))
                QApplication.processEvents()

        core.DeviceManager.clear_cache()
        pbar.close()
        progress_dialog.setValue(100)

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            return 0

        print("Matting completed")
        self.propagated = True
        return 1


# ---------------------------------------------------------------------------
# MEMatte backend
# ---------------------------------------------------------------------------


class MEMatteManager(MattingManager):
    BACKEND = "mematte"

    _KEY_MAP = {
        "MEMatte Small": "mematte_vits",
        "MEMatte Base": "mematte_vitb",
        "MEMatte Robust": "mematte_robust",
    }

    def load_matting_model(self, load_to_cpu=False, parent_window=None):
        settings_mgr = get_settings_manager()
        matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone")
        registry_key = self._KEY_MAP.get(matting_model, "mematte_robust")

        if not ensure_local_model(registry_key, parent=parent_window):
            return False
        d2 = _import_detectron2(parent_window)
        if d2 is None:
            return False
        LazyConfig, instantiate, DetectionCheckpointer = d2

        device = 'cpu' if load_to_cpu else core.DeviceManager.get_device()

        config_path = os.path.join(os.path.dirname(__file__), "mematte", "configs", "common", "model.py")
        cfg = LazyConfig.load(config_path)

        cfg.model.backbone.use_rel_pos = True
        cfg.model.backbone.topk = 0.25
        cfg.model.backbone.window_block_indexes = [0, 1, 3, 4, 6, 7, 9, 10]
        cfg.model.backbone.multi_score = True

        if registry_key == "mematte_vitb":
            cfg.model.backbone.embed_dim = 768
            cfg.model.backbone.num_heads = 12
            cfg.model.decoder.in_chans = 768

        cfg.model.teacher_backbone = None
        cfg.model.backbone.max_number_token = 20000 # ceiling based on the mematte paper, should be good up to 6K

        self.model = instantiate(cfg.model)
        self.model.to(device)
        self.model.eval()

        checkpoint_path = str(MODEL_REGISTRY[registry_key].final_path)
        DetectionCheckpointer(self.model).load(checkpoint_path)

        print(f"Loaded MEMatte ({registry_key}) model to {device} "
              f"(max_number_token={cfg.model.backbone.max_number_token})")
        return True

    @torch.inference_mode()
    def run_matting(self, points_list, parent_window, combined=False):
        if self.model is None:
            print("Matting model not loaded")
            return 0

        core.DeviceManager.clear_cache()
        device = core.DeviceManager.get_device()

        start_frame, end_frame, frames_to_process = self._get_frame_range()
        print(f"Processing matting from frame {start_frame} to {end_frame} ({frames_to_process} frames)")

        object_ids = sorted(list(set(point['object_id'] for point in points_list if 'object_id' in point)))
        if not object_ids:
            print("No objects found for matting")
            return 0

        if combined and len(object_ids) > 1:
            combine_ids = object_ids
            object_ids = [0]
        else:
            combine_ids = None

        total_operations = len(object_ids) * frames_to_process
        progress_dialog, pbar = self._make_progress_dialog(parent_window, total_operations)

        os.makedirs(core.matting_dir, exist_ok=True)
        if combined and os.path.exists(core.matting_dir):
            for frame_dirname in os.listdir(core.matting_dir):
                frame_dir = os.path.join(core.matting_dir, frame_dirname)
                if os.path.isdir(frame_dir):
                    for f in os.listdir(frame_dir):
                        if f != "0.png":
                            os.remove(os.path.join(frame_dir, f))

        images = self._collect_image_paths(start_frame, end_frame)
        operations_completed = 0

        for object_id in object_ids:
            if progress_dialog.wasCanceled():
                break
            pbar.set_description(f"Object {object_id}")

            for frame_path in images:
                if progress_dialog.wasCanceled():
                    break

                frame_number = int(os.path.splitext(os.path.basename(frame_path))[0])
                mask, original_size = self._load_mask_for_matting(object_id, frame_number, device, combine_ids=combine_ids)

                if mask is not None:
                    image = cv2.imread(frame_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = self._resize_image(image)

                    mask_np = mask.detach().cpu().numpy().astype(np.uint8)
                    trimap = generate_trimap(mask_np)

                    image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
                    trimap_tensor = torch.from_numpy(trimap).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)

                    output, _, _ = self.model({"image": image_tensor, "trimap": trimap_tensor}, patch_decoder=True)
                    output = output['phas'].flatten(0, 2)

                    trimap_flat = trimap_tensor.squeeze(0).squeeze(0)
                    output[trimap_flat == 0] = 0
                    output[trimap_flat == 1] = 1

                    alpha = output.clamp(0, 1).mul(255).byte().cpu().numpy()
                    alpha = self._restore_image_size(alpha, original_size)

                    frame_dir = os.path.join(core.matting_dir, f"{frame_number:05d}")
                    os.makedirs(frame_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(frame_dir, f"{object_id}.png"), alpha)

                operations_completed += 1
                pbar.update(1)
                progress_dialog.setValue(min(int(operations_completed / total_operations * 100), 99))
                QApplication.processEvents()

        core.DeviceManager.clear_cache()
        pbar.close()
        progress_dialog.setValue(100)

        if progress_dialog.wasCanceled():
            print("Matting cancelled")
            return 0

        print("Matting completed")
        self.propagated = True
        return 1

# ---------------------------------------------------------------------------
# Manager Selection
# ---------------------------------------------------------------------------


def create_matting_manager() -> MattingManager:
    """
    Instantiate the correct matting manager based on the 'matting_model' session setting.
    """
    settings_mgr = get_settings_manager()
    matting_model = settings_mgr.get_session_setting("matany_model", "MatAnyone")


    if matting_model == "VideoMaMa":
        return VideoMaMaManager()
    if matting_model == "LTX Alpha":
        return LTXAlphaManager()
    if matting_model == "CorridorKey":
        return CorridorKeyManager()
    if matting_model.startswith("ViTMatte"):
        return ViTMatteManager()
    if matting_model.startswith("MEMatte"):
        return MEMatteManager()
    return MatAnyManager()
