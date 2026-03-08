from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pydicom
import torch
from PIL import Image, ImageDraw

# Allow importing project modules when backend is run from Website/backend
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import LUNA16_Dataset, sliding_window_3d  # noqa: E402
from model import VitDet3D  # noqa: E402


@dataclass
class VolumeMetadata:
    origin_xyz: np.ndarray
    spacing_xyz: np.ndarray
    source_files: int


class ModelRunner:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint: str | None = None
        self.model: VitDet3D | None = None

    def load(self, checkpoint_path: Path) -> VitDet3D:
        checkpoint_abs = str(checkpoint_path.resolve())
        if self.model is not None and self._checkpoint == checkpoint_abs:
            return self.model

        model = VitDet3D.from_pretrained(checkpoint_abs)
        model = model.to(self.device).eval()
        self.model = model
        self._checkpoint = checkpoint_abs
        return model


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_origin_xyz(ds: pydicom.Dataset) -> np.ndarray:
    if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
        return np.array([_safe_float(v, 0.0) for v in ds.ImagePositionPatient[:3]], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _parse_spacing_xyz(ds: pydicom.Dataset, z_spacing: float | None) -> np.ndarray:
    pixel_spacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    spacing_y = _safe_float(pixel_spacing[0], 1.0) if len(pixel_spacing) > 0 else 1.0
    spacing_x = _safe_float(pixel_spacing[1], 1.0) if len(pixel_spacing) > 1 else spacing_y
    spacing_z = z_spacing if z_spacing is not None else _safe_float(getattr(ds, "SliceThickness", 1.0), 1.0)
    return np.array([spacing_x, spacing_y, spacing_z], dtype=np.float32)


def _sort_key(item: tuple[Path, pydicom.Dataset]) -> tuple[Any, ...]:
    path, ds = item
    if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
        return (0, _safe_float(ds.ImagePositionPatient[2], 0.0), _safe_float(getattr(ds, "InstanceNumber", 0), 0), str(path))
    return (1, _safe_float(getattr(ds, "InstanceNumber", 0), 0), str(path))


def _series_key(ds: pydicom.Dataset, arr: np.ndarray) -> tuple[str, tuple[int, int]]:
    series_uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
    return series_uid, (int(arr.shape[0]), int(arr.shape[1]))


def load_dicom_series(folder: Path) -> tuple[np.ndarray, VolumeMetadata]:
    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        raise ValueError(f"鏈湪鐩綍涓壘鍒版枃浠? {folder}")

    dicom_items: list[tuple[Path, pydicom.Dataset, np.ndarray]] = []
    for file_path in files:
        try:
            ds = pydicom.dcmread(str(file_path), force=True)
            if not hasattr(ds, "PixelData"):
                continue

            arr = ds.pixel_array
            if arr.ndim != 2:
                continue

            dicom_items.append((file_path, ds, arr.astype(np.float32)))
        except Exception:
            continue

    if not dicom_items:
        raise ValueError(f"鏈湪鐩綍涓壘鍒板彲璇诲彇鐨?DICOM 褰卞儚: {folder}")

    series_counts = Counter(_series_key(ds, arr) for _, ds, arr in dicom_items)
    target_key = max(
        series_counts.items(),
        key=lambda item: (item[1], bool(item[0][0]), item[0][1][0] * item[0][1][1]),
    )[0]
    dicom_items = [item for item in dicom_items if _series_key(item[1], item[2]) == target_key]
    dicom_items.sort(key=lambda item: _sort_key((item[0], item[1])))

    z_positions: list[float] = []
    slices: list[np.ndarray] = []
    for _, ds, arr in dicom_items:
        slope = _safe_float(getattr(ds, "RescaleSlope", 1.0), 1.0)
        intercept = _safe_float(getattr(ds, "RescaleIntercept", 0.0), 0.0)
        arr = arr * slope + intercept
        slices.append(arr)

        if hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
            z_positions.append(_safe_float(ds.ImagePositionPatient[2], 0.0))

    volume = np.stack(slices, axis=0)

    z_spacing = None
    if len(z_positions) >= 2:
        z_positions = sorted(z_positions)
        diffs = np.diff(z_positions)
        diffs = np.abs(diffs[diffs != 0])
        if len(diffs) > 0:
            z_spacing = float(np.median(diffs))

    first_ds = dicom_items[0][1]
    origin_xyz = _parse_origin_xyz(first_ds)
    spacing_xyz = _parse_spacing_xyz(first_ds, z_spacing)

    return volume, VolumeMetadata(origin_xyz=origin_xyz, spacing_xyz=spacing_xyz, source_files=len(dicom_items))


def _pad_volume_to_window(volume: np.ndarray, window_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pad = np.maximum(window_size - np.array(volume.shape), 0)
    if np.any(pad > 0):
        pad_width = [(0, int(pad[0])), (0, int(pad[1])), (0, int(pad[2]))]
        volume = np.pad(volume, pad_width, mode="edge")
    return volume, pad


def _compute_iou_3d(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    inter_low = np.maximum(box[:3], boxes[:, :3])
    inter_high = np.minimum(box[3:], boxes[:, 3:])
    inter_size = np.maximum(inter_high - inter_low, 0)
    inter_vol = inter_size[:, 0] * inter_size[:, 1] * inter_size[:, 2]

    box_vol = np.prod(np.maximum(box[3:] - box[:3], 0))
    boxes_vol = np.prod(np.maximum(boxes[:, 3:] - boxes[:, :3], 0), axis=1)
    union = box_vol + boxes_vol - inter_vol
    return np.divide(inter_vol, union, out=np.zeros_like(inter_vol), where=union > 0)


def _nms_3d(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.2) -> list[int]:
    if len(boxes) == 0:
        return []

    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while len(order) > 0:
        idx = int(order[0])
        keep.append(idx)
        if len(order) == 1:
            break

        rest = order[1:]
        ious = _compute_iou_3d(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]

    return keep


def detect_nodules(
    model: VitDet3D,
    volume: np.ndarray,
    probability_threshold: float = 0.5,
    batch_size: int = 24,
) -> list[dict[str, Any]]:
    window_size = np.array(model.config.image_size)
    stride_size = np.maximum((window_size * 0.75).astype(int), 1)

    original_shape = np.array(volume.shape)
    work_volume, _ = _pad_volume_to_window(volume, window_size)

    offsets, windows = sliding_window_3d(work_volume, window_size, stride_size)
    windows = (windows - LUNA16_Dataset.mean) / LUNA16_Dataset.std

    tensor_windows = torch.tensor(windows, dtype=torch.float32).unsqueeze(1)

    all_boxes: list[np.ndarray] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for start in range(0, len(tensor_windows), batch_size):
            end = min(start + batch_size, len(tensor_windows))
            batch = tensor_windows[start:end].to(model.device)
            outputs = model(pixel_values=batch)

            logits = outputs.logits.squeeze(-1).detach().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            rel_boxes = outputs.bbox.detach().cpu().numpy()

            for i, prob in enumerate(probs):
                if float(prob) < probability_threshold:
                    continue
                offset = offsets[start + i]
                abs_box = rel_boxes[i] * np.tile(window_size, 2) + np.tile(offset, 2)

                # Clip to original volume so padded regions do not leak into final results.
                low = np.floor(abs_box[:3]).astype(int)
                high = np.ceil(abs_box[3:]).astype(int)
                low = np.clip(low, 0, original_shape - 1)
                high = np.clip(high, 0, original_shape - 1)

                if np.any(high <= low):
                    continue

                clipped_box = np.concatenate([low, high]).astype(np.float32)
                all_boxes.append(clipped_box)
                all_scores.append(float(prob))

    if not all_boxes:
        return []

    boxes_np = np.array(all_boxes)
    scores_np = np.array(all_scores)
    keep_indices = _nms_3d(boxes_np, scores_np, iou_threshold=0.2)

    results: list[dict[str, Any]] = []
    for idx in keep_indices:
        box = boxes_np[idx]
        low = box[:3]
        high = box[3:]
        center = (low + high) / 2.0
        results.append(
            {
                "probability": float(scores_np[idx]),
                "bbox_zyx": [int(low[0]), int(low[1]), int(low[2]), int(high[0]), int(high[1]), int(high[2])],
                "center_zyx": [float(center[0]), float(center[1]), float(center[2])],
            }
        )

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def _window_to_uint8(slice_img: np.ndarray, wc: float = -600.0, ww: float = 1500.0) -> np.ndarray:
    low = wc - ww / 2.0
    high = wc + ww / 2.0
    out = np.clip((slice_img - low) / (high - low), 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return out


def save_detection_frames(
    volume: np.ndarray,
    detections: list[dict[str, Any]],
    output_dir: Path,
    max_frames: int = 120,
) -> list[dict[str, int | str]]:
    z_slices, height, width = volume.shape
    step = max(1, int(np.ceil(z_slices / max_frames)))

    saved_frames: list[dict[str, int | str]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, z in enumerate(range(0, z_slices, step)):
        base = _window_to_uint8(volume[z])
        frame = Image.fromarray(base, mode="L").convert("RGB")
        draw = ImageDraw.Draw(frame)

        for det in detections:
            z1, y1, x1, z2, y2, x2 = det["bbox_zyx"]
            if z1 <= z <= z2:
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 30, 30), width=2)

        max_side = 512
        if max(frame.size) > max_side:
            scale = max_side / float(max(frame.size))
            frame = frame.resize((int(frame.width * scale), int(frame.height * scale)), Image.Resampling.BILINEAR)

        file_name = f"frame_{idx:04d}.png"
        frame.save(str(output_dir / file_name), format="PNG")
        saved_frames.append({"file_name": file_name, "z_index": int(z)})

    if not saved_frames:
        raise ValueError("Unable to render preview frames")

    return saved_frames


def save_detection_gif(
    volume: np.ndarray,
    detections: list[dict[str, Any]],
    output_file: Path,
    max_frames: int = 120,
) -> None:
    z_slices, height, width = volume.shape
    step = max(1, int(np.ceil(z_slices / max_frames)))

    frames: list[Image.Image] = []
    for z in range(0, z_slices, step):
        base = _window_to_uint8(volume[z])
        frame = Image.fromarray(base, mode="L").convert("RGB")
        draw = ImageDraw.Draw(frame)

        for det in detections:
            z1, y1, x1, z2, y2, x2 = det["bbox_zyx"]
            if z1 <= z <= z2:
                draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 30, 30), width=2)

        # Keep preview lightweight for browser rendering.
        max_side = 512
        if max(frame.size) > max_side:
            scale = max_side / float(max(frame.size))
            frame = frame.resize((int(frame.width * scale), int(frame.height * scale)), Image.Resampling.BILINEAR)

        frames.append(frame)

    if not frames:
        raise ValueError("鏃犳硶鐢熸垚鍙鍖栧抚")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        str(output_file),
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=False,
    )


def voxel_center_to_world_xyz(center_zyx: list[float], metadata: VolumeMetadata) -> list[float]:
    z, y, x = center_zyx
    origin_x, origin_y, origin_z = metadata.origin_xyz.tolist()
    spacing_x, spacing_y, spacing_z = metadata.spacing_xyz.tolist()
    world_x = origin_x + x * spacing_x
    world_y = origin_y + y * spacing_y
    world_z = origin_z + z * spacing_z
    return [float(world_x), float(world_y), float(world_z)]


