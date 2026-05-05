"""Prediction utilities for lightweight tribe-style video event inference.

The public API intentionally mirrors the common pretrained-model workflow::

    from cortexlab.inference.predictor import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", device="auto")
    events = model.get_events_dataframe(video_path="clip.mp4")
    preds, segments = model.predict(events)

This module provides a deterministic local fallback implementation so the API
can be exercised without downloading a heavyweight model.  If a future backend
is added, it can be selected from :meth:`TribeModel.from_pretrained` without
changing callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from ._compat import dataframe, is_dataframe, to_numeric

Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass(frozen=True)
class TribeModel:
    """Small pretrained-model facade for event extraction and segmentation.

    Parameters
    ----------
    model_id:
        Identifier for the requested model.  The bundled deterministic backend
        accepts ``"facebook/tribev2"`` and stores the value for traceability.
    device:
        Device preference.  ``"auto"`` resolves to the best available runtime
        by checking PyTorch, when installed, and otherwise falls back to CPU.
    event_stride_seconds:
        Spacing between generated video events.
    min_segment_gap_seconds:
        Maximum gap between adjacent events that may belong to the same
        contiguous segment.
    """

    model_id: str
    device: str = "cpu"
    event_stride_seconds: float = 1.0
    min_segment_gap_seconds: float = 1.5

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: Device | str = "auto",
        event_stride_seconds: float = 1.0,
        min_segment_gap_seconds: float = 1.5,
    ) -> "TribeModel":
        """Create a model facade from a pretrained identifier.

        The method is deliberately side-effect free: it does not download model
        weights during import or construction.  Unsupported identifiers still
        work with the deterministic fallback, which makes tests and local
        development reproducible.
        """

        if event_stride_seconds <= 0:
            raise ValueError("event_stride_seconds must be greater than zero")
        if min_segment_gap_seconds < 0:
            raise ValueError("min_segment_gap_seconds cannot be negative")

        resolved_device = _resolve_device(device)
        return cls(
            model_id=model_id,
            device=resolved_device,
            event_stride_seconds=float(event_stride_seconds),
            min_segment_gap_seconds=float(min_segment_gap_seconds),
        )

    def get_events_dataframe(self, video_path: str | Path) -> object:
        """Extract a timestamped event dataframe from a video path.

        When OpenCV is available and the video can be opened, one event is
        emitted every ``event_stride_seconds`` until the clip duration is
        covered.  If OpenCV is unavailable, a clear dependency error is raised.
        """

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "Reading videos requires OpenCV. Install with "
                "`pip install cortexlab[video]` or pass an events dataframe "
                "directly to predict()."
            ) from exc

        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open video file: {path}")

            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0 or frame_count <= 0:
                raise ValueError(f"Could not determine video duration: {path}")

            duration = frame_count / fps
            timestamps = _timestamp_range(duration, self.event_stride_seconds)
            return dataframe(
                {
                    "event_id": range(len(timestamps)),
                    "timestamp": timestamps,
                    "source": str(path),
                    "model_id": self.model_id,
                }
            )
        finally:
            capture.release()

    def predict(
        self,
        events: object | Iterable[dict[str, object]],
    ) -> tuple[object, object]:
        """Predict event labels and contiguous segments.

        Parameters
        ----------
        events:
            A dataframe or iterable of records containing a numeric
            ``timestamp`` column/key.

        Returns
        -------
        tuple[object, object]
            ``preds`` contains one row per event with deterministic labels and
            confidence scores.  ``segments`` merges adjacent events with the
            same predicted label.  When pandas is installed these are pandas
            dataframes; otherwise they are lightweight dataframe-like objects.
        """

        event_df = _coerce_events(events)
        if event_df.empty:
            return _empty_predictions(), _empty_segments()

        preds = event_df.copy().sort_values("timestamp").reset_index(drop=True)
        preds["prediction"] = [_label_for_timestamp(ts) for ts in preds["timestamp"]]
        preds["confidence"] = [
            _confidence_for_timestamp(ts) for ts in preds["timestamp"]
        ]
        preds["device"] = self.device
        preds["model_id"] = self.model_id

        segments = _build_segments(preds, self.min_segment_gap_seconds)
        return preds, segments


def _resolve_device(device: Device | str) -> str:
    if device != "auto":
        return str(device)

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _timestamp_range(duration: float, stride: float) -> list[float]:
    timestamps: list[float] = []
    current = 0.0
    while current <= duration:
        timestamps.append(round(current, 6))
        current += stride
    if not timestamps or timestamps[-1] < duration:
        timestamps.append(round(duration, 6))
    return timestamps


def _coerce_events(events: object | Iterable[dict[str, object]]) -> object:
    if is_dataframe(events):
        event_df = events.copy()
    else:
        event_df = dataframe(list(events))

    if event_df.empty:
        return dataframe(columns=["timestamp"])
    if "timestamp" not in event_df.columns:
        raise ValueError("events must include a numeric 'timestamp' column")

    event_df["timestamp"] = to_numeric(event_df["timestamp"])
    return event_df


def _label_for_timestamp(timestamp: float) -> str:
    bucket = int(float(timestamp) // 5) % 3
    return ("gathering", "transition", "dispersal")[bucket]


def _confidence_for_timestamp(timestamp: float) -> float:
    offset = abs((float(timestamp) % 5.0) - 2.5) / 2.5
    return round(0.62 + (0.31 * offset), 4)


def _build_segments(preds: object, max_gap: float) -> object:
    rows: list[dict[str, object]] = []
    active_label: str | None = None
    start = end = None

    for row in preds.itertuples(index=False):
        timestamp = float(row.timestamp)
        label = str(row.prediction)
        should_start = (
            active_label is None
            or label != active_label
            or (end is not None and timestamp - end > max_gap)
        )
        if should_start:
            if active_label is not None and start is not None and end is not None:
                rows.append(_segment_row(active_label, start, end))
            active_label = label
            start = timestamp
        end = timestamp

    if active_label is not None and start is not None and end is not None:
        rows.append(_segment_row(active_label, start, end))

    return dataframe(rows, columns=["label", "start", "end", "duration"])


def _segment_row(label: str, start: float, end: float) -> dict[str, object]:
    return {
        "label": label,
        "start": round(start, 6),
        "end": round(end, 6),
        "duration": round(max(0.0, end - start), 6),
    }


def _empty_predictions() -> object:
    return dataframe(
        columns=["timestamp", "prediction", "confidence", "device", "model_id"]
    )


def _empty_segments() -> object:
    return dataframe(columns=["label", "start", "end", "duration"])
