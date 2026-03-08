"""Central ID Authority - THE single source of truth for all entity IDs.

This module wraps ByteTrack (via supervision) and guarantees:
- IDs are monotonically increasing and never reused.
- Only this class assigns or modifies track IDs.
- All downstream modules receive pre-tracked crops with stable IDs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import supervision as sv

from vlm.common.datatypes import (
    BoundingBox,
    CapturedFrame,
    DetectionResult,
    TrackedEntity,
    TrackingState,
)

logger = logging.getLogger(__name__)


class IDAuthority:
    """Central ID management using ByteTrack.

    Args:
        max_age: Frames before a lost track is retired.
        min_hits: Detections required before a track is confirmed.
        iou_threshold: Minimum IoU for matching detections to tracks.
        max_entities: Hard limit on simultaneous active tracks.
        frame_rate: Expected capture FPS (affects ByteTrack internals).
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_entities: int = 100,
        frame_rate: int = 2,
    ):
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=max_age,
            minimum_matching_threshold=iou_threshold,
            frame_rate=frame_rate,
        )
        self._max_entities = max_entities
        self._min_hits = min_hits

        # ID mapping: supervision internal ID -> our stable ID
        self._id_map: dict[int, int] = {}
        self._next_id = 0

        # Current entity state
        self._entities: dict[int, TrackedEntity] = {}
        self._prev_active_ids: set[int] = set()

        # min_hits gate: tracks must be seen N times before being confirmed
        self._hit_counts: dict[int, int] = {}
        self._confirmed: set[int] = set()

    def update(
        self, frame: CapturedFrame, detections: DetectionResult
    ) -> TrackingState:
        """Process detections and return tracking state with stable IDs.

        This is the ONLY method that assigns or modifies track IDs.
        """
        if not detections.boxes:
            return self._handle_empty_detections(frame)

        # Convert to supervision format
        sv_detections = self._to_sv_detections(detections)

        # Run ByteTrack
        tracked = self._tracker.update_with_detections(sv_detections)

        # Map to stable IDs and build entities
        current_ids: set[int] = set()
        new_ids: list[int] = []

        if tracked.tracker_id is not None:
            for i, sv_tracker_id in enumerate(tracked.tracker_id):
                # Map supervision ID to our stable ID
                is_brand_new = sv_tracker_id not in self._id_map
                if is_brand_new:
                    self._id_map[sv_tracker_id] = self._next_id
                    self._next_id += 1

                stable_id = self._id_map[sv_tracker_id]

                # min_hits gate: count hits and only confirm after threshold
                if stable_id not in self._confirmed:
                    self._hit_counts[stable_id] = self._hit_counts.get(stable_id, 0) + 1
                    if self._hit_counts[stable_id] >= self._min_hits:
                        self._confirmed.add(stable_id)
                        new_ids.append(stable_id)
                elif is_brand_new:
                    # Already confirmed (shouldn't happen for brand new), but handle it
                    new_ids.append(stable_id)

                current_ids.add(stable_id)

                # Build bounding box
                xyxy = tracked.xyxy[i]
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                bbox = BoundingBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self._resolve_class_name(cls_id, detections),
                )

                # Crop from original frame (clamp to bounds)
                crop = self._crop_entity(frame.image, bbox)

                # Update or create entity
                prev = self._entities.get(stable_id)
                self._entities[stable_id] = TrackedEntity(
                    track_id=stable_id,
                    class_name=bbox.class_name,
                    bbox=bbox,
                    crop=crop,
                    frames_alive=(prev.frames_alive + 1) if prev else 1,
                    frames_since_seen=0,
                    is_active=True,
                )

        # Detect lost IDs (only report confirmed entities)
        lost_ids: list[int] = []
        for eid in self._prev_active_ids - current_ids:
            entity = self._entities.get(eid)
            if entity and entity.is_active:
                entity.is_active = False
                entity.frames_since_seen += 1
                entity.crop = None  # Free memory
                if eid in self._confirmed:
                    lost_ids.append(eid)
                else:
                    # Unconfirmed entity vanished — silently remove
                    self._hit_counts.pop(eid, None)
                    self._entities.pop(eid, None)

        # Detect recovered IDs (were lost, now active again; confirmed only)
        recovered_ids = [
            eid for eid in current_ids
            if eid not in new_ids and eid not in self._prev_active_ids
            and eid in self._confirmed
        ]

        self._prev_active_ids = current_ids

        # Enforce max entities limit
        if len(current_ids) > self._max_entities:
            self._prune_low_confidence(current_ids)

        state = TrackingState(
            frame_id=frame.metadata.frame_id,
            entities=dict(self._entities),
            new_ids=new_ids,
            lost_ids=lost_ids,
            recovered_ids=recovered_ids,
        )

        logger.debug(
            "frame=%d active=%d new=%d lost=%d recovered=%d",
            frame.metadata.frame_id,
            len(current_ids),
            len(new_ids),
            len(lost_ids),
            len(recovered_ids),
        )

        return state

    def reset(self) -> None:
        """Reset all tracks (e.g. on scene cut)."""
        self._tracker.reset()
        self._id_map.clear()
        self._entities.clear()
        self._prev_active_ids.clear()
        self._hit_counts.clear()
        self._confirmed.clear()
        logger.info("All tracks reset")

    def get_entity(self, track_id: int) -> Optional[TrackedEntity]:
        return self._entities.get(track_id)

    def get_active_entities(self) -> dict[int, TrackedEntity]:
        return {
            eid: e for eid, e in self._entities.items() if e.is_active
        }

    def _handle_empty_detections(self, frame: CapturedFrame) -> TrackingState:
        """Handle frame with no detections."""
        lost_ids: list[int] = []
        for eid in list(self._prev_active_ids):
            entity = self._entities.get(eid)
            if entity:
                entity.is_active = False
                entity.frames_since_seen += 1
                entity.crop = None
                if eid in self._confirmed:
                    lost_ids.append(eid)
                else:
                    # Unconfirmed — silently remove
                    self._hit_counts.pop(eid, None)
                    self._entities.pop(eid, None)

        self._prev_active_ids = set()

        return TrackingState(
            frame_id=frame.metadata.frame_id,
            entities=dict(self._entities),
            new_ids=[],
            lost_ids=lost_ids,
            recovered_ids=[],
        )

    @staticmethod
    def _to_sv_detections(detections: DetectionResult) -> sv.Detections:
        n = len(detections.boxes)
        xyxy = np.zeros((n, 4), dtype=np.float32)
        confidence = np.zeros(n, dtype=np.float32)
        class_id = np.zeros(n, dtype=int)

        for i, box in enumerate(detections.boxes):
            xyxy[i] = [box.x1, box.y1, box.x2, box.y2]
            confidence[i] = box.confidence
            class_id[i] = box.class_id

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

    @staticmethod
    def _crop_entity(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        h, w = image.shape[:2]
        x1 = max(0, int(bbox.x1))
        y1 = max(0, int(bbox.y1))
        x2 = min(w, int(bbox.x2))
        y2 = min(h, int(bbox.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _resolve_class_name(
        class_id: int, detections: DetectionResult
    ) -> str:
        for box in detections.boxes:
            if box.class_id == class_id:
                return box.class_name
        return f"class_{class_id}"

    def _prune_low_confidence(self, current_ids: set[int]) -> None:
        """Remove lowest-confidence tracks when over limit."""
        active = [
            (eid, self._entities[eid])
            for eid in current_ids
            if eid in self._entities
        ]
        active.sort(key=lambda x: x[1].bbox.confidence)
        to_remove = len(active) - self._max_entities
        for i in range(to_remove):
            eid = active[i][0]
            self._entities[eid].is_active = False
            current_ids.discard(eid)
            logger.warning("Pruned low-confidence track E%d", eid)
