"""Main pipeline orchestrator - wires all modules together.

Brain-inspired processing pipeline:
  V1 Saliency → V2 Predictive Coding → Detection → Tracking
  → MT/V5 Optical Flow → Per-ID Analysis → Scene Graph
  → Working Memory → Delta Encoding → LLM Narration
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import Optional

from vlm.aggregation.delta_encoder import DeltaEncoder
from vlm.aggregation.feature_store import FeatureStore
from vlm.aggregation.scene_graph import SceneGraphBuilder
from vlm.aggregation.token_budget import TokenBudgetManager
from vlm.analysis.expression import ExpressionDetector
from vlm.analysis.optical_flow import OpticalFlowMotion
from vlm.analysis.per_id_analyzer import PerIDAnalyzer
from vlm.analysis.pose import PoseEstimator
from vlm.capture.change_detector import ChangeDetector
from vlm.capture.predictive_coder import PredictiveCoder
from vlm.capture.saliency import SaliencyDetector
from vlm.capture.screen import ScreenCapture
from vlm.common.config import get_nested, load_config
from vlm.common.datatypes import ChangeLevel, EntityFeatures, FrameDelta
from vlm.common.device import detect_device
from vlm.detection.yolo_detector import YOLODetector
from vlm.narration.llm_client import NarrationEngine
from vlm.tracking.id_authority import IDAuthority
from vlm.tracking.track_store import TrackStore
from vlm.tracking.working_memory import WorkingMemory

logger = logging.getLogger(__name__)


class Pipeline:
    """Main recognition pipeline with brain-inspired processing.

    Processing stages (mapped to visual cortex areas):
      1. V1: Saliency detection (bottom-up attention)
      2. V2: Predictive coding (change region detection)
      3. IT: Object detection (YOLOv8)
      4. Central: Tracking (ByteTrack ID authority)
      5. MT/V5: Optical flow (pixel-level motion)
      6. IT+: Per-ID analysis (pose, expression)
      7. Parietal: Scene graph (spatial relations)
      8. Hippocampus: Working memory (Re-ID, episodes)
      9. Prefrontal: Delta encoding + LLM narration
    """

    def __init__(self, config_path: Optional[str] = None):
        self._config = load_config(config_path)
        self._running = False

        # Detect device
        device_pref = get_nested(self._config, "device.prefer", "auto")
        self._device = detect_device(device_pref)
        logger.info("Device: %s (%s)", self._device.device_name, self._device.device_type.name)

        # Initialize all modules
        self._init_capture()
        self._init_detection()
        self._init_tracking()
        self._init_analysis()
        self._init_aggregation()
        self._init_narration()

        # State
        self._rapid_change_count = 0
        self._rapid_change_window_start = 0.0
        self._frames_since_narration = 0
        self._accumulated_deltas: list[FrameDelta] = []

    # ── Initialization ──

    def _init_capture(self) -> None:
        self._capture = ScreenCapture(
            monitor=get_nested(self._config, "capture.monitor", 1),
            target_fps=get_nested(self._config, "capture.target_fps", 2.0),
            max_dimension=get_nested(self._config, "capture.max_dimension", 1920),
        )
        self._change_detector = ChangeDetector(
            phash_threshold=get_nested(self._config, "change_detection.phash_threshold", 12),
            ssim_threshold_major=get_nested(self._config, "change_detection.ssim_threshold_major", 0.50),
            ssim_threshold_moderate=get_nested(self._config, "change_detection.ssim_threshold_moderate", 0.80),
            ssim_downscale=get_nested(self._config, "change_detection.ssim_downscale", 256),
            periodic_interval=get_nested(self._config, "change_detection.periodic_interval", 10),
        )
        # Brain V2: Predictive coding - detect WHERE changes occurred
        self._predictive_coder = PredictiveCoder(
            diff_threshold=get_nested(self._config, "predictive_coding.diff_threshold", 30),
            min_region_area=get_nested(self._config, "predictive_coding.min_region_area", 500),
            merge_distance=get_nested(self._config, "predictive_coding.merge_distance", 50),
        )
        # Brain V1: Saliency - detect WHAT is visually prominent
        self._saliency = SaliencyDetector(
            saliency_threshold=get_nested(self._config, "saliency.threshold", 0.4),
            min_region_area=get_nested(self._config, "saliency.min_region_area", 500),
            change_weight=get_nested(self._config, "saliency.change_weight", 0.6),
            saliency_weight=get_nested(self._config, "saliency.saliency_weight", 0.4),
        )

    def _init_detection(self) -> None:
        small_model = get_nested(self._config, "detection.small_model", "yolov8n")
        mid_model = get_nested(self._config, "detection.mid_model", "yolov8m")
        conf = get_nested(self._config, "detection.confidence_threshold", 0.35)
        nms = get_nested(self._config, "detection.nms_threshold", 0.45)
        imgsz = get_nested(self._config, "detection.input_size", 640)

        logger.info("Loading small detector: %s", small_model)
        self._small_detector = YOLODetector(
            small_model, self._device, conf, nms, imgsz, tier="small"
        )
        self._mid_detector: Optional[YOLODetector] = None
        self._mid_model_name = mid_model
        self._det_conf = conf
        self._det_nms = nms
        self._det_imgsz = imgsz

    def _init_tracking(self) -> None:
        self._id_authority = IDAuthority(
            max_age=get_nested(self._config, "tracking.max_age", 30),
            min_hits=get_nested(self._config, "tracking.min_hits", 3),
            iou_threshold=get_nested(self._config, "tracking.iou_threshold", 0.3),
            max_entities=get_nested(self._config, "tracking.max_entities", 100),
            frame_rate=int(get_nested(self._config, "capture.target_fps", 2)),
        )
        self._track_store = TrackStore()
        # Brain Hippocampus: Working memory for Re-ID + episodic memory
        self._working_memory = WorkingMemory(
            memory_duration=get_nested(self._config, "working_memory.duration", 30.0),
            max_remembered=get_nested(self._config, "working_memory.max_remembered", 50),
            reid_threshold=get_nested(self._config, "working_memory.reid_threshold", 0.6),
            max_episodes=get_nested(self._config, "working_memory.max_episodes", 100),
        )

    def _init_analysis(self) -> None:
        pose = PoseEstimator(
            model_complexity=get_nested(self._config, "analysis.pose_model_complexity", 1),
            min_detection_confidence=get_nested(self._config, "analysis.pose_min_confidence", 0.5),
        )
        expr = ExpressionDetector()
        self._analyzer = PerIDAnalyzer(
            pose_estimator=pose,
            expression_detector=expr,
            skip_iou_threshold=get_nested(self._config, "analysis.skip_if_iou_above", 0.9),
            skip_min_frames=get_nested(self._config, "analysis.skip_if_frames_below", 5),
        )
        # Brain MT/V5: Optical flow for pixel-level motion
        self._optical_flow = OpticalFlowMotion()

    def _init_aggregation(self) -> None:
        self._feature_store = FeatureStore()
        self._delta_encoder = DeltaEncoder(
            feature_store=self._feature_store,
            coordinate_precision=get_nested(self._config, "aggregation.coordinate_precision", 5),
            min_movement=get_nested(self._config, "aggregation.min_movement_threshold", 10.0),
        )
        # Brain Parietal: Scene graph for spatial relations
        self._scene_graph = SceneGraphBuilder(
            near_threshold=get_nested(self._config, "scene_graph.near_threshold", 200.0),
            overlap_iou_threshold=get_nested(self._config, "scene_graph.overlap_iou", 0.15),
            containment_threshold=get_nested(self._config, "scene_graph.containment", 0.7),
        )
        self._token_budget = TokenBudgetManager(
            max_tokens=get_nested(self._config, "aggregation.max_tokens", 4000),
        )
        self._batch_size = get_nested(self._config, "aggregation.batch_frames", 5)

    def _init_narration(self) -> None:
        self._send_screenshot = get_nested(self._config, "narration.send_screenshot", True)
        self._narration = NarrationEngine(
            model=get_nested(self._config, "narration.model", "gemini/gemini-2.0-flash"),
            fallback_model=get_nested(self._config, "narration.fallback_model", None),
            delta_encoder=self._delta_encoder,
            min_interval=get_nested(self._config, "narration.min_interval_seconds", 5.0),
            max_context_entries=get_nested(self._config, "narration.max_context_entries", 3),
            max_crops=get_nested(self._config, "narration.max_crops", 2),
            jpeg_quality=get_nested(self._config, "narration.crop_jpeg_quality", 70),
            screenshot_max_dim=get_nested(self._config, "narration.screenshot_max_dimension", 960),
            screenshot_jpeg_quality=get_nested(self._config, "narration.screenshot_jpeg_quality", 50),
        )

    # ── Main loop ──

    def run(self) -> None:
        """Run the brain-inspired pipeline loop."""
        self._running = True
        signal.signal(signal.SIGINT, self._handle_signal)

        logger.info("Pipeline started (brain-inspired mode). Press Ctrl+C to stop.")

        for frame in self._capture.stream():
            if not self._running:
                break

            t0 = time.perf_counter()

            # ── Stage 1: V1+V2 - Change detection + Predictive coding ──
            change_level = self._change_detector.evaluate(frame)

            if change_level == ChangeLevel.NONE:
                continue

            # Rapid scene cut detection
            if change_level == ChangeLevel.MAJOR:
                if self._detect_rapid_changes():
                    self._handle_scene_cut()
                    continue

            # V2: Compute change regions (WHERE changed)
            change_regions = self._predictive_coder.compute_change_regions(frame.image)

            # V1: Combine with saliency (WHAT is prominent)
            scored_regions = self._saliency.combine_with_changes(
                frame.image, change_regions
            )

            # Log attention info
            if scored_regions:
                top = scored_regions[0]
                logger.debug(
                    "Top attention: (%d,%d,%d,%d) sal=%.2f chg=%.2f comb=%.2f",
                    top.x, top.y, top.w, top.h,
                    top.saliency_score, top.change_score, top.combined_score,
                )

            # ── Stage 2: IT - Object detection ──
            if change_level == ChangeLevel.MAJOR:
                self._lazy_load_mid_detector()
                detector = self._mid_detector or self._small_detector
            else:
                detector = self._small_detector
            detections = detector.detect(frame)

            # ── Stage 3: Central tracking (ID Authority) ──
            tracking_state = self._id_authority.update(frame, detections)

            # ── Stage 4: MT/V5 - Optical flow (full-frame) ──
            self._optical_flow.update_frame(frame.image)

            # ── Stage 5: Hippocampus - Working memory (Re-ID + episodes) ──
            # Process lost entities
            for eid in tracking_state.lost_ids:
                entity = tracking_state.entities.get(eid)
                if entity:
                    self._working_memory.on_entity_lost(entity, frame.metadata.frame_id)

            # Process new entities (attempt Re-ID)
            reid_notes: dict[int, str] = {}
            for eid in tracking_state.new_ids:
                entity = tracking_state.entities.get(eid)
                if entity:
                    match = self._working_memory.on_entity_new(
                        entity, frame.metadata.frame_id
                    )
                    if match:
                        reid_notes[eid] = (
                            f"E{eid} is likely former E{match.old_track_id} "
                            f"(sim={match.similarity})"
                        )

            # ── Stage 6: IT+ - Per-ID analysis (pose, expression, motion) ──
            features: dict[int, EntityFeatures] = {}
            for eid, entity in tracking_state.entities.items():
                if not entity.is_active:
                    continue
                prev = self._feature_store.get_latest(eid)
                feat = self._analyzer.analyze(entity, frame.metadata.frame_id, prev)

                # Override motion with optical flow data if available
                if self._optical_flow.has_flow:
                    flow_motion = self._optical_flow.compute_entity_motion(
                        eid, entity.bbox
                    )
                    feat.motion = flow_motion

                features[eid] = feat
                self._feature_store.store(feat)
                self._track_store.store(entity)

            # ── Stage 7: Parietal - Scene graph (spatial relations) ──
            rel_added, rel_removed = self._scene_graph.build_delta(
                tracking_state.entities
            )

            # ── Stage 8: Prefrontal - Delta encoding ──
            delta = self._delta_encoder.encode(
                tracking_state, features, change_level
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── Output ──
            active = len([e for e in tracking_state.entities.values() if e.is_active])
            logger.info(
                "frame=%d change=%s active=%d new=%d lost=%d regions=%d rels=+%d/-%d time=%.0fms",
                frame.metadata.frame_id,
                change_level.name,
                active,
                len(tracking_state.new_ids),
                len(tracking_state.lost_ids),
                len(scored_regions),
                len(rel_added),
                len(rel_removed),
                elapsed_ms,
            )

            # Print compact output
            if delta.entity_deltas:
                compact = self._delta_encoder.to_compact_text(delta)
                print(compact)

                # Print Re-ID notes
                for note in reid_notes.values():
                    print(f"  REID: {note}")

                # Print spatial relation changes
                rel_text = self._scene_graph.to_delta_text(rel_added, rel_removed)
                if rel_text:
                    print(rel_text)

                print()

            # ── Stage 9: LLM Narration (conditional) ──
            self._accumulated_deltas.append(delta)
            self._frames_since_narration += 1

            if self._should_narrate(change_level):
                key_crops = self._select_key_crops(tracking_state)
                # Build scene graph + memory text for LLM
                all_rels = self._scene_graph.build(tracking_state.entities)
                rel_text_for_llm = self._scene_graph.to_compact_text(all_rels)
                mem_text_for_llm = self._working_memory.get_episodes_text(n=5)
                narration = self._narration.narrate(
                    list(self._accumulated_deltas), key_crops,
                    relations_text=rel_text_for_llm,
                    memory_text=mem_text_for_llm,
                    screenshot=frame.image if self._send_screenshot else None,
                )
                if narration:
                    print(f"\n{'='*60}")
                    print("[LLM Narration]")
                    print(narration)
                    print(f"{'='*60}\n")
                    self._frames_since_narration = 0
                    self._accumulated_deltas.clear()

        self._cleanup()

    # ── Helpers ──

    def _should_narrate(self, change_level: ChangeLevel) -> bool:
        has_entity_deltas = any(
            d.entity_deltas for d in self._accumulated_deltas
        )
        if change_level == ChangeLevel.MAJOR:
            return has_entity_deltas or self._send_screenshot
        if self._frames_since_narration >= self._batch_size:
            return has_entity_deltas or self._send_screenshot
        return False

    def _detect_rapid_changes(self) -> bool:
        now = time.monotonic()
        if now - self._rapid_change_window_start > 5.0:
            self._rapid_change_count = 0
            self._rapid_change_window_start = now
        self._rapid_change_count += 1
        return self._rapid_change_count >= 3

    def _handle_scene_cut(self) -> None:
        logger.warning("Scene cut detected! Resetting all tracks.")
        self._id_authority.reset()
        self._feature_store.clear()
        self._track_store.clear()
        self._working_memory.reset()
        self._scene_graph.reset()
        self._optical_flow.reset()
        self._predictive_coder.reset()
        self._narration.clear_context()
        self._rapid_change_count = 0

    def _lazy_load_mid_detector(self) -> None:
        if self._mid_detector is not None:
            return
        try:
            logger.info("Lazy-loading mid detector: %s", self._mid_model_name)
            self._mid_detector = YOLODetector(
                self._mid_model_name, self._device,
                self._det_conf, self._det_nms, self._det_imgsz, tier="mid",
            )
        except Exception as e:
            logger.warning("Failed to load mid detector, using small: %s", e)

    def _select_key_crops(self, tracking_state) -> list[tuple[int, object]]:
        crops = []
        for eid in tracking_state.new_ids:
            entity = tracking_state.entities.get(eid)
            if entity and entity.crop is not None:
                crops.append((eid, entity.crop))
        if len(crops) < 2:
            active = [
                (eid, e) for eid, e in tracking_state.entities.items()
                if e.is_active and e.crop is not None and eid not in [c[0] for c in crops]
            ]
            active.sort(key=lambda x: x[1].bbox.confidence, reverse=True)
            for eid, e in active:
                if len(crops) >= 2:
                    break
                crops.append((eid, e.crop))
        return crops[:2]

    def _handle_signal(self, signum, frame) -> None:
        logger.info("Received signal %d, stopping...", signum)
        self._running = False

    def _cleanup(self) -> None:
        logger.info("Cleaning up...")
        self._analyzer.close()
        logger.info("Pipeline stopped.")


def main():
    """Entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    pipeline = Pipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
