from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from sledge.autoencoder.modeling.models.rvae.rvae_config import RVAEConfig
from sledge.autoencoder.preprocessing.feature_builders.sledge.sledge_feature_processing import (
    sledge_raw_feature_processing,
)
from sledge.autoencoder.preprocessing.features.map_id_feature import MAP_ID_TO_NAME
from sledge.autoencoder.preprocessing.features.sledge_raster_feature import SledgeRaster
from sledge.autoencoder.preprocessing.features.sledge_vector_feature import (
    SledgeConfig,
    SledgeVector,
    SledgeVectorElement,
    SledgeVectorRaw,
)
from sledge.common.visualization.sledge_visualization_utils import (
    get_sledge_raster,
    get_sledge_vector_as_raster,
)
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.semantic_control import (
    NaturalLanguagePromptParser,
    PromptAlignmentEvaluator,
    SemanticSceneEditor,
)
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
    save_raw_scene,
)
from sledge.semantic_control.prompt_spec import PromptSpec, SceneEditROI


DEFAULT_ALIGNMENT_THRESHOLD = 0.70


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Semantic edit first, then low-noise masked diffusion repair. "
            "The script preserves edited semantics while producing multiple repaired variants."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input", help="Path to one source sledge_raw.gz")
    source_group.add_argument(
        "--input-dir",
        help="Cache root; recursively process all matching sledge_raw.gz files",
    )

    parser.add_argument("--output", required=True, help="Output directory for reports / debug artifacts")
    parser.add_argument("--prompt", required=True, help="Natural-language control prompt")
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--autoencoder-checkpoint", required=True, help="RVAE checkpoint path")
    parser.add_argument("--diffusion-checkpoint", required=True, help="DiT / diffusion pipeline checkpoint path")
    parser.add_argument(
        "--scenario-cache-root",
        default=None,
        help="Where to write final sledge_vector.gz variants. Defaults to $SLEDGE_EXP_ROOT/caches/scenario_cache_semantic_half_denoise",
    )
    parser.add_argument("--map-id", type=int, default=None, help="Optional override for city label")

    parser.add_argument("--num-inference-timesteps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--low-noise-start-step-seq", default="6,8,10", help="Comma-separated start-step candidates")
    parser.add_argument("--repair-attempts", type=int, default=6, help="Maximum stochastic repair attempts per scene")
    parser.add_argument("--variants-per-scene", type=int, default=3, help="How many accepted repaired variants to keep")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--alignment-threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD)
    parser.add_argument("--min-preservation-ratio", type=float, default=0.95)
    parser.add_argument("--min-latent-l1-distance", type=float, default=0.0025)

    parser.add_argument("--diff-threshold", type=float, default=1e-4)
    parser.add_argument("--diff-mask-dilation", type=int, default=2)
    parser.add_argument("--roi-mask-dilation", type=int, default=1)

    parser.add_argument("--pedestrian-roi-strength", type=float, default=1.00)
    parser.add_argument("--roadside-anchor-strength", type=float, default=1.00)
    parser.add_argument("--lane-anchor-strength", type=float, default=0.95)
    parser.add_argument("--crossing-corridor-strength", type=float, default=0.80)
    parser.add_argument("--generic-roi-strength", type=float, default=0.90)

    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--output-layout", choices=["mirror", "flat"], default="mirror")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--save-visuals", action="store_true")
    return parser


def resolve_map_id(scene_path: Path, override_map_id: Optional[int], parsed_map_id: Optional[int]) -> int:
    if override_map_id is not None:
        return int(override_map_id)
    if parsed_map_id is not None:
        return int(parsed_map_id)
    path_str = str(scene_path).lower()
    for map_id, map_name in MAP_ID_TO_NAME.items():
        if map_name in path_str:
            return int(map_id)
    return 3


def save_image(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def encode_raster(autoencoder_model, raster: SledgeRaster, device: str) -> torch.Tensor:
    raster_tensor = raster.to_feature_tensor().data.unsqueeze(0).to(device)
    encoder = autoencoder_model.get_encoder().to(device)
    encoder.eval()
    with torch.no_grad():
        latent_dist = encoder(raster_tensor)
    return latent_dist.mu


def build_raster_diff_mask(
    original_raster: SledgeRaster,
    edited_raster: SledgeRaster,
    latent_shape: torch.Size,
    device: str,
    diff_threshold: float,
    dilation: int,
) -> torch.Tensor:
    original = original_raster.to_feature_tensor().data.float().unsqueeze(0).to(device)
    edited = edited_raster.to_feature_tensor().data.float().unsqueeze(0).to(device)
    diff = (edited - original).abs().sum(dim=1, keepdim=True)
    mask = (diff > diff_threshold).float()
    if dilation > 0:
        kernel = 1 + 2 * dilation
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    mask = F.interpolate(mask, size=(latent_shape[2], latent_shape[3]), mode="nearest")
    return mask.clamp(0.0, 1.0)


def _roi_strength(tag: str, args: argparse.Namespace) -> float:
    tag = (tag or "").lower()
    if tag == "pedestrian":
        return float(args.pedestrian_roi_strength)
    if tag == "roadside_spawn_anchor":
        return float(args.roadside_anchor_strength)
    if tag == "lane_edge_conflict_anchor":
        return float(args.lane_anchor_strength)
    if tag == "crossing_corridor":
        return float(args.crossing_corridor_strength)
    return float(args.generic_roi_strength)


def build_roi_soft_mask(
    rois: List[SceneEditROI],
    config: SledgeConfig,
    latent_shape: torch.Size,
    device: str,
    dilation: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    _, _, latent_h, latent_w = latent_shape
    pixel_width, pixel_height = config.pixel_frame
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)

    for roi in rois:
        strength = _roi_strength(getattr(roi, "tag", ""), args)
        x_min = int(np.floor((roi.x_min + config.frame[0] / 2.0) / config.pixel_size))
        x_max = int(np.ceil((roi.x_max + config.frame[0] / 2.0) / config.pixel_size))
        y_min = int(np.floor((roi.y_min + config.frame[1] / 2.0) / config.pixel_size))
        y_max = int(np.ceil((roi.y_max + config.frame[1] / 2.0) / config.pixel_size))
        x_min, x_max = max(0, x_min), min(pixel_width, x_max)
        y_min, y_max = max(0, y_min), min(pixel_height, y_max)
        if x_min >= x_max or y_min >= y_max:
            continue
        raster_mask[x_min:x_max, y_min:y_max] = np.maximum(raster_mask[x_min:x_max, y_min:y_max], strength)

    mask = torch.from_numpy(raster_mask).view(1, 1, pixel_width, pixel_height).to(device)
    mask = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")
    if dilation > 0:
        kernel = 1 + 2 * dilation
        mask = F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=dilation)
    return mask.clamp(0.0, 1.0)


def make_simulation_compatible_vector(processed_vector: SledgeVector, edited_raw: SledgeVectorRaw) -> SledgeVector:
    raw_ego_states = np.asarray(edited_raw.ego.states)
    raw_ego_mask = np.asarray(edited_raw.ego.mask)
    ego_speed = float(raw_ego_states.reshape(-1)[0]) if raw_ego_states.size > 0 else 0.0
    ego_valid = bool(raw_ego_mask.reshape(-1)[0]) if raw_ego_mask.size > 0 else True
    sim_ego = SledgeVectorElement(
        states=np.asarray([ego_speed], dtype=np.float32),
        mask=np.asarray([ego_valid], dtype=np.float32),
    )
    return SledgeVector(
        lines=processed_vector.lines,
        vehicles=processed_vector.vehicles,
        pedestrians=processed_vector.pedestrians,
        static_objects=processed_vector.static_objects,
        green_lights=processed_vector.green_lights,
        red_lights=processed_vector.red_lights,
        ego=sim_ego,
    )


_SEVERITY_ACCEPTANCE = {
    "mild": {
        "pedestrian_presence_score": 0.75,
        "roadside_emergence_score": 0.25,
        "crossing_direction_score": 0.35,
        "ego_lane_conflict_score": 0.30,
        "immediacy_score": 0.08,
        "total": 0.70,
    },
    "moderate": {
        "pedestrian_presence_score": 0.75,
        "roadside_emergence_score": 0.30,
        "crossing_direction_score": 0.40,
        "ego_lane_conflict_score": 0.35,
        "immediacy_score": 0.15,
        "total": 0.72,
    },
    "aggressive": {
        "pedestrian_presence_score": 0.80,
        "roadside_emergence_score": 0.30,
        "crossing_direction_score": 0.45,
        "ego_lane_conflict_score": 0.40,
        "immediacy_score": 0.25,
        "total": 0.75,
    },
}


def summarize_crossing_semantics(alignment: object, prompt_spec: PromptSpec, threshold: float) -> Dict[str, object]:
    details = dict(getattr(alignment, "details", {}) or {})
    notes = list(getattr(alignment, "notes", []) or [])
    total = float(getattr(alignment, "total", 0.0))
    scenario_type = str(getattr(prompt_spec, "scenario_type", "generic"))
    severity_level = str(getattr(prompt_spec, "severity_level", "moderate") or "moderate").lower()
    if severity_level not in _SEVERITY_ACCEPTANCE:
        severity_level = "moderate"
    crossing_prompt = scenario_type in {"pedestrian_crossing", "sudden_pedestrian_crossing"}

    ped = float(details.get("pedestrian_presence_score", 0.0))
    roadside = float(details.get("roadside_emergence_score", 0.0))
    direction = float(details.get("crossing_direction_score", 0.0))
    conflict = float(details.get("ego_lane_conflict_score", 0.0))
    immediacy = float(details.get("immediacy_score", 0.0))

    tier_thresholds = dict(_SEVERITY_ACCEPTANCE[severity_level])
    tier_thresholds["total"] = max(float(tier_thresholds["total"]), float(threshold))

    checks = {
        "pedestrian_presence_ok": ped >= tier_thresholds["pedestrian_presence_score"],
        "roadside_emergence_ok": roadside >= tier_thresholds["roadside_emergence_score"],
        "crossing_direction_ok": direction >= tier_thresholds["crossing_direction_score"],
        "ego_lane_conflict_ok": conflict >= tier_thresholds["ego_lane_conflict_score"],
        "immediacy_ok": immediacy >= tier_thresholds["immediacy_score"],
        "total_ok": total >= tier_thresholds["total"],
    }

    if crossing_prompt:
        semantic_pass = all(checks.values())
    else:
        semantic_pass = total >= float(threshold)

    return {
        "crossing_prompt": crossing_prompt,
        "scenario_type": scenario_type,
        "severity_level": severity_level,
        "pedestrian_presence_score": ped,
        "roadside_emergence_score": roadside,
        "crossing_direction_score": direction,
        "ego_lane_conflict_score": conflict,
        "immediacy_score": immediacy,
        "semantic_pass": bool(semantic_pass),
        "threshold": float(threshold),
        "effective_thresholds": tier_thresholds,
        "checks": checks,
        "notes": notes,
    }


class SemanticHalfDenoiseRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out_root = Path(args.output)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.cfg = OmegaConf.load(args.config)
        self.cfg.autoencoder_checkpoint = args.autoencoder_checkpoint
        self.cfg.diffusion_checkpoint = args.diffusion_checkpoint

        ae_cfg_dict = OmegaConf.to_container(self.cfg.autoencoder_model.config, resolve=True)
        if not isinstance(ae_cfg_dict, dict):
            raise TypeError(f"Expected autoencoder_model.config to resolve to dict, got {type(ae_cfg_dict)}")
        filtered = {k: v for k, v in ae_cfg_dict.items() if k in RVAEConfig.__annotations__}
        self.ae_config = RVAEConfig(**filtered)

        self.prompt_parser = NaturalLanguagePromptParser()
        self.scene_editor = SemanticSceneEditor()
        self.alignment_evaluator = PromptAlignmentEvaluator()

        self.autoencoder_model = build_autoencoder_torch_module_wrapper(self.cfg)
        if hasattr(self.autoencoder_model, "eval"):
            self.autoencoder_model.eval()

        self.pipeline = build_pipeline_from_checkpoint(self.cfg)
        self.pipeline.to(args.device)
        if hasattr(self.pipeline, "transformer") and self.pipeline.transformer is not None:
            self.pipeline.transformer.eval()
        if hasattr(self.pipeline, "unet") and self.pipeline.unet is not None:
            self.pipeline.unet.eval()

        self.scenario_cache_root = self._resolve_scenario_cache_root(args.scenario_cache_root)
        self.scenario_cache_root.mkdir(parents=True, exist_ok=True)

        raw_seq = [s.strip() for s in str(args.low_noise_start_step_seq).split(",") if s.strip()]
        self.start_step_candidates = sorted(list({max(1, int(v)) for v in raw_seq})) if raw_seq else [6]
        self.num_classes = int(self.cfg.get("num_classes", 5))

    def _resolve_scenario_cache_root(self, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        sledge_exp_root = os.environ.get("SLEDGE_EXP_ROOT")
        if not sledge_exp_root:
            raise EnvironmentError(
                "SLEDGE_EXP_ROOT is not set. Export it or pass --scenario-cache-root explicitly."
            )
        return Path(sledge_exp_root) / "caches" / "scenario_cache_semantic_half_denoise"

    def _scene_output_dir(self, scene_path: Path, index: int) -> Path:
        if self.args.input:
            return self.out_root
        assert self.args.input_dir is not None
        root = Path(self.args.input_dir)
        if self.args.output_layout == "flat":
            stem = scene_path.parent.name
            return self.out_root / f"{index:06d}_{stem}"
        rel = scene_path.parent.relative_to(root)
        return self.out_root / rel

    def _scenario_cache_dir(self, scene_path: Path, index: int) -> Path:
        if self.args.input_dir:
            rel = scene_path.parent.relative_to(Path(self.args.input_dir))
            return self.scenario_cache_root / rel
        parts = scene_path.parent.parts
        rel = Path(*parts[-3:]) if len(parts) >= 3 else Path(scene_path.parent.name)
        if self.args.output_layout == "flat":
            rel = Path(f"{index:06d}_{scene_path.parent.name}")
        return self.scenario_cache_root / rel

    def _latent_distance_ok(self, final_latents: torch.Tensor, existing_latents: List[torch.Tensor]) -> bool:
        if not existing_latents:
            return True
        current = final_latents.detach().float().cpu()
        for prev in existing_latents:
            dist = torch.mean(torch.abs(current - prev.detach().float().cpu())).item()
            if dist < float(self.args.min_latent_l1_distance):
                return False
        return True

    def _save_vector_variant(self, base_dir: Path, name: str, vector: SledgeVector) -> Optional[Path]:
        base_dir.mkdir(parents=True, exist_ok=True)
        payload = feature_to_raw_scene_dict(vector)
        return save_gz_pickle(base_dir / name / "sledge_vector", payload)

    def _attempt_repair(
        self,
        init_latents: torch.Tensor,
        preserve_mask: torch.Tensor,
        map_id: int,
        attempt_idx: int,
        scene_index: int,
    ) -> Tuple[SledgeVector, torch.Tensor, int]:
        start_idx = self.start_step_candidates[attempt_idx % len(self.start_step_candidates)]
        gen = torch.Generator(device=self.args.device)
        gen.manual_seed(int(self.args.seed) + scene_index * 1000 + attempt_idx)

        with torch.no_grad():
            denoised_vectors, final_latents = self.pipeline(
                class_labels=[map_id],
                num_inference_timesteps=self.args.num_inference_timesteps,
                guidance_scale=self.args.guidance_scale,
                num_classes=self.num_classes,
                init_latents=init_latents,
                start_timestep_index=start_idx,
                preserve_mask=preserve_mask,
                generator=gen,
                return_latents=True,
            )
        vector = denoised_vectors[0].torch_to_numpy(apply_sigmoid=True)
        return vector, final_latents, start_idx

    def _maybe_save_visuals(
        self,
        out_dir: Path,
        original_vector: SledgeVector,
        edited_vector: SledgeVector,
        original_raster: SledgeRaster,
        edited_raster: SledgeRaster,
        candidate_vectors: List[Tuple[str, SledgeVector]],
    ) -> None:
        if not self.args.save_visuals:
            return
        try:
            original_raster_vis = SledgeRaster(original_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            edited_raster_vis = SledgeRaster(edited_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            save_image(out_dir / "original_raster.png", get_sledge_raster(original_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "edited_raster.png", get_sledge_raster(edited_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "original_vector.png", get_sledge_vector_as_raster(original_vector, self.ae_config))
            save_image(out_dir / "edited_vector.png", get_sledge_vector_as_raster(edited_vector, self.ae_config))
            for name, vector in candidate_vectors:
                save_image(out_dir / f"{name}.png", get_sledge_vector_as_raster(vector, self.ae_config))
        except Exception as exc:
            warning = {
                "warning_type": type(exc).__name__,
                "warning": repr(exc),
                "traceback": traceback.format_exc(),
            }
            save_json(out_dir / "visualization_warning.json", warning)

    def run_one(self, scene_path: Path, out_dir: Path, index: int = 1) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)

        raw_scene, source_format = load_raw_scene(scene_path)
        prompt_spec = self.prompt_parser.parse(self.args.prompt)
        map_id = resolve_map_id(scene_path, self.args.map_id, prompt_spec.map_id)

        original_vector, original_raster = sledge_raw_feature_processing(raw_scene, self.ae_config)

        edited_raw, edit_result = self.scene_editor.edit(raw_scene, prompt_spec)
        edited_vector, edited_raster = sledge_raw_feature_processing(edited_raw, self.ae_config)
        edited_alignment = self.alignment_evaluator.evaluate(edited_vector, prompt_spec)
        edited_semantic = summarize_crossing_semantics(edited_alignment, prompt_spec, self.args.alignment_threshold)

        init_latents = encode_raster(self.autoencoder_model, edited_raster, self.args.device)
        diff_mask = build_raster_diff_mask(
            original_raster=original_raster,
            edited_raster=edited_raster,
            latent_shape=init_latents.shape,
            device=self.args.device,
            diff_threshold=self.args.diff_threshold,
            dilation=self.args.diff_mask_dilation,
        )
        roi_mask = build_roi_soft_mask(
            rois=edit_result.preserved_rois,
            config=self.ae_config,
            latent_shape=init_latents.shape,
            device=self.args.device,
            dilation=self.args.roi_mask_dilation,
            args=self.args,
        )
        preserve_mask = torch.maximum(diff_mask, roi_mask).clamp(0.0, 1.0)

        candidate_rows: List[Dict[str, object]] = []
        accepted_variants: List[Dict[str, object]] = []
        accepted_latents: List[torch.Tensor] = []

        max_attempts = max(1, int(self.args.repair_attempts))
        target_variants = max(1, int(self.args.variants_per_scene))

        for attempt_idx in range(max_attempts):
            repaired_vector, final_latents, used_start_step = self._attempt_repair(
                init_latents=init_latents,
                preserve_mask=preserve_mask,
                map_id=map_id,
                attempt_idx=attempt_idx,
                scene_index=index,
            )
            repaired_alignment = self.alignment_evaluator.evaluate(repaired_vector, prompt_spec)
            repaired_semantic = summarize_crossing_semantics(
                repaired_alignment, prompt_spec, self.args.alignment_threshold
            )

            edited_total = max(float(edited_alignment.total), 1e-6)
            preservation_ratio = float(repaired_alignment.total) / edited_total
            semantic_preserved = bool(
                repaired_semantic["semantic_pass"] and preservation_ratio >= float(self.args.min_preservation_ratio)
            )
            latent_unique = self._latent_distance_ok(final_latents, accepted_latents)

            row = {
                "attempt_idx": int(attempt_idx),
                "start_timestep_index": int(used_start_step),
                "alignment_total": float(repaired_alignment.total),
                "semantic_summary": repaired_semantic,
                "semantic_preserved": bool(semantic_preserved),
                "preservation_ratio": float(preservation_ratio),
                "latent_unique": bool(latent_unique),
            }
            candidate_rows.append(row)

            if semantic_preserved and latent_unique:
                accepted_latents.append(final_latents.detach().cpu())
                accepted_variants.append(
                    {
                        "name": f"variant_{len(accepted_variants):03d}",
                        "vector": repaired_vector,
                        "latents": final_latents,
                        "alignment": repaired_alignment,
                        "semantic": repaired_semantic,
                        "preservation_ratio": preservation_ratio,
                        "start_timestep_index": used_start_step,
                    }
                )
                if len(accepted_variants) >= target_variants:
                    break

        edited_baseline_vector = make_simulation_compatible_vector(edited_vector, edited_raw)

        if not accepted_variants:
            accepted_variants.append(
                {
                    "name": "variant_000_fallback_edited",
                    "vector": edited_baseline_vector,
                    "latents": init_latents.detach().cpu(),
                    "alignment": edited_alignment,
                    "semantic": edited_semantic,
                    "preservation_ratio": 1.0,
                    "start_timestep_index": None,
                }
            )

        save_raw_scene(out_dir / "edited_sledge_raw", edited_raw, source_format=source_format)
        save_json(out_dir / "prompt_spec.json", prompt_spec.to_dict())
        save_json(out_dir / "edit_report.json", edit_result.to_dict())
        save_json(
            out_dir / "edited_prompt_alignment.json",
            {
                **edited_alignment.to_dict(),
                **edited_semantic,
                "accepted": bool(edited_semantic["semantic_pass"]),
                "map_id": int(map_id),
                "map_name": MAP_ID_TO_NAME.get(map_id, "unknown"),
                "source_format": source_format,
            },
        )
        save_json(out_dir / "repair_candidates.json", candidate_rows)

        scenario_cache_dir = self._scenario_cache_dir(scene_path, index)
        saved_variant_rows: List[Dict[str, object]] = []
        for variant in accepted_variants:
            variant_name = str(variant["name"])
            variant_dir = out_dir / variant_name
            variant_dir.mkdir(parents=True, exist_ok=True)

            save_json(
                variant_dir / "generated_prompt_alignment.json",
                {
                    **variant["alignment"].to_dict(),
                    **variant["semantic"],
                    "semantic_preserved": bool(variant["semantic"]["semantic_pass"]),
                    "preservation_ratio": float(variant["preservation_ratio"]),
                    "used_start_timestep_index": variant["start_timestep_index"],
                },
            )
            scenario_vector_path = self._save_vector_variant(scenario_cache_dir, variant_name, variant["vector"])
            variant_row = {
                "name": variant_name,
                "scenario_cache_vector_path": str(scenario_vector_path) if scenario_vector_path else None,
                "alignment_total": float(variant["alignment"].total),
                "semantic_summary": variant["semantic"],
                "preservation_ratio": float(variant["preservation_ratio"]),
                "used_start_timestep_index": variant["start_timestep_index"],
            }
            saved_variant_rows.append(variant_row)

            if self.args.save_visuals:
                save_image(variant_dir / "vector.png", get_sledge_vector_as_raster(variant["vector"], self.ae_config))

            if self.args.save_latents:
                torch.save(variant["latents"], variant_dir / "final_latents.pt")

        if self.args.save_latents:
            torch.save(init_latents.detach().cpu(), out_dir / "init_latents.pt")
            torch.save(diff_mask.detach().cpu(), out_dir / "diff_mask.pt")
            torch.save(roi_mask.detach().cpu(), out_dir / "roi_mask.pt")
            torch.save(preserve_mask.detach().cpu(), out_dir / "preserve_mask.pt")

        self._maybe_save_visuals(
            out_dir=out_dir,
            original_vector=original_vector,
            edited_vector=edited_vector,
            original_raster=original_raster,
            edited_raster=edited_raster,
            candidate_vectors=[(row["name"], row["vector"]) for row in accepted_variants],
        )

        summary = {
            "scene_path": str(scene_path),
            "output_dir": str(out_dir),
            "scenario_cache_dir": str(scenario_cache_dir),
            "map_id": int(map_id),
            "map_name": MAP_ID_TO_NAME.get(map_id, "unknown"),
            "source_format": source_format,
            "prompt_type": prompt_spec.scenario_type,
            "edited_alignment_total": float(edited_alignment.total),
            "edited_semantic_pass": bool(edited_semantic["semantic_pass"]),
            "num_candidates_tested": int(len(candidate_rows)),
            "num_variants_saved": int(len(saved_variant_rows)),
            "variants": saved_variant_rows,
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    def iter_scene_paths(self) -> List[Path]:
        if self.args.input:
            return [Path(self.args.input)]
        scene_paths = sorted(Path(self.args.input_dir).glob(self.args.glob_pattern))
        if self.args.max_scenes is not None:
            scene_paths = scene_paths[: self.args.max_scenes]
        return scene_paths

    def run_batch(self) -> None:
        scene_paths = self.iter_scene_paths()
        total = len(scene_paths)
        summary_rows: List[Dict[str, object]] = []

        for index, scene_path in enumerate(scene_paths, start=1):
            out_dir = self._scene_output_dir(scene_path, index)
            marker = out_dir / "summary.json"
            if self.args.skip_existing and marker.exists():
                print(f"[{index}/{total}] skipped: {scene_path}")
                continue

            print(f"[{index}/{total}] processing: {scene_path}")
            try:
                row = self.run_one(scene_path, out_dir, index=index)
                summary_rows.append(row)
                print(
                    f"[{index}/{total}] done: {scene_path} | "
                    f"edited_alignment={row['edited_alignment_total']:.4f} | "
                    f"variants={row['num_variants_saved']}"
                )
            except Exception as exc:
                error_payload = {
                    "scene_path": str(scene_path),
                    "error_type": type(exc).__name__,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                save_json(out_dir / "error.json", error_payload)
                print(f"[{index}/{total}] failed: {scene_path}\n{repr(exc)}")

        batch_summary = {
            "total_seen": total,
            "finished": len(summary_rows),
            "scenario_cache_root": str(self.scenario_cache_root),
            "rows": summary_rows,
        }
        save_json(self.out_root / "batch_summary.json", batch_summary)
        with open(self.out_root / "batch_summary.jsonl", "w", encoding="utf-8") as fp:
            for row in summary_rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = build_argparser().parse_args()
    runner = SemanticHalfDenoiseRunner(args)
    if args.input:
        runner.run_one(Path(args.input), Path(args.output), index=1)
        return
    runner.run_batch()


if __name__ == "__main__":
    main()
