from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from pathlib import Path
from typing import Dict, List, Optional

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
)
from sledge.common.visualization.sledge_visualization_utils import (
    get_sledge_raster,
    get_sledge_vector_as_raster,
)
from sledge.script.builders.diffusion_builder import build_pipeline_from_checkpoint
from sledge.script.builders.model_builder import build_autoencoder_torch_module_wrapper
from sledge.semantic_control import NaturalLanguagePromptParser, PromptAlignmentEvaluator
from sledge.semantic_control.io import (
    feature_to_raw_scene_dict,
    load_raw_scene,
    save_gz_pickle,
    save_json,
)

DEFAULT_ALIGNMENT_THRESHOLD = 0.70


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run low-noise half-denoising from already edited tiered crossing cache. "
            "The script does not re-edit scenes. It chooses one best repaired output per scene."
        )
    )
    parser.add_argument("--original-dir", required=True, help="Root of original raw cache")
    parser.add_argument("--edited-dir", required=True, help="Root of already edited tiered_crossing_raw_cache")
    parser.add_argument("--output", required=True, help="Output directory for reports / debug artifacts")
    parser.add_argument("--config", required=True, help="Expanded OmegaConf yaml")
    parser.add_argument("--autoencoder-checkpoint", required=True, help="RVAE checkpoint path")
    parser.add_argument("--diffusion-checkpoint", required=True, help="DiT / diffusion pipeline checkpoint path")
    parser.add_argument(
        "--scenario-cache-root",
        default=None,
        help="Where to write final best sledge_vector.gz. Defaults to $SLEDGE_EXP_ROOT/caches/scenario_cache_half_denoise_best",
    )
    parser.add_argument("--map-id", type=int, default=None)
    parser.add_argument("--glob-pattern", default="**/sledge_raw.gz")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--output-layout", choices=["mirror", "flat"], default="mirror")

    parser.add_argument("--num-inference-timesteps", type=int, default=24)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--low-noise-start-step-seq", default="6,8,10")
    parser.add_argument("--repair-attempts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--alignment-threshold", type=float, default=DEFAULT_ALIGNMENT_THRESHOLD)
    parser.add_argument("--min-preservation-ratio", type=float, default=0.95)

    parser.add_argument("--diff-threshold", type=float, default=1e-4)
    parser.add_argument("--diff-mask-dilation", type=int, default=2)
    parser.add_argument("--roi-mask-dilation", type=int, default=1)

    parser.add_argument("--pedestrian-roi-strength", type=float, default=1.00)
    parser.add_argument("--roadside-anchor-strength", type=float, default=1.00)
    parser.add_argument("--lane-anchor-strength", type=float, default=0.95)
    parser.add_argument("--crossing-corridor-strength", type=float, default=0.80)
    parser.add_argument("--generic-roi-strength", type=float, default=0.90)

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
    roi_dicts: List[Dict[str, float]],
    config: SledgeConfig,
    latent_shape: torch.Size,
    device: str,
    dilation: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    _, _, latent_h, latent_w = latent_shape
    pixel_width, pixel_height = config.pixel_frame
    raster_mask = np.zeros((pixel_width, pixel_height), dtype=np.float32)

    for roi in roi_dicts:
        strength = _roi_strength(str(roi.get("tag", "")), args)
        x_min = int(np.floor((float(roi["x_min"]) + config.frame[0] / 2.0) / config.pixel_size))
        x_max = int(np.ceil((float(roi["x_max"]) + config.frame[0] / 2.0) / config.pixel_size))
        y_min = int(np.floor((float(roi["y_min"]) + config.frame[1] / 2.0) / config.pixel_size))
        y_max = int(np.ceil((float(roi["y_max"]) + config.frame[1] / 2.0) / config.pixel_size))
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


def load_json(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_simulation_compatible_vector(processed_vector: SledgeVector, edited_raw) -> SledgeVector:
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


def summarize_crossing_semantics(alignment: object, prompt_spec: object, threshold: float) -> Dict[str, object]:
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


def basic_scene_compliance(vector: SledgeVector) -> Dict[str, object]:
    issues: List[str] = []

    def _check_elem(name: str, elem) -> None:
        states = np.asarray(elem.states)
        mask = np.asarray(elem.mask)
        if np.isnan(states).any() or np.isinf(states).any():
            issues.append(f"{name}: invalid numeric values")
        if np.isnan(mask).any() or np.isinf(mask).any():
            issues.append(f"{name}: invalid mask values")

    _check_elem("lines", vector.lines)
    _check_elem("vehicles", vector.vehicles)
    _check_elem("pedestrians", vector.pedestrians)
    _check_elem("static_objects", vector.static_objects)
    _check_elem("green_lights", vector.green_lights)
    _check_elem("red_lights", vector.red_lights)
    _check_elem("ego", vector.ego)

    ped_mask = np.asarray(vector.pedestrians.mask).astype(bool)
    if not np.any(ped_mask):
        issues.append("no valid pedestrian after repair")

    ego_states = np.asarray(vector.ego.states).reshape(-1)
    if ego_states.size == 0:
        issues.append("missing ego state")

    return {
        "compliant": len(issues) == 0,
        "issues": issues,
    }


class HalfDenoiseFromTieredCacheRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.original_dir = Path(args.original_dir).resolve()
        self.edited_dir = Path(args.edited_dir).resolve()
        self.out_root = Path(args.output).resolve()
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
        self.alignment_evaluator = PromptAlignmentEvaluator()

        self.autoencoder_model = build_autoencoder_torch_module_wrapper(self.cfg)
        if hasattr(self.autoencoder_model, "eval"):
            self.autoencoder_model.eval()

        self.pipeline = build_pipeline_from_checkpoint(self.cfg)
        self.pipeline.to(args.device)
        if hasattr(self.pipeline, "transformer") and self.pipeline.transformer is not None:
            self.pipeline.transformer.eval()
        self.num_classes = int(self.cfg.get("num_classes", 5))

        raw_seq = [s.strip() for s in str(args.low_noise_start_step_seq).split(",") if s.strip()]
        self.start_step_candidates = sorted(list({max(1, int(v)) for v in raw_seq})) if raw_seq else [6]

        self.scenario_cache_root = self._resolve_scenario_cache_root(args.scenario_cache_root)
        self.scenario_cache_root.mkdir(parents=True, exist_ok=True)

        self.scene_paths = sorted(self.edited_dir.glob(args.glob_pattern))
        if args.max_scenes is not None:
            self.scene_paths = self.scene_paths[: args.max_scenes]

    def _resolve_scenario_cache_root(self, override: Optional[str]) -> Path:
        if override:
            return Path(override)
        sledge_exp_root = os.environ.get("SLEDGE_EXP_ROOT")
        if not sledge_exp_root:
            raise EnvironmentError(
                "SLEDGE_EXP_ROOT is not set. Export it or pass --scenario-cache-root explicitly."
            )
        return Path(sledge_exp_root) / "caches" / "scenario_cache_half_denoise_best"

    def _scene_output_dir(self, edited_scene_path: Path, index: int) -> Path:
        if self.args.output_layout == "flat":
            stem = edited_scene_path.parent.name
            return self.out_root / f"{index:06d}_{stem}"
        rel = edited_scene_path.parent.relative_to(self.edited_dir)
        return self.out_root / rel

    def _scenario_cache_dir(self, edited_scene_path: Path, index: int) -> Path:
        rel = edited_scene_path.parent.relative_to(self.edited_dir)
        if self.args.output_layout == "flat":
            rel = Path(f"{index:06d}_{edited_scene_path.parent.name}")
        return self.scenario_cache_root / rel

    def _attempt_repair(self, init_latents: torch.Tensor, preserve_mask: torch.Tensor, map_id: int, attempt_idx: int, scene_index: int):
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

    def _load_prompt_spec(self, edited_scene_dir: Path):
        severity_label_path = edited_scene_dir / "severity_label.json"
        if severity_label_path.exists():
            payload = load_json(severity_label_path)
            prompt = str(payload.get("prompt", "突发的行人横穿马路"))
            return self.prompt_parser.parse(prompt), prompt
        raise FileNotFoundError(f"Missing severity_label.json under {edited_scene_dir}")

    def run_one(self, edited_scene_path: Path, out_dir: Path, index: int) -> Dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        rel = edited_scene_path.relative_to(self.edited_dir)
        original_scene_path = self.original_dir / rel
        if not original_scene_path.exists():
            raise FileNotFoundError(f"Cannot find paired original scene: {original_scene_path}")

        prompt_spec, prompt = self._load_prompt_spec(edited_scene_path.parent)
        map_id = resolve_map_id(edited_scene_path, self.args.map_id, getattr(prompt_spec, "map_id", None))

        original_raw, _ = load_raw_scene(original_scene_path)
        edited_raw, source_format = load_raw_scene(edited_scene_path)
        original_vector, original_raster = sledge_raw_feature_processing(original_raw, self.ae_config)
        edited_vector, edited_raster = sledge_raw_feature_processing(edited_raw, self.ae_config)

        edited_alignment = self.alignment_evaluator.evaluate(edited_vector, prompt_spec)
        edited_semantic = summarize_crossing_semantics(edited_alignment, prompt_spec, self.args.alignment_threshold)
        edited_sim_vector = make_simulation_compatible_vector(edited_vector, edited_raw)
        edited_compliance = basic_scene_compliance(edited_sim_vector)

        init_latents = encode_raster(self.autoencoder_model, edited_raster, self.args.device)
        diff_mask = build_raster_diff_mask(
            original_raster=original_raster,
            edited_raster=edited_raster,
            latent_shape=init_latents.shape,
            device=self.args.device,
            diff_threshold=self.args.diff_threshold,
            dilation=self.args.diff_mask_dilation,
        )

        edit_report_path = edited_scene_path.parent / "edit_report.json"
        roi_dicts: List[Dict[str, float]] = []
        if edit_report_path.exists():
            edit_report = load_json(edit_report_path)
            roi_dicts = list(edit_report.get("preserved_rois", []))
        roi_mask = build_roi_soft_mask(
            roi_dicts=roi_dicts,
            config=self.ae_config,
            latent_shape=init_latents.shape,
            device=self.args.device,
            dilation=self.args.roi_mask_dilation,
            args=self.args,
        )
        preserve_mask = torch.maximum(diff_mask, roi_mask).clamp(0.0, 1.0)

        candidates: List[Dict[str, object]] = []
        edited_total = max(float(edited_alignment.total), 1e-6)

        if bool(edited_semantic["semantic_pass"]) and bool(edited_compliance["compliant"]):
            candidates.append(
                {
                    "source": "edited_baseline",
                    "vector": edited_sim_vector,
                    "alignment_total": float(edited_alignment.total),
                    "semantic": edited_semantic,
                    "preservation_ratio": 1.0,
                    "compliance": edited_compliance,
                    "used_start_timestep_index": None,
                    "final_latents": init_latents.detach().cpu(),
                    "rank_score": 100.0 + float(edited_alignment.total),
                }
            )

        for attempt_idx in range(max(1, int(self.args.repair_attempts))):
            repaired_vector, final_latents, used_start_idx = self._attempt_repair(
                init_latents=init_latents,
                preserve_mask=preserve_mask,
                map_id=map_id,
                attempt_idx=attempt_idx,
                scene_index=index,
            )
            repaired_alignment = self.alignment_evaluator.evaluate(repaired_vector, prompt_spec)
            repaired_semantic = summarize_crossing_semantics(repaired_alignment, prompt_spec, self.args.alignment_threshold)
            repaired_sim_vector = make_simulation_compatible_vector(repaired_vector, edited_raw)
            repaired_compliance = basic_scene_compliance(repaired_sim_vector)
            preservation_ratio = float(repaired_alignment.total) / edited_total
            semantic_ok = bool(repaired_semantic["semantic_pass"]) and preservation_ratio >= float(self.args.min_preservation_ratio)
            compliance_ok = bool(repaired_compliance["compliant"])
            rank_score = (
                50.0 * float(semantic_ok) +
                30.0 * float(compliance_ok) +
                10.0 * float(preservation_ratio) +
                float(repaired_alignment.total)
            )
            candidates.append(
                {
                    "source": f"repair_attempt_{attempt_idx:03d}",
                    "vector": repaired_sim_vector,
                    "alignment_total": float(repaired_alignment.total),
                    "semantic": repaired_semantic,
                    "preservation_ratio": float(preservation_ratio),
                    "compliance": repaired_compliance,
                    "used_start_timestep_index": int(used_start_idx),
                    "final_latents": final_latents.detach().cpu(),
                    "rank_score": float(rank_score),
                }
            )

        valid_candidates = [
            c for c in candidates
            if bool(c["semantic"]["semantic_pass"]) and bool(c["compliance"]["compliant"]) and float(c["preservation_ratio"]) >= float(self.args.min_preservation_ratio)
        ]
        pool = valid_candidates if valid_candidates else candidates
        best = max(pool, key=lambda c: float(c["rank_score"]))

        save_json(
            out_dir / "edited_prompt_alignment.json",
            {
                **edited_alignment.to_dict(),
                **edited_semantic,
                "prompt": prompt,
                "compliance": edited_compliance,
                "accepted": bool(edited_semantic["semantic_pass"] and edited_compliance["compliant"]),
            },
        )
        save_json(
            out_dir / "candidate_scores.json",
            [
                {
                    "source": c["source"],
                    "alignment_total": c["alignment_total"],
                    "semantic_summary": c["semantic"],
                    "preservation_ratio": c["preservation_ratio"],
                    "compliance": c["compliance"],
                    "used_start_timestep_index": c["used_start_timestep_index"],
                    "rank_score": c["rank_score"],
                }
                for c in candidates
            ],
        )

        scenario_cache_dir = self._scenario_cache_dir(edited_scene_path, index)
        scenario_cache_dir.mkdir(parents=True, exist_ok=True)
        scenario_vector_path = save_gz_pickle(
            scenario_cache_dir / "sledge_vector",
            feature_to_raw_scene_dict(best["vector"]),
        )

        save_json(
            out_dir / "final_prompt_alignment.json",
            {
                "source": best["source"],
                "alignment_total": best["alignment_total"],
                "semantic_summary": best["semantic"],
                "preservation_ratio": best["preservation_ratio"],
                "compliance": best["compliance"],
                "used_start_timestep_index": best["used_start_timestep_index"],
            },
        )

        if self.args.save_latents:
            torch.save(init_latents.detach().cpu(), out_dir / "init_latents.pt")
            torch.save(diff_mask.detach().cpu(), out_dir / "diff_mask.pt")
            torch.save(roi_mask.detach().cpu(), out_dir / "roi_mask.pt")
            torch.save(preserve_mask.detach().cpu(), out_dir / "preserve_mask.pt")
            torch.save(best["final_latents"], out_dir / "best_final_latents.pt")

        if self.args.save_visuals:
            original_raster_vis = SledgeRaster(original_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            edited_raster_vis = SledgeRaster(edited_raster.to_feature_tensor().data.unsqueeze(0).cpu())
            save_image(out_dir / "original_raster.png", get_sledge_raster(original_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "edited_raster.png", get_sledge_raster(edited_raster_vis, self.ae_config.pixel_frame))
            save_image(out_dir / "original_vector.png", get_sledge_vector_as_raster(original_vector, self.ae_config))
            save_image(out_dir / "edited_vector.png", get_sledge_vector_as_raster(edited_vector, self.ae_config))
            save_image(out_dir / "best_vector.png", get_sledge_vector_as_raster(best["vector"], self.ae_config))

        summary = {
            "scene_path": str(edited_scene_path),
            "original_scene_path": str(original_scene_path),
            "output_dir": str(out_dir),
            "scenario_cache_vector_path": str(scenario_vector_path),
            "prompt": prompt,
            "source_format": source_format,
            "edited_alignment_total": float(edited_alignment.total),
            "edited_semantic_pass": bool(edited_semantic["semantic_pass"]),
            "edited_compliant": bool(edited_compliance["compliant"]),
            "selected_source": best["source"],
            "selected_alignment_total": float(best["alignment_total"]),
            "selected_semantic_pass": bool(best["semantic"]["semantic_pass"]),
            "selected_compliant": bool(best["compliance"]["compliant"]),
            "selected_preservation_ratio": float(best["preservation_ratio"]),
            "used_start_timestep_index": best["used_start_timestep_index"],
        }
        save_json(out_dir / "summary.json", summary)
        return summary

    def run_batch(self) -> None:
        total = len(self.scene_paths)
        summary_rows: List[Dict[str, object]] = []
        for index, scene_path in enumerate(self.scene_paths, start=1):
            out_dir = self._scene_output_dir(scene_path, index)
            marker = out_dir / "summary.json"
            if self.args.skip_existing and marker.exists():
                print(f"[{index}/{total}] skipped: {scene_path}")
                continue
            print(f"[{index}/{total}] processing: {scene_path}")
            try:
                row = self.run_one(scene_path, out_dir, index)
                summary_rows.append(row)
                print(
                    f"[{index}/{total}] done: {scene_path} | selected={row['selected_source']} | "
                    f"semantic={row['selected_semantic_pass']} | compliant={row['selected_compliant']}"
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
    runner = HalfDenoiseFromTieredCacheRunner(args)
    runner.run_batch()


if __name__ == "__main__":
    main()
