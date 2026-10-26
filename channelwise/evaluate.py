"""Seeded native LeWM Cube evaluation with paired-start manifests."""

import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler

from eval import img_transform


class SeededWorld(swm.World):
    def __init__(self, *args, fallback_seed, **kwargs):
        self.fallback_seed = fallback_seed
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        return super().reset(seed=self.fallback_seed if seed is None else seed, options=options)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(type(value).__name__)


@hydra.main(version_base=None, config_path="../config/eval", config_name="channelwise_cube")
def run(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output.directory)
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(cfg.policy)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    dataset = swm.data.load_dataset(cfg.eval.dataset_name, keys_to_cache=["action"])
    column = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_ids = dataset.get_col_data(column)
    steps = dataset.get_col_data("step_idx")
    episodes, inverse = np.unique(episode_ids, return_inverse=True)
    last_step = np.zeros(len(episodes), dtype=np.int64)
    np.maximum.at(last_step, inverse, steps)
    valid = np.flatnonzero(steps <= last_step[inverse] - cfg.eval.goal_offset_steps)
    # Retain the upstream exclusion of the final valid row for paired legacy starts.
    chosen = np.random.default_rng(cfg.seed).choice(len(valid) - 1,
                                                    size=cfg.eval.num_eval, replace=False)
    chosen = np.sort(valid[chosen])
    initial = dataset.get_row_data(chosen)
    model = swm.wm.utils.load_pretrained(str(checkpoint)).cuda().eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("Nonfinite checkpoint")
    actions = dataset.get_col_data("action")
    scaler = StandardScaler().fit(actions[~np.isnan(actions).any(axis=1)])
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=swm.PlanConfig(**cfg.plan_config), process={"action": scaler},
        transform={"pixels": img_transform(cfg), "goal": img_transform(cfg)},
    )
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = SeededWorld(**cfg.world, image_shape=(224, 224), fallback_seed=int(cfg.seed))
    manifest = {"config": OmegaConf.to_container(cfg, resolve=True),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "rows": chosen, "episodes": initial[column], "start_steps": initial["step_idx"],
                "selection_rule": "upstream_excludes_last_valid_row",
                "evaluation_kind": "native_dataset_replay_not_heldout_trajectory_generalization"}
    (out / "manifest.json").write_text(json.dumps(manifest, default=json_default, indent=2))
    try:
        for spec in cfg.eval.callables:
            if not hasattr(world.envs.envs[0].unwrapped, spec.method):
                raise AttributeError(spec.method)
        world.set_policy(policy)
        start = time.monotonic()
        metrics = world.evaluate(
            dataset=dataset, start_steps=initial["step_idx"].tolist(),
            episodes_idx=initial[column].tolist(), goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
            video=out if cfg.output.video else None,
        )
        metrics["elapsed_seconds"] = time.monotonic() - start
        metrics["effective_reset_seeds"] = getattr(world.envs, "seeds", None)
        (out / "metrics.json").write_text(json.dumps(metrics, default=json_default, indent=2))
        print("EVALUATION_COMPLETE", json.dumps(metrics, default=json_default), flush=True)
    finally:
        world.close()


if __name__ == "__main__":
    run()
