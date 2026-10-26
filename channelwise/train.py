"""Train LeWM controls and channelwise variants using the same data pipeline."""

import hashlib
import json
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining._config import get_config as get_spt_config

from module import SIGReg
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


class RoutingStepCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        self.last_step = int(trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if int(trainer.global_step) == self.last_step:
            return
        routing = getattr(pl_module.model.predictor, "routing", None)
        if routing is not None:
            metrics = routing.after_step()
            pl_module.log_dict({f"train/route_{key}": value for key, value in metrics.items()},
                               on_step=True, on_epoch=False, sync_dist=False)
        self.last_step = int(trainer.global_step)


def training_forward(self, batch, stage, cfg):
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act = output["emb"], output["act_emb"]
    if emb.size(1) != cfg.history_size + cfg.num_preds or cfg.num_preds != 1:
        raise ValueError("This experiment preserves LeWM's one-step aligned targets")
    losses = self.model.prediction_loss(
        emb[:, :cfg.history_size], act[:, :cfg.history_size], emb[:, cfg.num_preds:],
        training_step=int(self.global_step),
    )
    output.update({key: losses[key] for key in ("pred_loss", "route_credit_loss")})
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = (output["pred_loss"] + cfg.loss.sigreg.weight * output["sigreg_loss"]
                      + cfg.model.routing_credit.weight * output["route_credit_loss"])
    if not torch.isfinite(output["loss"]):
        raise FloatingPointError("Nonfinite training loss")
    self.log_dict({f"{stage}/{key}": value.detach() for key, value in output.items()
                   if "loss" in key}, on_step=True, on_epoch=True,
                  sync_dist=True, batch_size=emb.size(0))
    if losses["routing_metrics"]:
        self.log_dict({f"{stage}/{key}": value.detach()
                       for key, value in losses["routing_metrics"].items()},
                      on_step=True, on_epoch=True, sync_dist=True, batch_size=emb.size(0))
    return output


@hydra.main(version_base=None, config_path="../config/train", config_name="channelwise_cube")
def run(cfg):
    pl.seed_everything(int(cfg.seed), workers=True)
    spt.set(default_loggers={"registry": False})
    get_spt_config().cache_dir = None
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    name = dataset_cfg.pop("name")
    if not Path(name).is_file():
        raise FileNotFoundError(name)
    dataset = swm.data.load_dataset(name, transform=None, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    with open_dict(cfg):
        for column in cfg.data.dataset.keys_to_load:
            if not column.startswith("pixels"):
                transforms.append(get_column_normalizer(dataset, column, column))
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")
    dataset.transform = spt.data.transforms.Compose(*transforms)
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=generator,
    )
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, shuffle=True,
                                       drop_last=True, generator=generator)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    model = hydra.utils.instantiate(cfg.model)
    world_model = spt.Module(
        model=model, sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(training_forward, cfg=cfg),
        optim={"model_opt": {
            "modules": "model", "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch",
        }},
    )
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    root = Path(__file__).resolve().parents[1]
    source_files = [*root.glob("*.py"), *root.glob("channelwise/*.py"),
                    *root.glob("config/train/**/*.yaml")]
    indices = getattr(train_set, "indices", [])
    manifest = {
        "arm": cfg.arm_name, "seed": cfg.seed, "dataset": str(Path(name).resolve()),
        "dataset_bytes": Path(name).stat().st_size,
        "train_samples": len(train_set), "val_samples": len(val_set),
        "train_indices_sha256": hashlib.sha256(torch.as_tensor(indices).numpy().tobytes()).hexdigest(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        "torch": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("RUN_MANIFEST", json.dumps(manifest, indent=2), flush=True)
    resume = ModelCheckpoint(dirpath=str(run_dir / "training_state"), filename="epoch-{epoch:02d}",
                             save_last=True, save_top_k=1, every_n_epochs=1, save_weights_only=False)
    trainer = pl.Trainer(
        **cfg.trainer, callbacks=[SaveCkptCallback(cfg.output_model_name, cfg.model),
                                 RoutingStepCallback(), resume],
        num_sanity_val_steps=1, logger=CSVLogger(str(run_dir), name="metrics", version="",
                                               flush_logs_every_n_steps=10),
        default_root_dir=str(run_dir),
    )
    checkpoint = run_dir / "training_state/last.ckpt"
    manager = spt.Manager(
        trainer=trainer, module=world_model, data=spt.data.DataModule(train=train, val=val),
        seed=int(cfg.seed), weights_only=False,
        ckpt_path=checkpoint if checkpoint.exists() else None,
    )
    manager()
    print("TRAINING_COMPLETE", cfg.arm_name, flush=True)


if __name__ == "__main__":
    run()
