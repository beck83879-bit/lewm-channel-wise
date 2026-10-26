"""Full-size synthetic CUDA check before real-data training (not task results)."""

import argparse
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir

from module import SIGReg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="learned")
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(3072)
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config/train"),
                               version_base=None):
        cfg = compose(config_name="channelwise_cube", overrides=[f"arm={args.arm}",
                      "model.action_encoder.input_dim=25"])
    model = hydra.utils.instantiate(cfg.model).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr,
                                   weight_decay=cfg.optimizer.weight_decay)
    regularizer = SIGReg(**cfg.loss.sigreg.kwargs).cuda()
    torch.cuda.reset_peak_memory_stats()
    records = []
    for step in (0, 104, 105):
        optimizer.zero_grad(set_to_none=True)
        pixels = torch.randn(args.batch, 4, 3, 224, 224, device="cuda")
        actions = torch.randn(args.batch, 4, 25, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode({"pixels": pixels, "action": actions})
            emb, act = encoded["emb"], encoded["act_emb"]
            losses = model.prediction_loss(emb[:, :3], act[:, :3], emb[:, 1:], step)
            reg = regularizer(emb.transpose(0, 1))
            loss = losses["pred_loss"] + 0.1 * reg + 0.01 * losses["route_credit_loss"]
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if step == 104 and args.arm == "learned":
            assert losses["routing_metrics"]["credit_probe_rate"] == 1
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.predictor.routing.routers.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if model.predictor.routing is not None:
            model.predictor.routing.after_step()
        records.append({"step": step, "loss": float(loss.detach()),
                        "prediction_loss": float(losses["pred_loss"].detach()),
                        "credit_loss": float(losses["route_credit_loss"].detach())})
        del pixels, actions, encoded, emb, act, loss, losses, reg
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.predict(torch.randn(300, 3, 192, device="cuda"),
                               torch.randn(300, 3, 192, device="cuda"))
        assert result.shape == (300, 3, 192) and torch.isfinite(result).all()
    torch.cuda.synchronize()
    print(json.dumps({"status": "passed", "arm": args.arm, "batch": args.batch,
                      "gpu": torch.cuda.get_device_name(), "steps": records,
                      "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "cem_candidate_batch": 300}, indent=2), flush=True)


if __name__ == "__main__":
    main()
