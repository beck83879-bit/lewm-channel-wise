"""LeWM objective plus the source channelwise matched-swap router credit."""

import torch
from torch.nn import functional as F

from jepa import JEPA as OriginalJEPA


class JEPA(OriginalJEPA):
    def __init__(self, *args, routing_credit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_credit = dict(routing_credit or {})

    def _probe(self, embeddings, actions, overrides):
        predictions = self.predictor(embeddings, actions, routing_overrides=overrides,
                                     routing_record=False)
        shape = predictions.shape
        # Fresh buffers prevent probe BatchNorm updates and autograd version changes.
        buffers = {name: buffer.detach().clone()
                   for name, buffer in self.pred_proj.named_buffers()}
        predictions = torch.func.functional_call(
            self.pred_proj, buffers, (predictions.flatten(0, 1),),
        ).reshape(shape)
        return predictions

    def prediction_loss(self, embeddings, actions, target, training_step):
        routing = getattr(self.predictor, "routing", None)
        if routing is not None:
            self.predictor.set_routing_step(training_step)
        predictions = self.predict(embeddings, actions)
        if predictions.shape != target.shape:
            raise ValueError("LeWM prediction and target shapes must match exactly")
        loss = F.mse_loss(predictions, target)
        credit = loss.new_zeros(())
        metrics = {}
        if routing is not None and self.training and torch.is_grad_enabled():
            trace = self.predictor.get_last_route_trace()
            credit, metrics = self._credit_loss(embeddings, actions, target, trace, training_step)
        return {"pred_loss": loss, "route_credit_loss": credit, "routing_metrics": metrics}

    def _credit_loss(self, embeddings, actions, target, trace, step):
        cfg = self.routing_credit
        routing = self.predictor.routing
        zero = target.new_zeros(())
        metrics = {name: zero for name in (
            "credit_probe_rate", "credit_valid_rate", "credit_swap_win_rate",
            "credit_advantage_mean", "credit_advantage_abs", "credit_original_mse",
        )}
        if (not cfg.get("enabled", False) or routing.mode != "learned"
                or step < int(cfg.get("warmup_steps", 100))
                or step % max(1, int(cfg.get("interval", 8)))):
            return zero, metrics
        devices = [embeddings.device.index] if embeddings.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            swap = routing.make_swap(trace)
            block, removed, added = swap
            original_routes = routing.make_overrides(trace)
            swapped_routes = routing.make_overrides(trace, swap=swap)
            cpu_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(embeddings.device) if devices else None
            original = self._probe(embeddings.detach(), actions.detach(), original_routes)
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, embeddings.device)
            swapped = self._probe(embeddings.detach(), actions.detach(), swapped_routes)
            # Trace order is prefix-major (T,B), while model outputs are (B,T,D).
            original_mse = (original.float() - target.detach().float()).square().mean(-1)
            swapped_mse = (swapped.float() - target.detach().float()).square().mean(-1)
            original_mse = original_mse.transpose(0, 1).reshape(-1)
            swapped_mse = swapped_mse.transpose(0, 1).reshape(-1)
            advantage = original_mse - swapped_mse
            threshold = float(cfg.get("min_margin", 1e-5))
            threshold = threshold + float(cfg.get("relative_margin", 1e-3)) * original_mse
            valid = advantage.abs() > threshold
        scores = routing.rescore(block, trace[block]["inputs"]).float()
        margin = scores.gather(1, added[:, None]).squeeze(1)
        margin = margin - scores.gather(1, removed[:, None]).squeeze(1)
        credit = (F.softplus(-advantage.sign() * margin) * valid).sum() / valid.sum().clamp_min(1)
        metrics.update({
            "credit_probe_rate": zero + 1,
            "credit_valid_rate": valid.float().mean(),
            "credit_swap_win_rate": (advantage > threshold).float().mean(),
            "credit_advantage_mean": advantage.mean(),
            "credit_advantage_abs": advantage.abs().mean(),
            "credit_original_mse": original_mse.mean(),
        })
        return credit, metrics
