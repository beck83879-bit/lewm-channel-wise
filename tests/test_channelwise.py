import copy
import ast
import hashlib
import io
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from module import ARPredictor, ConditionalBlock, MLP
from channelwise.model import JEPA
from channelwise.predictor import ChannelwiseARPredictor, batched_active_block
from channelwise.source_ops import SourceSparseOps


def build(mode="learned", dropout=0.0):
    return ChannelwiseARPredictor(
        num_frames=3, input_dim=12, hidden_dim=28, output_dim=12,
        depth=3, heads=2, dim_head=8, mlp_dim=48, dropout=dropout,
        routing=dict(enabled=True, sparse_compute=True, mode=mode, shared_dim=12,
                     group_size=4, top_k=1, selected_weight_mode="binary",
                     router_hidden_dim=16, warmup_steps=0),
    )


def initialize_gates(predictor):
    for block in predictor.transformer.layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)


class ChannelwiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(3072)

    def test_upstream_and_source_router_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        expected = {
            "module.py": "0b258a9e8dc24c29fcb1e8c50a09ec78b8ea85aeb79e21dd8adf712396646620",
            "jepa.py": "41bad7fd21e0f14aea4c9c3d39a9c87037e787746d953ab62cdc0677e938ce96",
            "train.py": "5d5666a785148635f8bcb50b49f98dd4527c5aba8a02d90206eb5d46406079d2",
            "channelwise/channel_routing.py": "6a221fc2bab0ba86ea105c992562ef7ba9df997e1879e28ea80b58a5fbc3fc85",
            "channelwise/sparse_linear.py": "3bdb071237ca091da2b2ca0d525fcdf31aeb6c00cf705bd1fe645b854ada1359",
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), digest, name)

    def test_sparse_primitives_match_source_text(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "channelwise/source_ops.py").read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        # Source fingerprints are independent of the installed Python AST version.
        expected = {
            "_active_attention": "3399d386af735b943cc869a6df89e9ceb239f4619a3d20a9b138fd8f02d32869",
            "_batched_linear": "f74c1ec0926bc0be7a605c2afd631ec29d5e2eb5f843a6ff26a4274383309a9c",
        }
        for node in cls.body:
            if isinstance(node, ast.FunctionDef):
                digest = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
                self.assertEqual(digest, expected.pop(node.name))
        self.assertEqual(expected, {})

    def _assert_swap_dropout_masks_match(self, device):
        predictor = build(dropout=0.5).to(device).train()
        initialize_gates(predictor)
        batch, length = 4, 3
        mask = torch.zeros(batch * length, 4, device=device, dtype=torch.bool)
        groups = torch.arange(batch, device=device).repeat(length)
        mask.scatter_(1, groups[:, None], True)
        original = [{"mask": mask.clone(), "weights": mask.float()} for _ in range(3)]
        swapped = copy.deepcopy(original)
        swap_groups = torch.tensor([1, 0, 2, 3], device=device).repeat(length)
        swapped[0]["mask"].zero_().scatter_(1, swap_groups[:, None], True)
        swapped[0]["weights"] = swapped[0]["mask"].float()
        x = torch.randn(batch, length, 12, device=device)
        c = torch.randn_like(x)
        source_dropout = F.dropout

        def probe(overrides):
            masks = []

            def dropout(inputs, p=0.5, training=True, inplace=False):
                applied = source_dropout(torch.ones_like(inputs), p, training, False)
                if training and p:
                    masks.append(applied.detach().clone())
                return inputs.mul_(applied) if inplace else inputs * applied

            devices = [torch.cuda.current_device()] if device == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad(), patch.object(F, "dropout", dropout):
                torch.manual_seed(20261027)
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        output = predictor(x, c, routing_overrides=overrides)
                else:
                    output = predictor(x, c, routing_overrides=overrides)
            return output, masks

        first, first_masks = probe(original)
        second, second_masks = probe(swapped)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(first_masks), length * 3 * 4)
        self.assertEqual(len(first_masks), len(second_masks))
        for index, (a, b) in enumerate(zip(first_masks, second_masks)):
            self.assertTrue(torch.equal(a, b), f"Dropout call {index} changed sample masks")

    def test_swapped_routes_preserve_each_sample_dropout_cpu(self):
        self._assert_swap_dropout_masks_match("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_swapped_routes_preserve_each_sample_dropout_cuda(self):
        self._assert_swap_dropout_masks_match("cuda")

    def test_disabled_is_exact_upstream(self):
        args = dict(num_frames=3, input_dim=12, hidden_dim=28, output_dim=12,
                    depth=3, heads=2, dim_head=8, mlp_dim=48, dropout=0.0)
        state = torch.random.get_rng_state()
        original = ARPredictor(**args).eval()
        torch.random.set_rng_state(state)
        adapted = ChannelwiseARPredictor(**args, routing={"enabled": False}).eval()
        self.assertEqual(list(original.state_dict()), list(adapted.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, adapted.state_dict()[key]), key)
        x, c = torch.randn(4, 3, 12), torch.randn(4, 3, 12)
        self.assertTrue(torch.equal(original(x, c), adapted(x, c)))

    def test_selected_block_matches_explicit_reduced_block(self):
        full = ConditionalBlock(28, 2, 8, 48, 0.0).eval()
        nn.init.normal_(full.adaLN_modulation[-1].weight, std=0.03)
        indices = torch.cat((torch.arange(12), torch.arange(20, 24)))
        reduced = ConditionalBlock(16, 2, 8, 48, 0.0).eval()
        source = full.state_dict()
        target = {}
        for name in reduced.state_dict():
            value = source[name]
            if name == "adaLN_modulation.1.weight":
                rows = torch.cat([indices + i * 28 for i in range(6)])
                value = value.index_select(0, rows).index_select(1, indices)
            elif name == "adaLN_modulation.1.bias":
                value = value.index_select(0, torch.cat([indices + i * 28 for i in range(6)]))
            elif name in {"attn.to_qkv.weight", "mlp.net.1.weight"}:
                value = value.index_select(1, indices)
            elif name in {"attn.to_out.0.weight", "attn.to_out.0.bias", "mlp.net.4.weight",
                          "mlp.net.4.bias", "attn.norm.weight", "attn.norm.bias",
                          "mlp.net.0.weight", "mlp.net.0.bias"}:
                value = value.index_select(0, indices)
            target[name] = value
        reduced.load_state_dict(target, strict=True)
        x = torch.randn(4, 3, 16, requires_grad=True)
        c = torch.randn(4, 3, 16, requires_grad=True)
        full_x = x.new_zeros(4, 3, 28).scatter(-1, indices[None, None].expand_as(x), x)
        full_c = c.new_zeros(4, 3, 28).scatter(-1, indices[None, None].expand_as(c), c)
        actual = batched_active_block(full, full_x, full_c, torch.zeros(4, dtype=torch.long),
                                      indices[None]).index_select(-1, indices)
        expected = reduced(x, c)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual_grad = torch.autograd.grad(actual.square().mean(), (x, c), retain_graph=True)
        expected_grad = torch.autograd.grad(expected.square().mean(), (x, c))
        for a, e in zip(actual_grad, expected_grad):
            torch.testing.assert_close(a, e, rtol=1e-4, atol=1e-6)

    def test_real_widths_compute_selected_submatrices(self):
        block = ConditionalBlock(336, 16, 64, 2048, 0.0).eval()
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.03)
        indices = torch.stack([torch.cat((torch.arange(144), torch.arange(
            144 + group * 48, 192 + group * 48))) for group in range(4)])
        x, c = torch.randn(2, 3, 336), torch.randn(2, 3, 336)
        groups = torch.tensor([0, 3])
        shapes = []
        source_linear = SourceSparseOps._batched_linear

        def observe(inputs, bank, selected, *args, **kwargs):
            shapes.append((inputs.size(-1), tuple(bank[0].shape[1:])))
            return source_linear(inputs, bank, selected, *args, **kwargs)

        with patch.object(SourceSparseOps, "_batched_linear", observe), \
                patch.object(block, "forward", side_effect=AssertionError("Full block invoked")), \
                patch.object(block.attn, "forward", side_effect=AssertionError("Full attention invoked")), \
                patch.object(block.mlp, "forward", side_effect=AssertionError("Full MLP invoked")):
            actual = batched_active_block(block, x, c, groups, indices)
        self.assertEqual(shapes, [(192, (1152, 192)), (192, (3072, 192)),
                                  (1024, (192, 1024)), (192, (2048, 192)),
                                  (2048, (192, 2048))])
        active = torch.zeros_like(x, dtype=torch.bool)
        active.scatter_(-1, indices[groups][:, None, :].expand(-1, 3, -1), True)
        self.assertTrue(torch.equal(actual[~active], x[~active]))
        self.assertFalse(torch.equal(actual[active], x[active]))

    def test_routing_is_causal(self):
        predictor = build().eval()
        initialize_gates(predictor)
        x, c = torch.randn(4, 3, 12), torch.randn(4, 3, 12)
        before = predictor(x, c)
        changed_x, changed_c = x.clone(), c.clone()
        changed_x[:, 1:] += torch.randn_like(changed_x[:, 1:]) * 100
        changed_c[:, 1:] += torch.randn_like(changed_c[:, 1:]) * 100
        after = predictor(changed_x, changed_c)
        torch.testing.assert_close(before[:, :1], after[:, :1], rtol=0, atol=0)
        torch.testing.assert_close(before[:, :2], predictor(x[:, :2], c[:, :2]), rtol=0, atol=0)
        torch.testing.assert_close(before[:, :1], predictor(x[:, :1], c[:, :1]), rtol=0, atol=0)

    def test_fixed_route_matches_reduced_upstream_predictor(self):
        predictor = build().eval()
        initialize_gates(predictor)
        reference = ARPredictor(num_frames=3, input_dim=12, hidden_dim=16,
                                output_dim=12, depth=3, heads=2, dim_head=8,
                                mlp_dim=48, dropout=0.0).eval()
        indices = predictor.active_indices[2]
        source = predictor.state_dict()
        state = {}
        for name, desired in reference.state_dict().items():
            value = source[name]
            if "adaLN_modulation.1" in name:
                value = value.index_select(0, torch.cat([indices + i * 28 for i in range(6)]))
                if value.ndim == 2:
                    value = value.index_select(1, indices)
            else:
                for axis, (old, new) in enumerate(zip(value.shape, desired.shape)):
                    if old == 28 and new == 16:
                        value = value.index_select(axis, indices)
            state[name] = value
        reference.load_state_dict(state, strict=True)
        mask = torch.zeros(12, 4, dtype=torch.bool)
        mask[:, 2] = True
        overrides = [{"mask": mask, "weights": mask.float()} for _ in range(3)]
        x, c = torch.randn(4, 3, 12), torch.randn(4, 3, 12)
        with torch.no_grad():
            actual = predictor(x, c, routing_overrides=overrides)
            expected = reference(x, c)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_credit_gradients_rng_buffers_and_reload(self):
        predictor = build(dropout=0.1)
        initialize_gates(predictor)
        model = JEPA(nn.Identity(), predictor, nn.Identity(),
                     pred_proj=MLP(12, 24, 12, norm_fn=nn.BatchNorm1d),
                     routing_credit=dict(enabled=True, interval=1, warmup_steps=0,
                                         min_margin=0, relative_margin=0)).train()
        x = torch.randn(4, 3, 12, requires_grad=True)
        c = torch.randn(4, 3, 12, requires_grad=True)
        target = torch.randn(4, 3, 12, requires_grad=True)
        control = copy.deepcopy(model)
        control.routing_credit["enabled"] = False
        rng = torch.random.get_rng_state()
        control.prediction_loss(x, c, target, 0)
        expected_rng = torch.random.get_rng_state()
        expected_buffers = {n: t.clone() for n, t in control.pred_proj.named_buffers()}
        torch.random.set_rng_state(rng)
        losses = model.prediction_loss(x, c, target, 0)
        self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
        for name, value in model.pred_proj.named_buffers():
            self.assertTrue(torch.equal(value, expected_buffers[name]), name)
        self.assertEqual(float(losses["routing_metrics"]["credit_probe_rate"]), 1)
        named = list(model.named_parameters())
        grads = torch.autograd.grad(losses["route_credit_loss"],
                                    [p for _, p in named] + [x, c, target],
                                    retain_graph=True, allow_unused=True)
        router_sum = 0
        for (name, _), gradient in zip(named, grads):
            if ".routing.routers." in name:
                if gradient is not None:
                    router_sum += gradient.abs().sum().item()
            else:
                self.assertIsNone(gradient, name)
        self.assertGreater(router_sum, 0)
        self.assertTrue(all(g is None for g in grads[-3:]))
        (losses["pred_loss"] + 0.01 * losses["route_credit_loss"]).backward()
        self.assertGreater(target.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))
        routing = predictor.routing
        torch.testing.assert_close(routing.pending_samples, torch.full((3,), 12.0))
        routing.after_step()
        self.assertEqual(routing.step.item(), 1)
        self.assertEqual(routing.pending_samples.sum().item(), 0)
        model.eval()
        buffers = {name: value.clone() for name, value in routing.named_buffers()}
        with torch.no_grad():
            first = model.predict(x, c)
        for name, value in routing.named_buffers():
            self.assertTrue(torch.equal(value, buffers[name]))
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = copy.deepcopy(model)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(first, restored.predict(x, c), rtol=0, atol=0)

    def test_hard_route_has_no_main_loss_router_gradient(self):
        predictor = build()
        initialize_gates(predictor)
        output = predictor(torch.randn(4, 3, 12), torch.randn(4, 3, 12))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in predictor.routing.routers.parameters()))

    def test_probes_replay_exactly(self):
        predictor = build(dropout=0.1).train()
        initialize_gates(predictor)
        x, c = torch.randn(4, 3, 12), torch.randn(4, 3, 12)
        predictor(x, c)
        trace = predictor.get_last_route_trace()
        override = predictor.routing.make_overrides(trace)
        counts = predictor.routing.pending_counts.clone()
        rng = torch.random.get_rng_state()
        first = predictor(x, c, routing_overrides=override)
        torch.random.set_rng_state(rng)
        second = predictor(x, c, routing_overrides=override)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(counts, predictor.routing.pending_counts))
        self.assertIs(trace, predictor.get_last_route_trace())


if __name__ == "__main__":
    unittest.main()
