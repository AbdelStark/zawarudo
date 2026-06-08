"""Vendored LeWorldModel (Push-T) model definition for the `lewm` runtime backend.

This is a faithful, inference-only copy of the model classes from upstream
``galilai-group/stable-worldmodel`` (``stable_worldmodel/wm/lewm/{lewm,module}.py``, MIT-licensed)
pinned at commit ``1986aae61977434d3d55197d2779a5f300deb82f``, plus a ViT-tiny encoder builder that
reproduces ``stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14, image_size=224)``.

We vendor (rather than depend on the research repos) so the production image carries only torch +
transformers + einops — no Hydra/gymnasium/pygame/pymunk env stack (ADR-0001; LEWM_INTEGRATION_GUIDE §1).

Upstream: https://github.com/galilai-group/stable-worldmodel (MIT). The published Push-T checkpoint
is ``quentinll/lewm-pusht`` (config `_target_`: ``stable_worldmodel.wm.lewm.LeWM``); its ``weights.pt``
is a plain ``state_dict`` that loads into this module with ``strict=True``.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


# --- transformer building blocks (upstream module.py) ----------------------------------------


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning."""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        proj = self.adaLN_modulation[-1]
        assert isinstance(proj, nn.Linear)
        nn.init.zeros_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        block_class: type[nn.Module] = Block,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()
        for _ in range(depth):
            self.layers.append(block_class(hidden_dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        x = self.output_proj(x)
        return x


class Embedder(nn.Module):
    def __init__(self, input_dim: int = 10, smoothed_dim: int = 10, emb_dim: int = 10, mlp_scale: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.smoothed_dim = smoothed_dim
        self.emb_dim = emb_dim
        self.mlp_scale = mlp_scale
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn: Any = nn.LayerNorm,
        act_fn: Any = nn.GELU,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim, hidden_dim, output_dim or input_dim, depth, heads, dim_head, mlp_dim, dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]
        x = self.dropout(x)
        return self.transformer(x, c)


# --- the world model (upstream lewm.py, inference paths) --------------------------------------


class LeWM(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module, action_encoder: nn.Module,
                 projector: nn.Module | None = None, pred_proj: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: dict[str, Any]) -> dict[str, Any]:
        pixels = info["pixels"].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        return rearrange(preds, "(b t) d -> b t d", b=emb.size(0))

    def rollout(self, info: dict[str, Any], action_sequence: torch.Tensor, history_size: int | None = None) -> dict[str, Any]:
        if history_size is None:
            history_size = getattr(self.predictor, "num_frames", 3)
        assert "pixels" in info, "pixels not in info_dict"
        h = info["pixels"].size(2)
        b, s, t = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [h, t - h], dim=2)
        info["action"] = act_0

        if "emb" not in info:
            init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            init = self.encode(init)
            info["emb"] = init["emb"].detach().unsqueeze(1).expand(b, s, -1, -1)

        emb_init = rearrange(info["emb"], "b s ... -> (b s) ...")
        act_flat = rearrange(act_0, "b s ... -> (b s) ...")
        act_future_flat = rearrange(act_future, "b s ... -> (b s) ...")
        all_act_emb = self.action_encoder(torch.cat([act_flat, act_future_flat], dim=1))

        hs = history_size
        emb_list = list(emb_init.unbind(dim=1))
        for step in range(t - h + 1):
            lo = max(0, h + step - hs)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)
            act_trunc = all_act_emb[:, lo : h + step]
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)
        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=b, s=s)
        return info

    def criterion(self, info_dict: dict[str, Any]) -> torch.Tensor:
        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]
        goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)
        cost = F.mse_loss(
            pred_emb[..., -1:, :], goal_emb[..., -1:, :].detach(), reduction="none"
        ).sum(dim=tuple(range(2, pred_emb.ndim)))
        return cost

    def get_cost(self, info_dict: dict[str, Any], action_candidates: torch.Tensor) -> torch.Tensor:
        """Goal-conditioned cost for action candidates ``[B,S,T,A]`` -> costs ``[B,S]``."""
        assert "goal" in info_dict, "goal not in info_dict"
        if "goal_emb" not in info_dict:
            goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            goal["pixels"] = goal["goal"]
            for k in list(info_dict.keys()):
                if k.startswith("goal_"):
                    goal[k[len("goal_") :]] = goal.pop(k)
            goal.pop("action", None)
            goal = self.encode(goal)
            info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


# --- builders ---------------------------------------------------------------------------------


def build_vit_tiny_encoder(image_size: int = 224, patch_size: int = 14) -> nn.Module:
    """ViT-tiny (hidden 192, 12 layers, 3 heads) HF encoder — matches upstream ``vit_hf('tiny')``."""
    from transformers import ViTConfig, ViTModel  # type: ignore[import-untyped]  # lazy: lewm backend only

    config = ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=image_size,
        patch_size=patch_size,
        num_channels=3,
        qkv_bias=True,
    )
    return ViTModel(config, add_pooling_layer=False)


def build_lewm_from_config(config: Mapping[str, Any]) -> LeWM:
    """Construct a :class:`LeWM` from the published ``config.json`` (ignoring Hydra ``_target_``)."""
    enc = dict(config.get("encoder", {}))
    encoder = build_vit_tiny_encoder(int(enc.get("image_size", 224)), int(enc.get("patch_size", 14)))

    p = config["predictor"]
    predictor = Predictor(
        num_frames=p["num_frames"], input_dim=p["input_dim"], hidden_dim=p["hidden_dim"],
        output_dim=p.get("output_dim", p["input_dim"]), depth=p["depth"], heads=p["heads"],
        mlp_dim=p["mlp_dim"], dim_head=p.get("dim_head", 64), dropout=p.get("dropout", 0.0),
        emb_dropout=p.get("emb_dropout", 0.0),
    )

    a = config["action_encoder"]
    action_encoder = Embedder(
        input_dim=a["input_dim"], emb_dim=a["emb_dim"],
        smoothed_dim=a.get("smoothed_dim", a["input_dim"]), mlp_scale=a.get("mlp_scale", 4),
    )

    def _mlp(spec: Mapping[str, Any]) -> MLP:
        return MLP(
            input_dim=spec["input_dim"], hidden_dim=spec["hidden_dim"],
            output_dim=spec.get("output_dim"), norm_fn=partial(nn.BatchNorm1d),
        )

    projector = _mlp(config["projector"]) if "projector" in config else None
    pred_proj = _mlp(config["pred_proj"]) if "pred_proj" in config else None
    return LeWM(encoder, predictor, action_encoder, projector, pred_proj)
