from copy import deepcopy
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from src.feature_builders.pluto_feature_builder import PlutoFeatureBuilder
from src.models.pluto.layers.fourier_embedding import FourierEmbedding
from src.models.pluto.layers.transformer import TransformerEncoderLayer
from src.models.pluto.modules.agent_encoder import AgentEncoder
from src.models.pluto.modules.agent_predictor import AgentPredictor
from src.models.pluto.modules.map_encoder import MapEncoder
from src.models.pluto.modules.planning_decoder import PlanningDecoder
from src.models.pluto.modules.probabilistic_heads import (
    ProbTrajHead,
    TrajUncertaintyHead,
    gaussian_log_prob_from_eps,
)
from src.models.pluto.modules.static_objects_encoder import StaticObjectsEncoder

trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)


class ProbabilisticPlanningModel(TorchModuleWrapper):
    """Probabilistic-PLUTO policy with categorical mode + latent Gaussian noise."""

    def __init__(
        self,
        dim=128,
        state_channel=6,
        polygon_channel=6,
        history_channel=9,
        history_steps=21,
        future_steps=80,
        encoder_depth=4,
        decoder_depth=4,
        drop_path=0.2,
        dropout=0.1,
        num_heads=8,
        num_modes=6,
        latent_dim=8,
        latent_hidden_dim=256,
        use_ego_history=False,
        state_attn_encoder=True,
        state_dropout=0.75,
        use_uncertainty_head=True,
        init_from_pluto_ckpt: Optional[str] = None,
        feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),
    ) -> None:
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )
        self.dim = dim
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.radius = feature_builder.radius
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.use_uncertainty_head = use_uncertainty_head
        self.init_from_pluto_ckpt = init_from_pluto_ckpt

        self.pos_emb = FourierEmbedding(3, dim, 64)
        self.agent_encoder = AgentEncoder(
            state_channel=state_channel,
            history_channel=history_channel,
            dim=dim,
            hist_steps=history_steps,
            drop_path=drop_path,
            use_ego_history=use_ego_history,
            state_attn_encoder=state_attn_encoder,
            state_dropout=state_dropout,
        )
        self.map_encoder = MapEncoder(dim=dim, polygon_channel=polygon_channel, use_lane_boundary=True)
        self.static_objects_encoder = StaticObjectsEncoder(dim=dim)
        self.encoder_blocks = nn.ModuleList(
            TransformerEncoderLayer(dim=dim, num_heads=num_heads, drop_path=dp)
            for dp in [x.item() for x in torch.linspace(0, drop_path, encoder_depth)]
        )
        self.norm = nn.LayerNorm(dim)

        self.agent_predictor = AgentPredictor(dim=dim, future_steps=future_steps)
        self.planning_decoder = PlanningDecoder(
            num_mode=num_modes,
            decoder_depth=decoder_depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=4,
            dropout=dropout,
            cat_x=False,
            future_steps=future_steps,
        )

        self.prob_traj_head = ProbTrajHead(dim, latent_dim, latent_hidden_dim, future_steps, 4)
        self.log_std_z = nn.Parameter(torch.full((latent_dim,), math.log(0.1)))
        if use_uncertainty_head:
            self.uncertainty_head = TrajUncertaintyHead(dim, future_steps)

        self.apply(self._init_weights)

        if self.init_from_pluto_ckpt:
            self._load_from_pluto_checkpoint(self.init_from_pluto_ckpt)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_from_pluto_checkpoint(self, checkpoint: str) -> None:
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        pluto_state = {
            k.replace("model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.load_state_dict(pluto_state, strict=False)

    def _encode(self, data):
        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]
        agent_mask = data["agent"]["valid_mask"][:, :, : self.history_steps]
        polygon_center = data["map"]["polygon_center"]
        polygon_mask = data["map"]["valid_mask"]

        A = agent_pos.shape[1]
        position = torch.cat([agent_pos, polygon_center[..., :2]], dim=1)
        angle = torch.cat([agent_heading, polygon_center[..., 2]], dim=1)
        angle = (angle + math.pi) % (2 * math.pi) - math.pi
        pos = torch.cat([position, angle.unsqueeze(-1)], dim=-1)

        agent_key_padding = ~(agent_mask.any(-1))
        polygon_key_padding = ~(polygon_mask.any(-1))
        key_padding_mask = torch.cat([agent_key_padding, polygon_key_padding], dim=-1)

        x_agent = self.agent_encoder(data)
        x_polygon = self.map_encoder(data)
        x_static, static_pos, static_key_padding = self.static_objects_encoder(data)
        x = torch.cat([x_agent, x_polygon, x_static], dim=1)

        pos = torch.cat([pos, static_pos], dim=1)
        x = x + self.pos_emb(pos)
        key_padding_mask = torch.cat([key_padding_mask, static_key_padding], dim=-1)

        for blk in self.encoder_blocks:
            x = blk(x, key_padding_mask=key_padding_mask, return_attn_weights=False)
        return self.norm(x), key_padding_mask, A

    def forward(self, data: Dict[str, Dict[str, torch.Tensor]]):
        enc_emb, enc_key_padding_mask, A = self._encode(data)
        prediction = self.agent_predictor(enc_emb[:, 1:A])

        agent_pos = data["agent"]["position"][:, :, self.history_steps - 1]
        agent_heading = data["agent"]["heading"][:, :, self.history_steps - 1]

        ref_line_available = data["reference_line"]["position"].shape[1] > 0
        if not ref_line_available:
            return {
                "prediction": prediction,
                "trajectory": None,
                "probability": None,
            }

        trajectory, probability, query = self.planning_decoder(
            data,
            {"enc_emb": enc_emb, "enc_key_padding_mask": enc_key_padding_mask},
            return_query=True,
        )

        bs, R, M, _ = query.shape
        query_flat = query.reshape(bs, R * M, self.dim)
        mode_logits = probability.reshape(bs, R * M)

        out = {
            "prediction": prediction,
            "trajectory": trajectory,
            "probability": probability,
            "query_features": query_flat,
            "mode_logits": mode_logits,
        }

        if self.use_uncertainty_head:
            sigma = self.uncertainty_head(query_flat.reshape(-1, self.dim)).view(bs, R * M, self.future_steps, 2)
            out["trajectory_uncertainty"] = sigma

        if not self.training:
            # planner-compatibility outputs for simulation pipeline
            z_zero = torch.zeros(bs * R * M, self.latent_dim, device=query.device)
            candidate = self.prob_traj_head(query_flat.reshape(-1, self.dim), z_zero).view(
                bs, R, M, self.future_steps, 4
            )

            angle = torch.atan2(candidate[..., 3], candidate[..., 2])
            candidate_out = torch.cat([candidate[..., :2], angle.unsqueeze(-1)], dim=-1)
            flat_prob = probability.reshape(bs, -1)
            flat_candidate = candidate_out.reshape(bs, -1, self.future_steps, 3)

            best_idx = flat_prob.argmax(-1)
            batch_idx = torch.arange(bs, device=query.device)
            out["output_trajectory"] = flat_candidate[batch_idx, best_idx]
            out["candidate_trajectories"] = candidate_out

            output_prediction = torch.cat(
                [
                    prediction[..., :2] + agent_pos[:, 1:A, None],
                    torch.atan2(prediction[..., 3], prediction[..., 2]).unsqueeze(-1)
                    + agent_heading[:, 1:A, None, None],
                    prediction[..., 4:6],
                ],
                dim=-1,
            )
            out["output_prediction"] = output_prediction

        return out

    def sample(self, data: Dict[str, Dict[str, torch.Tensor]], deterministic: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(data)
        query = out["query_features"]
        logits = out["mode_logits"]
        probs = logits.softmax(dim=-1)

        if deterministic:
            mode_idx = probs.argmax(-1)
            eps = torch.zeros(query.shape[0], self.latent_dim, device=query.device)
        else:
            mode_idx = torch.distributions.Categorical(probs=probs).sample()
            eps = torch.randn(query.shape[0], self.latent_dim, device=query.device)

        sigma_z = self.log_std_z.exp()
        latent_z = eps * sigma_z.unsqueeze(0)
        batch_idx = torch.arange(query.shape[0], device=query.device)
        q_i = query[batch_idx, mode_idx]
        traj = self.prob_traj_head(q_i, latent_z)

        logp_disc = torch.log(probs[batch_idx, mode_idx] + 1e-8)
        logp_cont = gaussian_log_prob_from_eps(eps, self.log_std_z)

        return {
            "mode_index": mode_idx,
            "eps": eps,
            "latent_z": latent_z,
            "trajectory": traj,
            "logp": logp_disc + logp_cont,
            "logp_disc": logp_disc,
            "logp_cont": logp_cont,
            "mode_logits": logits,
        }

    def logprob(self, data: Dict[str, Dict[str, torch.Tensor]], mode_index: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        out = self.forward(data)
        logits = out["mode_logits"]
        probs = logits.softmax(dim=-1)
        batch_idx = torch.arange(probs.shape[0], device=probs.device)
        logp_disc = torch.log(probs[batch_idx, mode_index] + 1e-8)
        logp_cont = gaussian_log_prob_from_eps(eps, self.log_std_z)
        return logp_disc + logp_cont

    @classmethod
    def from_deterministic_checkpoint(cls, checkpoint: str, **kwargs):
        model = cls(**kwargs)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        stripped = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(stripped, strict=False)
        return model

    def freeze_backbone(self):
        for module in [
            self.agent_encoder,
            self.map_encoder,
            self.static_objects_encoder,
            self.encoder_blocks,
            self.norm,
            self.planning_decoder,
            self.agent_predictor,
        ]:
            for p in module.parameters():
                p.requires_grad = False

    def make_reference_policy(self):
        ref = deepcopy(self).eval()
        for p in ref.parameters():
            p.requires_grad = False
        return ref
