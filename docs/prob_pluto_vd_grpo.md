# Probabilistic-PLUTO + Continuous VD-GRPO (Implementation Notes)

This repository now includes modular building blocks for the two-stage pipeline:

1. **Probabilistic policy heads (Stage-1 warm-up)**
   - `src/models/pluto/modules/probabilistic_heads.py`
   - Adds latent-conditioned trajectory decoding (`ProbTrajectoryHead`) and optional aleatoric XY uncertainty (`TrajUncertaintyHead`).

2. **PLUTO model integration**
   - `src/models/pluto/pluto_model.py`
   - New model flags:
     - `probabilistic_policy`
     - `latent_dim`
     - `use_uncertainty_head`
   - When enabled, model emits:
     - `policy_logits`, `policy_query`, `log_std_z`
     - `prob_trajectory_mu`
     - `prob_trajectory_sigma_xy` (optional)

3. **Stage-1 NLL warm-up support**
   - `src/models/pluto/pluto_trainer.py`
   - Optional Gaussian NLL loss for selected best mode:
     - `use_prob_nll_loss`
     - `nll_weight`

4. **Stage-2 RL utilities**
   - `src/rl/prob_pluto_policy.py`: `(mode, latent)` sampling + log-prob computation.
   - `src/rl/pluto_world_model.py`: world-model wrapper over PLUTO agent predictor with optional IDM speed patch.
   - `src/rl/reward.py`: Plan-R1-style gated multi-objective reward + ESDF shaping helper.
   - `src/rl/vd_grpo.py`: scene-wise VD-GRPO advantage and loss computation.

These modules are intentionally decoupled from the legacy training loop so you can compose:

- Stage-1 supervised warm-up (IL + optional NLL)
- Stage-2 RL fine-tuning (VD-GRPO + KL/IL regularization)

without disrupting existing PLUTO workflows.
