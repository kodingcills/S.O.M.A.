"""§0.5 read-only Simulus instrumentation.

Exposes the ten contract objects for one decision point WITHOUT altering
the live policy trajectory:
  1 M_t                → caller-held RecurrentState snapshot (see CraftaxDriver)
  2 b_{u,a}            → obs+action token-embedding block (pre-backbone)
  3 g_{u,a}            → prediction latents fed to the curiosity heads
  4 head probs         → per-member categorical probs per modality
  5 J(u,a)             → native modality-aggregated JSD
  6 main token probs   → head_observations categoricals on g_{u,a}
  7 reward head        → probs / sym_log bin support / deterministic expectation
  8 termination prob   → softmax(head_ends)[1]
  9 controller logits/probs → read-only AC probe on a cloned actor LSTM state
 10 symbolic tokens    → caller-side (SomaCraftaxEnv.observation())

J(u,a) aggregation mirrors vendored envs/world_model_env.py::_sample_obs_
and_rewards exactly: sum over modalities of estimate_uncertainty()[0]
.mean(-1). Verified against the native path by test T4.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import torch


class ControllerDistribution(NamedTuple):
    logits: torch.Tensor
    probs: torch.Tensor


@dataclass
class InstrumentedActionOutput:
    action_idx: int
    J_ua: float
    controller_logits: torch.Tensor | None = None     # (43,)
    controller_probs: torch.Tensor | None = None      # (43,)
    reward_probs: torch.Tensor | None = None          # (129,) sym_log space
    reward_bins: torch.Tensor | None = None           # (129,) support
    reward_expectation: float | None = None           # real-space value
    reward_expectation_head: float | None = None      # RewardHead.forward out
    termination_prob: float | None = None
    main_token_probs: dict = field(default_factory=dict)   # modality → (B,T,V)
    b_ua: torch.Tensor | None = None                  # (B, n+m, E)
    g_ua: torch.Tensor | None = None                  # (B, n_obs, E)
    head_mean_probs: dict = field(default_factory=dict)    # modality → mean dist
    curiosity_member_probs: dict = field(default_factory=dict)  # modality → (E,B,T,V)


def _clone_recurrent_state(rs):
    if rs is None:
        return None
    if isinstance(rs, tuple):
        return tuple(_clone_recurrent_state(x) for x in rs)
    if isinstance(rs, torch.Tensor):
        return rs.clone()
    target = copy.copy(rs)
    target.state = _clone_recurrent_state(getattr(rs, "state", None))
    target.n = getattr(rs, "n", 0)
    return target


def embed_obs_block(wm, tokenizer, model_obs, device) -> torch.Tensor:
    """Current-obs token embeddings, flattened [B, n_obs, E].

    Mirrors world_model_env's context handling: embeddings get a t=1 dim,
    then flatten(1, 2) yields the inference-path token sequence.
    """
    tokens = {
        m: v.tokens.unsqueeze(1)
        for m, v in tokenizer.encode(model_obs, should_preprocess=True).items()
    }
    emb = wm.embed_obs_tokens(tokens, tokenizer)
    return emb.flatten(1, 2)


def embed_action_flat(wm, action_idx: int, device) -> torch.Tensor:
    action_t = torch.tensor([[action_idx]], dtype=torch.long, device=device)
    return wm.embed_actions(action_t).flatten(1, 2)


def build_block(wm, obs_emb_flat: torch.Tensor, act_emb_flat: torch.Tensor):
    """[B, n_obs+n_act, E] — obs tokens first, matching the pinned layout."""
    return torch.cat([obs_emb_flat, act_emb_flat], dim=1)


def _expand_recurrent_state(rs, batch: int):
    """Expand a B=1 recurrent state to B=batch (RetNet chunkwise state)."""
    import torch as _torch

    if rs is None:
        return None
    if isinstance(rs, tuple):
        return tuple(_expand_recurrent_state(x, batch) for x in rs)
    if isinstance(rs, _torch.Tensor):
        if rs.shape[0] == batch:
            return rs
        return rs.repeat_interleave(batch, dim=0)
    target = copy.copy(rs)
    inner = getattr(rs, "state", None)

    def expand(x):
        if isinstance(x, tuple):
            return tuple(expand(t) for t in x)
        if isinstance(x, _torch.Tensor):
            return x.repeat_interleave(batch, dim=0) if x.shape[0] == 1 else x
        return x

    target.state = expand(inner)
    n = getattr(rs, "n", 0)
    tensors = inner if isinstance(inner, tuple) else ()
    device = next(
        (t.device for t in tensors if isinstance(t, _torch.Tensor)),
        _torch.device("cpu"),
    )
    target.n = _torch.full((batch,), int(n), dtype=_torch.long, device=device)
    return target


class SimulusInstrumented:
    def __init__(self, agent) -> None:
        self._agent = agent
        self._wm = agent.world_model

    # -- world-model candidate evaluation ------------------------------------

    @torch.no_grad()
    def evaluate_action(
        self,
        model_obs: dict,
        prior_context_emb: torch.Tensor,
        recurrent_state,
        action_idx: int,
    ) -> InstrumentedActionOutput:
        """Evaluate one candidate action on cloned WM state.

        prior_context_emb: (B, n_obs, E) embedding of current obs tokens.
        recurrent_state: live RetNet RecurrentState — never mutated here.
        """
        wm = self._wm
        device = next(wm.parameters()).device
        obs_emb_flat = embed_obs_block(wm, self._agent.tokenizer, model_obs, device)
        act_emb_flat = embed_action_flat(wm, action_idx, device)
        block = build_block(wm, obs_emb_flat, act_emb_flat)   # b_ua
        block4d = block.unsqueeze(1)                            # (B,1,k1,E)

        cloned = _clone_recurrent_state(recurrent_state)
        outs = wm.forward_inference(block4d, recurrent_state=cloned)
        pred_latents = wm.compute_next_obs_pred_latents(cloned)[0]  # g_ua

        sizes = [wm.tokens_per_obs_dict[m] for m in wm.ordered_modalities]
        splits = torch.split(pred_latents, sizes, dim=1)

        jsd_terms, main_probs, head_means, member_probs = [], {}, {}, {}
        for m_i, modality in enumerate(wm.ordered_modalities):
            curiosity_head = wm.curiosity_head[modality.name]
            jsd_m, mean_probs_m = curiosity_head.estimate_uncertainty(splits[m_i])
            jsd_terms.append(jsd_m.mean(-1))
            head_means[modality.name] = mean_probs_m.detach()
            member_logits_m = curiosity_head.forward_all(splits[m_i])
            member_probs[modality.name] = torch.softmax(
                member_logits_m, dim=-1
            ).detach()
            obs_logits = wm.head_observations[modality.name](splits[m_i])
            main_probs[modality.name] = torch.softmax(obs_logits, dim=-1).detach()

        j_ua = torch.stack(jsd_terms, dim=-1).sum(dim=-1)

        latent_r = rearrange_last_obs_position(outs, wm.tokens_per_obs)
        reward_feats = wm.head_rewards.model(latent_r)
        reward_logits = wm.head_rewards.head.linear(reward_feats)
        reward_probs = torch.softmax(reward_logits, dim=-1)
        hl = wm.head_rewards.head.hl_gauss_loss
        bin_centers = (hl.support[:-1] + hl.support[1:]) / 2
        expectation_symlog = (reward_probs * bin_centers).sum(dim=-1)
        # RewardHead.forward(x) == sym_exp(transform_from_probs(softmax(linear(model(x)))))
        # — identical math to the two lines above, computed once.
        expectation_real = sym_exp(expectation_symlog)
        ends_logits = wm.head_ends(latent_r)
        term_prob = torch.softmax(ends_logits, dim=-1)[..., 1]
        controller = self.controller_distribution(model_obs)

        return InstrumentedActionOutput(
            action_idx=int(action_idx),
            J_ua=float(j_ua.reshape(-1)[0]),
            controller_logits=controller.logits,
            controller_probs=controller.probs,
            reward_probs=reward_probs.reshape(-1).cpu(),
            reward_bins=bin_centers.detach().reshape(-1).cpu(),
            reward_expectation=float(expectation_real.reshape(-1)[0]),
            reward_expectation_head=float(expectation_real.reshape(-1)[0]),
            termination_prob=float(term_prob.reshape(-1)[0]),
            main_token_probs={
                k: v.cpu() for k, v in main_probs.items()
            },
            b_ua=block.detach().cpu(),
            g_ua=pred_latents.detach().cpu(),
            head_mean_probs={
                k: v.cpu() for k, v in head_means.items()
            },
            curiosity_member_probs={
                k: v.cpu() for k, v in member_probs.items()
            },
        )

    def evaluate_all_actions(
        self,
        model_obs: dict,
        prior_context_emb: torch.Tensor,
        recurrent_state,
        n_actions: int = 43,
    ) -> list[InstrumentedActionOutput]:
        return [
            self.evaluate_action(
                model_obs, prior_context_emb, recurrent_state, a
            )
            for a in range(n_actions)
        ]

    @torch.no_grad()
    def evaluate_all_actions_batched(
        self,
        model_obs: dict,
        prior_context_flat: torch.Tensor,
        recurrent_state,
    ) -> list[InstrumentedActionOutput]:
        """All 43 candidates in ONE forward pass (B=43).

        Numerically equivalent to evaluate_all_actions within float noise;
        verified by test_instrumentation.py::test_batched_matches_sequential.
        The live recurrent state is never touched — a cloned copy is expanded.
        """
        wm = self._wm
        device = next(wm.parameters()).device
        n = wm.tokens_per_obs

        obs_emb = embed_obs_block(wm, self._agent.tokenizer, model_obs, device)
        if prior_context_flat is not None:
            obs_emb = prior_context_flat
        act_embs = torch.cat(
            [embed_action_flat(wm, a, device) for a in range(43)], dim=0
        )  # (43, m, E)
        obs_rep = obs_emb.expand(act_embs.shape[0], -1, -1)
        blocks = torch.cat([obs_rep, act_embs], dim=1)  # (43, k1, E)

        expanded = _expand_recurrent_state(
            _clone_recurrent_state(recurrent_state), blocks.shape[0]
        )
        outs = wm.forward_inference(blocks.unsqueeze(1), recurrent_state=expanded)
        preds = wm.compute_next_obs_pred_latents(expanded)[0]  # (43, n, E)

        sizes = [wm.tokens_per_obs_dict[m] for m in wm.ordered_modalities]
        splits = torch.split(preds, sizes, dim=1)
        jsd_terms, main_probs, head_means, member_probs = [], {}, {}, {}
        for i, modality in enumerate(wm.ordered_modalities):
            curiosity_head = wm.curiosity_head[modality.name]
            jsd_m, mean_probs_m = curiosity_head.estimate_uncertainty(splits[i])
            jsd_terms.append(jsd_m.mean(-1))
            head_means[modality.name] = mean_probs_m.detach().cpu()
            member_logits_m = curiosity_head.forward_all(splits[i])
            member_probs[modality.name] = torch.softmax(
                member_logits_m, dim=-1
            ).detach().cpu()
            obs_logits = wm.head_observations[modality.name](splits[i])
            main_probs[modality.name] = (
                torch.softmax(obs_logits, dim=-1).detach().cpu()
            )
        j_all = torch.stack(jsd_terms, dim=-1).sum(dim=-1)  # (43,)

        latent_r = rearrange_last_obs_position(outs, n)     # (43, E)
        reward_feats = wm.head_rewards.model(latent_r)
        reward_logits = wm.head_rewards.head.linear(reward_feats)
        reward_probs = torch.softmax(reward_logits, dim=-1)
        hl = wm.head_rewards.head.hl_gauss_loss
        bin_centers = (hl.support[:-1] + hl.support[1:]) / 2
        expectation_real = sym_exp((reward_probs * bin_centers).sum(dim=-1))
        term_prob = torch.softmax(wm.head_ends(latent_r), dim=-1)[..., 1]
        controller = self.controller_distribution(model_obs)

        return [
            InstrumentedActionOutput(
                action_idx=a,
                J_ua=float(j_all[a]),
                controller_logits=controller.logits,
                controller_probs=controller.probs,
                reward_probs=reward_probs[a].cpu(),
                reward_bins=bin_centers.detach().cpu(),
                reward_expectation=float(expectation_real[a]),
                reward_expectation_head=float(expectation_real[a]),
                termination_prob=float(term_prob[a]),
                main_token_probs={
                    k: v[a : a + 1] for k, v in main_probs.items()
                },
                b_ua=blocks[a : a + 1].detach().cpu(),
                g_ua=preds[a : a + 1].detach().cpu(),
                head_mean_probs={k: v[a : a + 1] for k, v in head_means.items()},
                curiosity_member_probs={
                    k: v[:, a : a + 1] for k, v in member_probs.items()
                },
            )
            for a in range(blocks.shape[0])
        ]

    # -- read-only controller probe -------------------------------------------

    @torch.no_grad()
    def controller_distribution(
        self, model_obs: dict, temperature: float = 1.0
    ) -> ControllerDistribution:
        """(logits (43,), probs (43,)) without advancing the actor LSTM."""
        ac = self._agent.actor_critic
        saved_actor = ac.actor_state
        saved_critic = ac.critic_state
        try:
            ac.actor_state = _clone_recurrent_state(saved_actor)
            ac.critic_state = _clone_recurrent_state(saved_critic)
            with torch.autocast(
                self._agent.device.type,
                dtype=torch.bfloat16,
                enabled=self._agent.use_bf16_autocast,
            ):
                actor_input = self._agent._embed_obs(model_obs)
                actor_output = ac(inputs=actor_input)[0]
                distribution = actor_output.get_actions_distributions(temperature)
                logits = distribution.logits[:, -1]
                probs = distribution.probs[:, -1]
        finally:
            ac.actor_state = saved_actor
            ac.critic_state = saved_critic
        return ControllerDistribution(
            logits=logits.reshape(-1).cpu(),
            probs=probs.reshape(-1).cpu(),
        )

    # -- native reference -------------------------------------------------------

    @torch.no_grad()
    def native_estimate_uncertainty(
        self, model_obs: dict, prior_context_emb: torch.Tensor, recurrent_state
    ) -> float:
        """Un-instrumented J(u,a_controller): runs the pinned intrinsic-reward
        formula on CLONED state — the T4 comparison oracle."""
        wm = self._wm
        cloned = _clone_recurrent_state(recurrent_state)
        device = next(wm.parameters()).device
        obs_emb_flat = embed_obs_block(wm, self._agent.tokenizer, model_obs, device)
        outs = wm.forward_inference(
            obs_emb_flat.unsqueeze(1), recurrent_state=cloned
        )
        del outs
        preds = wm.compute_next_obs_pred_latents(cloned)[0]
        sizes = [wm.tokens_per_obs_dict[m] for m in wm.ordered_modalities]
        parts = torch.split(preds, sizes, dim=1)
        total = torch.cat(
            [
                wm.curiosity_head[m.name].estimate_uncertainty(parts[i])[0]
                .mean(-1, keepdim=True)
                for i, m in enumerate(wm.ordered_modalities)
            ],
            dim=-1,
        ).sum(dim=-1, keepdim=True)
        return float(total.reshape(-1)[0])


def rearrange_last_obs_position(outs: torch.Tensor, n_obs: int) -> torch.Tensor:
    """Reward/end latent = output at the last obs-token slot of the block."""
    return outs[:, n_obs - 1, :]


def sym_exp(x: torch.Tensor) -> torch.Tensor:
    # mirrors vendored utils/math.py sym_exp
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
