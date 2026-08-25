"""tests/test_instrumentation.py — §0.5 required tests T1,T2,T3,T4,T6,T9.

Slow: loads the pinned checkpoint once per module (cpu).
"""
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

N_ACTIONS = 43


@pytest.fixture(scope="module")
def simulus():
    from backend.simulus.craftax_env import SomaCraftaxEnv
    from backend.simulus.instrumentation import SimulusInstrumented
    from backend.simulus.runtime import load_simulus_agent

    agent, _cfg, device = load_simulus_agent("cpu")
    env = SomaCraftaxEnv(seed=42)
    obs = env.reset(seed=42)
    agent.reset_actor_critic(n=1, burnin_observations=None, mask_padding=None)

    wm = agent.world_model
    obs_model = env.to_model_obs(device)
    from backend.simulus.instrumentation import (
        build_block,
        embed_action_flat,
        embed_obs_block,
    )

    obs_emb = embed_obs_block(wm, agent.tokenizer, obs_model, device)
    block = build_block(wm, obs_emb, embed_action_flat(wm, 0, device))
    rs = wm.get_empty_state()
    with torch.no_grad():
        wm.forward_inference(block.unsqueeze(1), recurrent_state=rs)

    return {
        "agent": agent,
        "wm": wm,
        "env": env,
        "device": device,
        "model_obs": obs_model,
        "prior_context": obs_emb,
        "recurrent_state": rs,
        "inst": SimulusInstrumented(agent),
    }


def _rs_fingerprint(rs) -> bytes:
    import hashlib

    h = hashlib.sha256()

    def feed(obj):
        if obj is None:
            h.update(b"<none>")
        elif isinstance(obj, torch.Tensor):
            h.update(obj.detach().cpu().numpy().tobytes())
        elif isinstance(obj, tuple):
            for x in obj:
                feed(x)
        else:
            h.update(str(getattr(obj, "n", "")).encode())
            feed(getattr(obj, "state", None))

    feed(rs)
    return h.digest()


def test_t1_candidate_eval_does_not_mutate_live_state(simulus):
    inst = simulus["inst"]
    before = _rs_fingerprint(simulus["recurrent_state"])
    inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    after = _rs_fingerprint(simulus["recurrent_state"])
    assert before == after


def test_t2_permutation_invariance(simulus):
    inst = simulus["inst"]
    forward = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    order = list(range(N_ACTIONS))
    np.random.default_rng(3).shuffle(order)
    shuffled_map = {
        a: inst.evaluate_action(
            simulus["model_obs"], simulus["prior_context"],
            simulus["recurrent_state"], a,
        )
        for a in order
    }
    for out in forward:
        other = shuffled_map[out.action_idx]
        assert out.J_ua == pytest.approx(other.J_ua, abs=1e-6)
        assert out.termination_prob == pytest.approx(other.termination_prob, abs=1e-6)
        assert torch.allclose(out.reward_probs, other.reward_probs, atol=1e-6)


def test_t3_determinism_same_inputs(simulus):
    inst = simulus["inst"]
    run_a = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    run_b = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    for a, b in zip(run_a, run_b):
        assert a.J_ua == pytest.approx(b.J_ua, abs=1e-7)
        assert a.reward_expectation == pytest.approx(b.reward_expectation, abs=1e-7)


def test_t4_matches_native_estimate_uncertainty(simulus):
    """Independent recompute via vendored calls only — no SOMA helpers."""
    inst, wm, agent = simulus["inst"], simulus["wm"], simulus["agent"]
    outs = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )

    # Recompute J(u,a) for one candidate directly from vendored primitives.
    a_probe = int(np.argmax([o.J_ua for o in outs]))
    from backend.simulus.instrumentation import (
        _clone_recurrent_state,
        build_block,
        embed_action_flat,
        embed_obs_block,
    )

    obs_emb = embed_obs_block(wm, agent.tokenizer, simulus["model_obs"], simulus["device"])
    block = build_block(wm, obs_emb, embed_action_flat(wm, a_probe, simulus["device"]))

    cloned = _clone_recurrent_state(simulus["recurrent_state"])
    with torch.no_grad():
        wm.forward_inference(block.unsqueeze(1), recurrent_state=cloned)
        preds = wm.compute_next_obs_pred_latents(cloned)[0]
        sizes = [wm.tokens_per_obs_dict[m] for m in wm.ordered_modalities]
        parts = torch.split(preds, sizes, dim=1)
        native_j = float(
            torch.cat(
                [
                    wm.curiosity_head[m.name]
                    .estimate_uncertainty(parts[i])[0]
                    .mean(-1, keepdim=True)
                    for i, m in enumerate(wm.ordered_modalities)
                ],
                dim=-1,
            ).sum(dim=-1, keepdim=True)
        )
    assert outs[a_probe].J_ua == pytest.approx(native_j, abs=1e-5)


def test_curiosity_member_probabilities_are_categorical(simulus):
    # Given: one instrumented candidate evaluated by the released ensemble.
    output = simulus["inst"].evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 0,
    )

    # When: each modality's per-member probabilities are inspected.
    for modality in simulus["wm"].ordered_modalities:
        probabilities = output.curiosity_member_probs[modality.name]
        head = simulus["wm"].curiosity_head[modality.name]

        # Then: every ensemble member supplies a finite normalized categorical.
        assert probabilities.shape[0] == head.ensemble_size
        assert probabilities.shape[1] == 1
        assert probabilities.shape[2] == simulus["wm"].tokens_per_obs_dict[modality]
        assert torch.isfinite(probabilities).all()
        assert torch.all(probabilities >= 0)
        assert torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones_like(probabilities[..., 0]),
            atol=1e-6,
        )


def test_curiosity_member_probabilities_recompute_j_ua(simulus):
    # Given: exposed per-member categorical distributions for one candidate.
    output = simulus["inst"].evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 7,
    )

    # When: JSD is recomputed independently from those distributions.
    recomputed = torch.zeros(())
    for probabilities in output.curiosity_member_probs.values():
        safe_member = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
        member_entropy = -(probabilities * safe_member.log()).sum(dim=-1)
        mean_probabilities = probabilities.mean(dim=0)
        safe_mean = mean_probabilities.clamp_min(
            torch.finfo(mean_probabilities.dtype).tiny
        )
        mean_entropy = -(mean_probabilities * safe_mean.log()).sum(dim=-1)
        recomputed += (mean_entropy - member_entropy.mean(dim=0)).mean(dim=-1)[0]

    # Then: the exposed distributions are exactly the source of J(u,a).
    assert output.J_ua == pytest.approx(float(recomputed), abs=1e-6)


def test_t5_reward_expectation_is_independent_of_curiosity_uncertainty(
    simulus, monkeypatch,
):
    # Given: a baseline candidate and a temporary curiosity-only logit transform.
    inst = simulus["inst"]
    baseline = inst.evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 11,
    )
    modality = simulus["wm"].ordered_modalities[0]
    head = simulus["wm"].curiosity_head[modality.name]
    released_forward_all = head.forward_all

    def contracted_forward_all(latents):
        return released_forward_all(latents) * 0.5

    monkeypatch.setattr(head, "forward_all", contracted_forward_all)

    # When: only curiosity-head uncertainty changes.
    perturbed = inst.evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 11,
    )

    # Then: reward inference remains exclusively a reward-head contract.
    assert perturbed.J_ua != pytest.approx(baseline.J_ua, abs=1e-7)
    assert perturbed.reward_expectation == pytest.approx(
        baseline.reward_expectation, abs=1e-7
    )


def test_t6_reward_expectation_probs_times_bins(simulus):
    inst = simulus["inst"]
    outs = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    for o in outs:
        probs = o.reward_probs.numpy() if isinstance(o.reward_probs, torch.Tensor) else o.reward_probs
        bins = o.reward_bins.numpy() if isinstance(o.reward_bins, torch.Tensor) else o.reward_bins
        assert np.all(np.isfinite(probs)) and np.all(probs >= -1e-6)
        assert abs(float(probs.sum()) - 1.0) < 1e-3
        manual_symlog = float((probs * bins).sum())
        manual_real = float(np.sign(manual_symlog) * (np.exp(abs(manual_symlog)) - 1))
        assert o.reward_expectation == pytest.approx(manual_real, rel=1e-4, abs=1e-7)
        assert o.reward_expectation == pytest.approx(
            o.reward_expectation_head, rel=1e-4, abs=1e-7
        )


def test_t9_clone_bit_identical_continuations(simulus):
    wm = simulus["wm"]
    from backend.simulus.instrumentation import (
        _clone_recurrent_state,
        rearrange_last_obs_position,
    )

    live = simulus["recurrent_state"]
    clone = _clone_recurrent_state(live)
    assert _rs_fingerprint(live) == _rs_fingerprint(clone)

    ctx = simulus["prior_context"]
    from backend.simulus.instrumentation import build_block, embed_action_flat

    block = build_block(wm, ctx, embed_action_flat(wm, 3, simulus["device"]))
    with torch.no_grad():
        out_live = wm.forward_inference(block.unsqueeze(1), recurrent_state=live)
        out_clone = wm.forward_inference(block.unsqueeze(1), recurrent_state=clone)
    assert torch.equal(out_live, out_clone)
    assert torch.equal(
        rearrange_last_obs_position(out_live, wm.tokens_per_obs),
        rearrange_last_obs_position(out_clone, wm.tokens_per_obs),
    )


def test_controller_distribution_uses_released_embedding_and_is_read_only(
    simulus, monkeypatch,
):
    from backend.simulus.instrumentation import ControllerDistribution

    agent = simulus["agent"]
    inst = simulus["inst"]
    ac = agent.actor_critic
    saved_hc = ac.actor_state[0].clone(), ac.actor_state[1].clone()
    embedded_inputs = []
    released_embed_obs = agent._embed_obs

    def recording_embed_obs(model_obs):
        embedded = released_embed_obs(model_obs)
        embedded_inputs.append(embedded)
        return embedded

    monkeypatch.setattr(agent, "_embed_obs", recording_embed_obs)

    distribution = inst.controller_distribution(simulus["model_obs"])

    assert isinstance(distribution, ControllerDistribution)
    assert len(embedded_inputs) == 1
    assert distribution.logits.shape == (N_ACTIONS,)
    assert distribution.probs.shape == (N_ACTIONS,)
    assert torch.isfinite(distribution.logits).all()
    assert torch.isfinite(distribution.probs).all()
    assert torch.all(distribution.probs >= 0)
    assert abs(float(distribution.probs.sum()) - 1.0) < 1e-6
    assert torch.allclose(
        torch.softmax(distribution.logits, dim=-1), distribution.probs, atol=1e-6
    )
    now_hc = ac.actor_state[0].clone(), ac.actor_state[1].clone()
    assert torch.equal(saved_hc[0], now_hc[0])
    assert torch.equal(saved_hc[1], now_hc[1])


def test_action_output_contains_controller_distribution(simulus):
    output = simulus["inst"].evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 5,
    )

    assert output.controller_logits.shape == (N_ACTIONS,)
    assert output.controller_probs.shape == (N_ACTIONS,)
    assert torch.allclose(
        torch.softmax(output.controller_logits, dim=-1),
        output.controller_probs,
        atol=1e-6,
    )


def test_batched_matches_sequential(simulus):
    inst = simulus["inst"]
    seq = inst.evaluate_all_actions(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    bat = inst.evaluate_all_actions_batched(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    assert len(bat) == 43
    for s, b in zip(seq, bat):
        assert s.action_idx == b.action_idx
        assert s.J_ua == pytest.approx(b.J_ua, rel=1e-3, abs=1e-7)
        assert s.termination_prob == pytest.approx(b.termination_prob, rel=1e-3, abs=1e-7)
        assert s.reward_expectation == pytest.approx(
            b.reward_expectation, rel=1e-3, abs=1e-7
        )
        assert torch.allclose(s.controller_logits, b.controller_logits, atol=1e-6)
        assert torch.allclose(s.controller_probs, b.controller_probs, atol=1e-6)
        assert s.curiosity_member_probs.keys() == b.curiosity_member_probs.keys()
        for modality in s.curiosity_member_probs:
            assert torch.allclose(
                s.curiosity_member_probs[modality],
                b.curiosity_member_probs[modality],
                rtol=1e-3,
                atol=1e-7,
            )


def test_batched_does_not_mutate_live_state(simulus):
    inst = simulus["inst"]
    before = _rs_fingerprint(simulus["recurrent_state"])
    inst.evaluate_all_actions_batched(
        simulus["model_obs"], simulus["prior_context"], simulus["recurrent_state"]
    )
    assert before == _rs_fingerprint(simulus["recurrent_state"])


def test_t10_manifest_records_live_instrumentation_contract(simulus, tmp_path):
    import json

    from backend.simulus import (
        CHECKPOINT_SHA256,
        CRAFTAX_VERSION,
        MODEL_REVISION,
        VENDORED_COMMIT,
    )
    from backend.simulus.instrumentation_manifest import (
        PreprocessingEvidence,
        build_instrumentation_manifest,
        write_instrumentation_manifest,
    )

    # Given: the live model, symbolic preprocessing, and a real measured output.
    output = simulus["inst"].evaluate_action(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"], 3,
    )
    symbolic_obs = simulus["env"].observation()

    # When: a deterministic manifest is built and atomically written.
    manifest = build_instrumentation_manifest(
        simulus["agent"],
        output,
        PreprocessingEvidence(symbolic_obs, simulus["model_obs"]),
    )
    destination = tmp_path / "instrumentation_manifest.json"
    write_instrumentation_manifest(destination, manifest)
    first_bytes = destination.read_bytes()
    write_instrumentation_manifest(destination, manifest)

    # Then: recorded shapes are measured from the live output, never hard-coded.
    decoded = json.loads(destination.read_text())
    tensor_shapes = {item["name"]: item["shape"] for item in decoded["tensors"]}
    assert tensor_shapes["b_ua"] == list(output.b_ua.shape)
    assert tensor_shapes["g_ua"] == list(output.g_ua.shape)
    assert tensor_shapes["reward_probs"] == list(output.reward_probs.shape)
    assert tensor_shapes["controller_logits"] == list(output.controller_logits.shape)
    for modality, probabilities in output.main_token_probs.items():
        assert tensor_shapes[f"main_token_probs.{modality}"] == list(
            probabilities.shape
        )
    for modality, probabilities in output.curiosity_member_probs.items():
        assert tensor_shapes[f"curiosity_member_probs.{modality}"] == list(
            probabilities.shape
        )

    assert decoded["action_conditioning"]["output_field"] == "b_ua"
    assert decoded["action_conditioning"]["prediction_field"] == "g_ua"
    assert decoded["jsd_pooling"] == {
        "ensemble": "categorical_jensen_shannon_divergence",
        "token": "mean",
        "modality": "sum",
    }
    preprocess = {item["name"]: item for item in decoded["symbolic_preprocessing"]}
    for name, value in symbolic_obs.items():
        assert preprocess[name]["input_shape"] == list(value.shape)
        assert preprocess[name]["input_dtype"] == str(value.dtype)
    assert preprocess["vector"]["transform"] == "sym_log"
    assert preprocess["token"]["transform"] == "categorical_identity"
    assert preprocess["token_2d"]["transform"] == "categorical_identity"
    assert decoded["artifacts"] == {
        "model_revision": MODEL_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "code_commit": VENDORED_COMMIT,
        "craftax_version": CRAFTAX_VERSION,
        "load_tokenizer": False,
        "load_world_model": True,
        "load_actor_critic": True,
        "strict_state_dict": True,
        "weights_only": True,
    }
    assert decoded["world_model"]["tokens_per_obs"] == simulus["wm"].tokens_per_obs
    assert decoded["world_model"]["embedding_dim"] == output.b_ua.shape[-1]
    assert destination.read_bytes() == first_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_t11_instrumented_indices_are_exact_craftax_action_indices(simulus):
    from craftax.craftax.constants import Action

    # Given/When: the installed enum and live batched instrumentation are read.
    enum_indices = [int(action.value) for action in Action]
    outputs = simulus["inst"].evaluate_all_actions_batched(
        simulus["model_obs"], simulus["prior_context"],
        simulus["recurrent_state"],
    )

    # Then: both expose the exact contiguous no-autoreset action index set.
    assert enum_indices == list(range(43))
    assert [output.action_idx for output in outputs] == enum_indices
