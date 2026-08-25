"""Deterministic manifest for the live Simulus instrumentation contract."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from backend.simulus import (
    CHECKPOINT_SHA256,
    CRAFTAX_VERSION,
    MODEL_REVISION,
    VENDORED_COMMIT,
)
from backend.simulus.instrumentation import InstrumentedActionOutput


class InstrumentationManifest(TypedDict):
    schema_version: int
    tensors: list[dict[str, str | list[int]]]
    action_conditioning: dict[str, str | list[str]]
    jsd_pooling: dict[str, str]
    symbolic_preprocessing: list[dict[str, str | list[int]]]
    artifacts: dict[str, str | bool]
    world_model: dict[str, int | dict[str, int]]


@dataclass(frozen=True, slots=True)
class PreprocessingEvidence:
    symbolic_observation: dict[str, np.ndarray]
    model_observation: dict


def _tensor_specs(output: InstrumentedActionOutput) -> list[dict[str, str | list[int]]]:
    direct = {
        "b_ua": output.b_ua,
        "controller_logits": output.controller_logits,
        "controller_probs": output.controller_probs,
        "g_ua": output.g_ua,
        "reward_bins": output.reward_bins,
        "reward_probs": output.reward_probs,
    }
    specs = [
        {"name": name, "shape": list(tensor.shape)}
        for name, tensor in direct.items()
        if tensor is not None
    ]
    grouped = {
        "curiosity_member_probs": output.curiosity_member_probs,
        "head_mean_probs": output.head_mean_probs,
        "main_token_probs": output.main_token_probs,
    }
    for field, tensors in grouped.items():
        specs.extend(
            {"name": f"{field}.{name}", "shape": list(tensor.shape)}
            for name, tensor in tensors.items()
        )
    return sorted(specs, key=lambda item: str(item["name"]))


def build_instrumentation_manifest(
    agent,
    output: InstrumentedActionOutput,
    preprocessing_evidence: PreprocessingEvidence,
) -> InstrumentationManifest:
    wm = agent.world_model
    symbolic_observation = preprocessing_evidence.symbolic_observation
    model_by_name = {
        modality.name: tensor
        for modality, tensor in preprocessing_evidence.model_observation.items()
    }
    transforms = {
        "token": "categorical_identity",
        "token_2d": "categorical_identity",
        "vector": "sym_log",
    }
    preprocessing = [
        {
            "name": name,
            "input_shape": list(symbolic_observation[name].shape),
            "input_dtype": str(symbolic_observation[name].dtype),
            "output_shape": list(model_by_name[name].shape),
            "output_dtype": str(model_by_name[name].dtype).removeprefix("torch."),
            "transform": transforms[name],
        }
        for name in sorted(symbolic_observation)
    ]
    tokens_per_obs = {
        modality.name: int(size)
        for modality, size in wm.tokens_per_obs_dict.items()
    }
    return {
        "schema_version": 1,
        "tensors": _tensor_specs(output),
        "action_conditioning": {
            "output_field": "b_ua",
            "operation": "concatenate",
            "order": ["symbolic_observation_embeddings_t", "action_embedding_a"],
            "prediction_field": "g_ua",
        },
        "jsd_pooling": {
            "ensemble": "categorical_jensen_shannon_divergence",
            "token": "mean",
            "modality": "sum",
        },
        "symbolic_preprocessing": preprocessing,
        "artifacts": {
            "model_revision": MODEL_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "code_commit": VENDORED_COMMIT,
            "craftax_version": CRAFTAX_VERSION,
            "load_tokenizer": False,
            "load_world_model": True,
            "load_actor_critic": True,
            "strict_state_dict": True,
            "weights_only": True,
        },
        "world_model": {
            "tokens_per_obs": int(wm.tokens_per_obs),
            "tokens_per_action": int(wm.tokens_per_action),
            "tokens_per_obs_by_modality": tokens_per_obs,
            "context_length": int(wm.context_length),
            "action_count": int(output.controller_probs.shape[-1]),
            "embedding_dim": int(output.b_ua.shape[-1]),
        },
    }


def write_instrumentation_manifest(
    destination: Path,
    manifest: InstrumentationManifest,
) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
