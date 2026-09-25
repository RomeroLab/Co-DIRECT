
from __future__ import annotations

import hashlib
import math
import pickle
import random
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import torch

from dive.signed_value.critic_protocol import (
    CRITIC_CALLS,
)
from dive.signed_value.critic_protocol import (
    PERTURBATION_NAMES,
    CriticRequest,
    CriticResponse,
)

class AF2CriticError(RuntimeError):
    pass

ModelFactory = Callable[..., object]
DeviceProvider = Callable[[str], list[object]]

class _InjectedFactoryDevice:

    platform = "gpu"

    def memory_stats(self) -> dict[str, int]:
        return {
            "peak_bytes_in_use": 0,
            "peak_bytes_reserved": 0,
            "peak_pool_bytes": 0,
            "bytes_limit": 0,
        }

def _injected_device_provider(platform: str) -> list[object]:
    return [_InjectedFactoryDevice()]

def hash_parameter_tree(tree: object) -> str:

    digest = hashlib.sha256()
    leaves = sorted(_array_leaves(tree), key=lambda item: item[0])
    if not leaves:
        raise AF2CriticError("loaded JAX parameter tree has no array leaves")
    for path, value in leaves:
        array = np.ascontiguousarray(np.asarray(value))
        for field in (
            path.encode("utf-8"),
            array.dtype.str.encode("ascii"),
            repr(tuple(array.shape)).encode("ascii"),
            array.tobytes(order="C"),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()

class FrozenAF2SequenceCritic:

    BINDER_REWARD_NAMES = (
        "plddt",
        "pae",
        "exp_res",
        "con",
        "i_con",
        "i_pae",
        "rg",
        "i_ptm",
        "i_ptm_energy",
        "nc_termini",
        "helix_binder",
        "alignment_bb_ca_binder",
        "dgram_cce",
        "min_ipae",
        "min_ipsae",
        "avg_ipsae",
        "max_ipsae",
        "min_ipsae_10",
        "max_ipsae_10",
        "avg_ipsae_10",
    )
    MODEL_NAMES = ("model_2_multimer_v3",)

    def __init__(
        self,
        config: Mapping[str, object],
        *,
        model_factory: ModelFactory | None = None,
        device_provider: DeviceProvider | None = None,
    ) -> None:
        self.config = _validate_config(config)
        if device_provider is None and model_factory is None:
            jax = _import_jax()
            device_provider = jax.devices
        elif device_provider is None:
            device_provider = _injected_device_provider
        devices = list(device_provider("gpu"))
        if len(devices) != 1 or getattr(devices[0], "platform", "gpu") != "gpu":
            raise AF2CriticError(
                f"critic requires exactly one visible JAX GPU, observed {len(devices)}"
            )
        self.device = devices[0]
        if model_factory is None:
            model_factory = _import_model_factory()
        parameter_path = Path(self.config["parameter_path"])
        self.model = model_factory(
            protocol="binder",
            use_multimer=True,
            num_recycles=0,
            recycle_mode="last",
            use_initial_guess=True,
            use_initial_atom_pos=False,
            use_bfloat16=self.config["use_bfloat16"],
            num_seq=1,
            learning_rate=1.0,
            data_dir=str(parameter_path.parent),
            device=self.device,
            model_names=["model_2_multimer_v3"],
        )
        self.loaded_model_names = tuple(getattr(self.model, "_model_names", ()))
        if self.loaded_model_names != self.MODEL_NAMES:
            raise AF2CriticError(
                f"loaded model identity drifted: {self.loaded_model_names!r}"
            )
        self.parameter_tree_sha256 = hash_parameter_tree(
            getattr(self.model, "_model_params", None)
        )
        self.weights = {name: 0.0 for name in self.BINDER_REWARD_NAMES}
        self.weights["i_pae"] = -1.0

    def evaluate(self, request: CriticRequest) -> CriticResponse:

        request.assert_exact()
        joint_reward, sequence_gradient, joint_state = self._evaluate_one(
            request, request.joint_logits, requires_grad=True
        )
        if sequence_gradient is None:
            raise AF2CriticError("gradient evaluation did not return a sequence gradient")
        repeat_reward, _, repeat_state = self._evaluate_one(
            request, request.joint_logits, requires_grad=False
        )

        repeat2_reward, _, repeat2_state = self._evaluate_one(
            request, request.joint_logits, requires_grad=False
        )

        _joint2_reward, sequence_gradient2, joint2_state = self._evaluate_one(
            request, request.joint_logits, requires_grad=True
        )
        if sequence_gradient2 is None:
            raise AF2CriticError("second gradient evaluation returned no gradient")
        rewards: dict[str, float] = {}
        states: list[Mapping[str, str]] = [
            joint_state,
            repeat_state,
            repeat2_state,
            joint2_state,
        ]
        for name in PERTURBATION_NAMES:
            reward, _, state = self._evaluate_one(
                request, request.perturbed_logits[name], requires_grad=False
            )
            rewards[name] = reward
            states.append(state)
        memory_stats = _memory_evidence(self.device)
        response = CriticResponse(
            joint_reward=joint_reward,
            repeat_reward=repeat_reward,
            repeat2_reward=repeat2_reward,
            sequence_gradient2=sequence_gradient2,
            perturbed_rewards=rewards,
            sequence_gradient=sequence_gradient,
            loaded_model_names=self.loaded_model_names,
            parameter_tree_sha256=self.parameter_tree_sha256,
            request_states=states,
            memory_stats=memory_stats,
            critic_calls=CRITIC_CALLS,
        )
        if sequence_gradient.shape != request.joint_logits.shape:
            raise AF2CriticError(
                "gradient shape must equal binder logits: "
                f"{tuple(sequence_gradient.shape)} != {tuple(request.joint_logits.shape)}"
            )
        response.assert_exact()
        return response

    def _evaluate_one(
        self, request: CriticRequest, sequence: torch.Tensor, *, requires_grad: bool
    ) -> tuple[float, torch.Tensor | None, Mapping[str, str]]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        python_before = ""
        numpy_before = ""
        python_after = ""
        numpy_after = ""
        prep_key = ""
        run_key = ""
        try:
            random.seed(0)
            np.random.seed(0)
            python_before = _state_hash(random.getstate())
            numpy_before = _state_hash(np.random.get_state())
            sequence_array = np.ascontiguousarray(sequence.detach().cpu().numpy())
            self.model.prep_inputs(
                pdb_filename=str(request.target_path),
                target_chain=request.target_chains,
                binder_len=len(sequence_array),
                seq=sequence_array,
                struct=None,
                seed=0,
                use_binder_template=False,
                rm_template_ic=False,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                hotspot=None,
            )
            prep_key = _model_key_hash(self.model)
            self.model.set_opt(
                hard=False,
                soft=True,
                temp=1.0,
                alpha=2.0,
                dropout=False,
                pssm_hard=False,
                weights=dict(self.weights),
            )
            self.model.run(
                num_recycles=0,
                sample_models=False,
                models=["model_2_multimer_v3"],
                backprop=requires_grad,
            )
            run_key = _model_key_hash(self.model)
            aux = getattr(self.model, "aux", None)
            if not isinstance(aux, Mapping) or "loss" not in aux:
                raise AF2CriticError("ColabDesign run did not return a scalar loss")
            reward = float(np.asarray(aux["loss"]).reshape(()))
            if not math.isfinite(reward):
                raise AF2CriticError("critic reward is non-finite")
            gradient = _sequence_gradient(aux, sequence.shape) if requires_grad else None
            python_after = _state_hash(random.getstate())
            numpy_after = _state_hash(np.random.get_state())
        except AF2CriticError:
            raise
        except Exception as error:
            raise AF2CriticError(f"model-2 critic request failed: {error}") from error
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        state = {
            "python_state_before_sha256": python_before,
            "numpy_state_before_sha256": numpy_before,
            "model_key_after_prep_sha256": prep_key,
            "model_key_after_run_sha256": run_key,
            "python_state_after_sha256": python_after,
            "numpy_state_after_sha256": numpy_after,
        }
        return reward, gradient, state

def _validate_config(config: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise AF2CriticError("critic config must be a mapping")
    required = {
        "protocol": "binder",
        "model_name": "model_2_multimer_v3",
        "use_multimer": True,
        "num_recycles": 0,
        "recycle_mode": "last",
        "use_initial_guess": True,
        "use_initial_atom_pos": False,

        "use_bfloat16": False,
        "num_seq": 1,
        "preparation_seed": 0,
        "reward": "i_pae",
        "reward_weight": -1.0,
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise AF2CriticError(
                f"critic config {name} must be {expected!r}, observed {config.get(name)!r}"
            )
    raw_path = config.get("parameter_path")
    if not isinstance(raw_path, str):
        raise AF2CriticError("critic parameter_path must be a string")
    parameter_path = Path(raw_path).expanduser().resolve()
    if not parameter_path.is_file():
        raise AF2CriticError(f"critic parameter file is missing: {parameter_path}")
    resolved = dict(config)
    resolved["parameter_path"] = str(parameter_path)
    return resolved

def _import_jax() -> object:
    try:
        import jax
    except ImportError as error:
        raise AF2CriticError("JAX is unavailable in the critic child") from error
    return jax

def _import_model_factory() -> ModelFactory:
    try:
        from colabdesign import mk_afdesign_model
    except ImportError as error:
        raise AF2CriticError("ColabDesign is unavailable in the critic child") from error
    return mk_afdesign_model

def _array_leaves(tree: object, path: str = "root") -> list[tuple[str, object]]:
    if isinstance(tree, Mapping):
        leaves: list[tuple[str, object]] = []
        for key in sorted(tree, key=lambda item: str(item)):
            leaves.extend(_array_leaves(tree[key], f"{path}.{key}"))
        return leaves
    if isinstance(tree, (list, tuple)):
        leaves = []
        for index, value in enumerate(tree):
            leaves.extend(_array_leaves(value, f"{path}[{index}]"))
        return leaves
    try:
        array = np.asarray(tree)
    except Exception as error:
        raise AF2CriticError(f"parameter tree leaf {path} is not array-like") from error
    if array.dtype == object:
        raise AF2CriticError(f"parameter tree leaf {path} has object dtype")
    return [(path, tree)]

def _sequence_gradient(aux: Mapping[str, object], expected_shape: torch.Size) -> torch.Tensor:
    gradients = aux.get("grad")
    if not isinstance(gradients, Mapping) or "seq" not in gradients:
        raise AF2CriticError("gradient evaluation lacks aux['grad']['seq']")
    array = np.asarray(gradients["seq"])
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if tuple(array.shape) != tuple(expected_shape):
        raise AF2CriticError(
            "gradient shape must equal binder logits: "
            f"{tuple(array.shape)} != {tuple(expected_shape)}"
        )
    if not np.issubdtype(array.dtype, np.floating) or not bool(np.isfinite(array).all()):
        raise AF2CriticError("sequence gradient must be finite floating point")
    if float(np.linalg.norm(array.astype(np.float64, copy=False))) == 0.0:
        raise AF2CriticError("sequence gradient has zero norm")
    return torch.from_numpy(np.ascontiguousarray(array)).cpu()

def _model_key_hash(model: object) -> str:
    key: object | None = None
    key_method = getattr(model, "key", None)
    owner = getattr(key_method, "__self__", None)
    if owner is not None:
        candidate = getattr(owner, "key", None)
        if candidate is not None and not callable(candidate):
            key = candidate
    if key is None:
        holder = getattr(model, "_key_holder", None)
        key = getattr(holder, "key", None)
    if key is None:
        key = getattr(model, "_key", None)
    if key is None:
        raise AF2CriticError("cannot inspect ColabDesign model key state")
    array = np.ascontiguousarray(np.asarray(key))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()

def _state_hash(state: object) -> str:
    return hashlib.sha256(pickle.dumps(state, protocol=5)).hexdigest()

def _memory_evidence(device: object) -> dict[str, int]:
    try:
        observed = device.memory_stats()
    except Exception as error:
        raise AF2CriticError(f"cannot query JAX device memory stats: {error}") from error
    if not isinstance(observed, Mapping):
        raise AF2CriticError("JAX device memory_stats did not return a mapping")
    for name in ("peak_bytes_in_use", "peak_bytes_reserved", "peak_pool_bytes", "bytes_limit"):
        if name not in observed:
            raise AF2CriticError(f"missing JAX memory stat {name}")
    stats: dict[str, int] = {}
    for name, value in observed.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise AF2CriticError(f"JAX memory stat {name} is not an integer")
        stats[str(name)] = int(value)
    stats.update(
        {
            "critic_peak_allocated_bytes": stats["peak_bytes_in_use"],
            "critic_peak_reserved_bytes": stats["peak_bytes_reserved"],
            "critic_peak_pool_bytes": stats["peak_pool_bytes"],
            "critic_gpu_total_memory_bytes": stats["bytes_limit"],
        }
    )
    return stats
