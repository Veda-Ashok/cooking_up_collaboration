import types
from typing import Any


VALID_SAMPLING_MODES = {"sample", "argmax"}


def normalize_sampling_mode(mode: str | None, default: str = "sample") -> str:
    if mode is None:
        mode = default
    normalized = str(mode).strip().lower()
    if normalized not in VALID_SAMPLING_MODES:
        raise ValueError(f"Unsupported sampling_mode={mode}. Use one of: {sorted(VALID_SAMPLING_MODES)}")
    return normalized


def validate_sampling_temperature(value: float) -> float:
    temperature = float(value)
    if temperature <= 0:
        raise ValueError(f"sampling_temperature must be > 0, got {value}")
    return temperature


def configure_model_sampling_temperature(model: Any, sampling_temperature: float) -> dict[str, Any]:
    """Patch SB3 policy action distribution to support temperature-scaled sampling.

    This affects both rollout sampling during training and stochastic inference
    (`predict(..., deterministic=False)`).
    """
    temperature = validate_sampling_temperature(sampling_temperature)
    policy = getattr(model, "policy", None)
    if policy is None:
        raise ValueError("Model does not expose a `policy` attribute.")

    previous_temp = getattr(policy, "_sampling_temperature", None)
    if previous_temp == temperature and getattr(policy, "_sampling_patch_applied", False):
        return {"patched": False, "reason": "already_configured", "sampling_temperature": float(temperature)}

    original_method = getattr(policy, "_sampling_original_get_action_dist_from_latent", None)
    if original_method is None:
        if not hasattr(policy, "_get_action_dist_from_latent"):
            raise ValueError("Policy does not expose `_get_action_dist_from_latent`; unsupported policy type.")
        original_method = policy._get_action_dist_from_latent
        policy._sampling_original_get_action_dist_from_latent = original_method

    if temperature == 1.0:
        policy._get_action_dist_from_latent = original_method
        policy._sampling_patch_applied = False
        policy._sampling_temperature = 1.0
        return {"patched": False, "reason": "identity_temperature", "sampling_temperature": 1.0}

    def _patched_get_action_dist_from_latent(self, *args, **kwargs):
        if not args:
            return original_method(*args, **kwargs)

        latent_pi = args[0]
        if not hasattr(self, "action_net") or not hasattr(self, "action_dist"):
            return original_method(*args, **kwargs)
        if not hasattr(self.action_dist, "proba_distribution"):
            return original_method(*args, **kwargs)

        # For discrete policies (Overcooked uses Discrete(6)), scale logits before sampling.
        action_logits = self.action_net(latent_pi) / float(self._sampling_temperature)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    policy._get_action_dist_from_latent = types.MethodType(_patched_get_action_dist_from_latent, policy)
    policy._sampling_patch_applied = True
    policy._sampling_temperature = float(temperature)
    return {"patched": True, "reason": "patched", "sampling_temperature": float(temperature)}
