import argparse
from typing import Any, Dict, Iterable, Mapping

from elms.connectors.linear_proj import LinearProjection

from configs.constants import ECG_ENCODERS, VISION_ENCODERS


def merge_dicts(*parts: Mapping[str, Any], allow_override: Iterable[str] = ()) -> Dict[str, Any]:
    """Merge dict-like parts with duplicate-key protection.
    Keys in `allow_override` are allowed to be overwritten by later parts.
    Later parts win for allowed keys; duplicates for other keys raise."""
    out: Dict[str, Any] = {}
    allowed = set(allow_override)
    for p in parts:
        for k, v in p.items():
            if k in out and k not in allowed:
                raise ValueError(f"Duplicate component keys when merging: {k}")
            out[k] = v
    return out


class ConnectNN:
    def __init__(self, llm_components: dict, encoder_components: dict, args: argparse.Namespace):
        self.args = args
        self.llm_components = llm_components
        self.encoder_components = encoder_components

    def connect_nn(
        self,
    ):
        if self.args.elm == "llava":
            encoder_llm_components = self.build_llava()
        elif self.args.elm == "fuyu":
            encoder_llm_components = self.build_fuyu()
        else:
            encoder_llm_components = {"elm": self.llm_components["llm"]}
        return merge_dicts(
            self.encoder_components,
            self.llm_components,
            encoder_llm_components,
            allow_override=("find_unused_parameters",),
        )

    def build_llava(
        self,
    ):
        from elms.llm_encoders.llava import LLaVA
        if self.args.encoder in VISION_ENCODERS:
            projection_dim = VISION_ENCODERS[self.args.encoder]["projection_dim"]
        else:
            projection_dim = ECG_ENCODERS[self.args.encoder]["projection_dim"]
        projection_layer = LinearProjection(projection_dim, self.args.llm)
        encoder_llm = LLaVA(
            self.llm_components["llm"], self.encoder_components["encoder"],
            projection_layer, self.args.update_encoder,
            True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_fuyu(
        self,
    ):
        from elms.llm_encoders.fuyu import Fuyu
        projection_dim = 12 * self.args.segment_len
        projection_layer = LinearProjection(projection_dim, self.args.llm)
        encoder_llm = Fuyu(self.llm_components["llm"], projection_layer,
                           True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}