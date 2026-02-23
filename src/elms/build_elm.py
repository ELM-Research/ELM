import argparse
import torch

from elms.build_llm import BuildLLM
from elms.build_encoder import BuildEncoder
from elms.connect_nns import ConnectNN

from utils.gpu_manager import is_main

class BuildELM:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_elm(self, llm_tokenizer):
        elm_components = None
        llm_components = BuildLLM(self.args, llm_tokenizer,).build_llm()
        encoder_components = BuildEncoder(self.args).build_encoder()
        elm_components = ConnectNN(llm_components, encoder_components, self.args).connect_nn()
        assert elm_components is not None, print("ELM Components is None")
        if self.args.elm_ckpt:
            self.load_elm_checkpoint(elm_components)
        return elm_components

    def load_elm_checkpoint(self, elm_components):
        elm_checkpoint = torch.load(self.args.elm_ckpt, map_location="cpu", weights_only=False)
        elm_components["elm"].load_state_dict(elm_checkpoint["model_state_dict"], strict=False)
        if is_main():
            print(f"Loaded ELM checkpoint from {self.args.elm_ckpt}")
