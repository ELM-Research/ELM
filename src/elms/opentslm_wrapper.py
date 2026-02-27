import sys
import os
import re

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from configs.constants import OPENTSLM_MODELS, PTB_ORDER


def _add_opentslm_to_path(opentslm_path):
    src = os.path.join(opentslm_path, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def build_opentslm(args, device):
    """Build, load checkpoint, and return an eval-ready OpenTSLMSP model."""
    cfg = OPENTSLM_MODELS[args.opentslm_model]
    _add_opentslm_to_path(args.opentslm_path)
    from model.llm.OpenTSLMSP import OpenTSLMSP
    from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder

    model = OpenTSLMSP(device=device, llm_id=cfg["base_llm"])
    model.encoder = TransformerCNNEncoder(max_patches=cfg["max_patches"]).to(device)
    model.enable_lora()
    ckpt = hf_hub_download(repo_id=cfg["checkpoint_repo"], filename=cfg["checkpoint_file"])
    model.load_from_file(ckpt)
    model.eval()
    return model


def build_prompt(ecg_signal, question, opentslm_path):
    """Build a FullPrompt for OpenTSLMSP from a raw 12-lead ECG and question."""
    _add_opentslm_to_path(opentslm_path)
    from prompt.full_prompt import FullPrompt
    from prompt.text_prompt import TextPrompt
    from prompt.text_time_series_prompt import TextTimeSeriesPrompt

    ts_prompts = []
    for i in range(ecg_signal.shape[0]):
        mean_val = float(np.mean(ecg_signal[i]))
        std_val = float(np.std(ecg_signal[i]))
        normalized = (ecg_signal[i] - mean_val) / std_val if std_val > 1e-6 else ecg_signal[i] - mean_val
        ts_prompts.append(TextTimeSeriesPrompt(
            f"ECG Lead {PTB_ORDER[i]} - sampled at ~250Hz, normalized (mean={mean_val:.3f}, std={std_val:.3f})",
            normalized.tolist(),
        ))

    pre = _PRE_PROMPT.format(question=question)
    return FullPrompt(TextPrompt(pre), ts_prompts, TextPrompt(_POST_PROMPT))


_ANSWER_RE = re.compile(r"Answer:\s*(.+?)(?=\n(?:[-#]{3,}|Output:|Question:)|\Z)", re.DOTALL)


def extract_answer(output):
    """Extract the final 'Answer: ...' text from model output."""
    matches = _ANSWER_RE.findall(output)
    if not matches:
        return ""
    return matches[-1].strip().replace("\n", " ").rstrip(" .")


def extract_qa(text):
    """Extract (question, answer) from a dataset text field.

    Handles both conversation format [{"from": ..., "value": ...}, ...]
    and list format [question_type, question, answer].
    """
    if isinstance(text[0], dict):
        question = " ".join(t["value"] for t in text if t["from"].lower() in ("human", "user"))
        answer = " ".join(t["value"] for t in text if t["from"].lower() not in ("human", "user", "system"))
    else:
        _, question, answer = text
    if isinstance(answer, list):
        answer = " ".join(answer)
    return question, answer


_PRE_PROMPT = """\
You are an expert cardiologist analyzing an ECG (electrocardiogram).

Clinical Context: 12-lead ECG recording.

Your task is to examine the ECG signal and answer the following medical question:

Question: {question}

Instructions:
- Begin by analyzing the time series without assuming a specific answer.
- Think step-by-step about what the observed patterns suggest regarding the cardiac condition.
- Write your rationale as a single, natural paragraph — do not use bullet points, numbered steps, or section headings.
- Do **not** mention any final answer until the very end.
- Consider the ECG morphology, intervals, and any abnormalities that relate to the question."""

_POST_PROMPT = """\
Based on your analysis of the ECG data, provide your answer.
Make sure that your last word is the answer. You MUST end your response with "Answer: \""""
