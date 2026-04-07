"""Count parameters for each ELM component (encoder, connector, llm)."""

from configs.config import get_args
from configs.constants import HF_LLMS, SIGNAL_TOKEN_PLACEHOLDER
from elms.build_elm import BuildELM
from transformers import AutoTokenizer


def build_tokenizer(args):
    llm_tokenizer = AutoTokenizer.from_pretrained(HF_LLMS[args.llm]["tokenizer"])
    if getattr(llm_tokenizer, "pad_token", None) is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    tokens_to_add = HF_LLMS[args.llm]["tokens_to_add"]
    tokens_to_add["additional_special_tokens"].append(SIGNAL_TOKEN_PLACEHOLDER)
    llm_tokenizer.add_special_tokens(tokens_to_add)
    return llm_tokenizer


def fmt(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def main():
    args = get_args("eval")
    tokenizer = build_tokenizer(args)
    elm = BuildELM(args).build_elm(tokenizer)["elm"]

    total = 0
    for name, module in elm.named_children():
        n = sum(p.numel() for p in module.parameters())
        total += n
        print(f"  {name:<20s} {fmt(n):>10s}  ({n:,})")
    print(f"  {'TOTAL':<20s} {fmt(total):>10s}  ({total:,})")


if __name__ == "__main__":
    main()
