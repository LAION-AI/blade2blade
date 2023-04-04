from transformers import AutoTokenizer, AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
)
from tokenizers import pre_tokenizers

MODEL_MAPPINGS = [MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING]

SPECIAL_TOKENS = {"context": "<|prompter|>", "response": "<|Assistant|>"}


def format_history(context, eos_token):

    conversation = [
        "{}{}{}".format(
            SPECIAL_TOKENS["context" if i % 2 == 0 else "response"],
            context[i],
            eos_token,
        )
        for i in range(len(context))
    ]
    return conversation


def get_tokenizer(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model, truncation_side="left")

    if hasattr(config, "per_digit_tokens") and config.per_digit_tokens:
        tokenizer._tokenizer.pre_processor = pre_tokenizers.Digits(True)

    if config.special_tokens:
        special_tokens = {
            "pad_token": config.special_tokens.pad_token,
            "eos_token": config.special_tokens.eos_token,
            "sep_token": config.special_tokens.sep_token,
        }
        tokenizer.add_special_tokens(special_tokens)

    tokenizer.add_special_tokens(
        {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    )
    tokenizer.truncation_side = "left"

    return tokenizer


def get_model(name):

    model_config = AutoConfig.from_pretrained(name)
    for mapping in MODEL_MAPPINGS:
        model = mapping.get(type(model_config), None)
        if model is not None:
            return model.from_pretrained(name, config=model_config)
