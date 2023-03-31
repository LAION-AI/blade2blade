from transformers import AutoTokenizer

SPECIAL_TOKENS = {
    "context":"<|prompter|>",
    "response":"<|Assistant|>"
}

def format_history(context, eos_token):
            
    conversation = ["{}{}{}".format(SPECIAL_TOKENS["context" if i%2==0 else "response"],context[i],eos_token)
                            for i in range(len(context))]
    return conversation


def get_tokenizer(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    
    if config.special_tokens:
        special_tokens = {
            "pad_token":config.special_tokens.pad_token,
            "eos_token":config.special_tokens.eos_token,
            "sep_token":config.special_tokens.sep_token
        }
        tokenizer.add_special_tokens(special_tokens)

    tokenizer.add_special_tokens({"additional_special_tokens":list(SPECIAL_TOKENS.values())})
    tokenizer.truncation_side = "left"

    return tokenizer