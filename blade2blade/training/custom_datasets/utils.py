

SPECIAL_TOKENS = {
    "context":"<|prompter|>",
    "response":"<|Assistant|>"
}

def format_history(self,context, eos_token):
            
    conversation = ["{}{}{}".format(SPECIAL_TOKENS["context" if i%2==0 else "response"],context[i],eos_token)
                            for i in range(context)]
    return conversation