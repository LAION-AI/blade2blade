from typing import Any, Dict

import torch
from transformers import AutoTokenizer
from transformers.pipelines import Conversation, ConversationalPipeline

from blade2blade.training.utils import get_model


class SafetyPipeline(ConversationalPipeline):
    def preprocess(
        self, conversation: Conversation, min_length_for_response=32
    ) -> Dict[str, Any]:
        if not isinstance(conversation, Conversation):
            raise ValueError("ConversationalPipeline, expects Conversation as inputs")
        if conversation.new_user_input is None:
            raise ValueError(
                f"Conversation with UUID {type(conversation.uuid)} does not contain new user input to process. "
                "Add user inputs with the conversation's `add_user_input` method"
            )
        inputs = []
        for is_user, text in conversation.iter_texts():
            if is_user:
                # We need to space prefix as it's being done within blenderbot
                inputs.append("<|prompter|>" + text + self.tokenizer.eos_token)
            else:
                # Generated responses should contain them already.
                inputs.append("<|assistant|>" + text + self.tokenizer.eos_token)

        input_ids, attn_mask = (
            self.tokenizer(
                "".join(inputs),
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            .to(self.device)
            .values()
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "conversation": conversation,
        }

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=False):
        output_ids = model_outputs["output_ids"]
        answer = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )
        return answer


class Blade2Blade:
    def __init__(self, model_name, **kwargs):
        self.model = get_model(model_name)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.config.max_length = self.tokenizer.model_max_length
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        self.pipeline = SafetyPipeline(
            self.model, self.tokenizer, self.device, minimum_tokens=0, **kwargs
        )

    def __call__(self, prompt: str, conversation=None):
        if not conversation:
            conversation = Conversation(prompt)
            resp = self.pipeline(conversation)
            return resp, conversation
        conversation.add_user_input(prompt)
        resp = self.pipeline(conversation)
        return resp, conversation

    def predict(self, prompt: str, **kwargs):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        output = self.model.generate(**inputs, **kwargs).detach().cpu().numpy()[0]
        output = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(output)
        )
        return output
