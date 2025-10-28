# Qwen3-4B in action.
import time
import torch
from typing import Optional, List
from unsloth import FastLanguageModel

torch.random.manual_seed(0)


class AAgent(object):
    def __init__(self, **kwargs):
        model_name = "Thunderbird2410/Llama-3-8B-Puzzles-Unsloth"

        # load the tokenizer and the model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            dtype = torch.bfloat16,
            device_map = "auto",
        )

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        # Ensure message is a list for batch processing
        messages_list = [message] if isinstance(message, str) else message
        
        # Prepare all messages for batch processing using the correct chat template
        batched_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ] for msg in messages_list
        ]

        # Tokenize the entire batch at once with padding
        inputs = self.tokenizer.apply_chat_template(
            batched_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
            padding = True,
        ).to(self.model.device)

        # Separate kwargs for generation from other potential kwargs
        tgps_show_var = kwargs.pop("tgps_show", False)
        
        # Default generation parameters, can be overridden by kwargs
        generation_params = {
            "max_new_tokens": 1024,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generation_params.update(kwargs)

        # Generate responses for the batch
        if tgps_show_var:
            start_time = time.time()
        
        generated_ids = self.model.generate(inputs, **generation_params)
        
        if tgps_show_var:
            generation_time = time.time() - start_time
        
        # Decode the batch, ensuring to skip the original prompt tokens
        input_lengths = inputs.shape[1]
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, input_lengths:], 
            skip_special_tokens=True,
        )
        
        # Strip any leading/trailing whitespace from each response
        decoded_outputs = [output.strip() for output in decoded_outputs]

        if tgps_show_var:
            total_new_tokens = sum(len(ids[input_lengths:]) for ids in generated_ids)
            return (decoded_outputs[0] if isinstance(message, str) else decoded_outputs, total_new_tokens, generation_time)
        
        return (decoded_outputs[0] if isinstance(message, str) else decoded_outputs, None, None)


if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    response, tl, gt = ans_agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print(f"Single response: {response}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(
        messages,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        tgps_show=True,
    )
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", temperature=0.8, max_new_tokens=512
    )
    print(f"Custom response: {response}")
