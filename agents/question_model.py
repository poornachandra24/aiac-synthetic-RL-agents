import time
import torch
from typing import Optional, List
from unsloth import FastLanguageModel

torch.random.manual_seed(0)


class QAgent(object):
    def __init__(self, **kwargs):
        model_name = "Thunderbird2410/Llama-3-8B-Puzzles-Unsloth"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length=4096,
            dtype = torch.bfloat16,
            device_map = "auto",
        )
        self.tokenizer.padding_side = "left"
        # Add a pad token if it's missing, which is common for Llama models
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            self.model.resize_token_embeddings(len(self.tokenizer))


    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        
        messages_list = [message] if isinstance(message, str) else message
        
        # === SYNTAX FIX IS HERE ===
        # The generator expression is now correctly wrapped in brackets [] to form a list comprehension.
        batched_messages = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg}
            ] for msg in messages_list
        ]
        # ==========================

        inputs = self.tokenizer.apply_chat_template(
            batched_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
            padding = True,
        ).to(self.model.device)

        tgps_show_var = kwargs.pop("tgps_show", False)
        
        generation_params = {
            "max_new_tokens": 1024,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        generation_params.update(kwargs)

        if tgps_show_var:
            start_time = time.time()
        
        generated_ids = self.model.generate(inputs, **generation_params)
        
        if tgps_show_var:
            generation_time = time.time() - start_time
        
        input_lengths = inputs.shape[1]
        decoded_outputs = self.tokenizer.batch_decode(
            generated_ids[:, input_lengths:], 
            skip_special_tokens=True,
        )
        
        decoded_outputs = [output.strip() for output in decoded_outputs]

        if tgps_show_var:
            total_new_tokens = sum(len(ids[input_lengths:]) for ids in generated_ids)
            return (decoded_outputs[0] if isinstance(message, str) else decoded_outputs, total_new_tokens, generation_time)
        
        return (decoded_outputs[0] if isinstance(message, str) else decoded_outputs, None, None)



if __name__ == "__main__":
    # Single example generation
    model = QAgent()
    prompt = f"""
    Question: Generate a hard MCQ based question as well as their 4 choices and its answers on the topic, Number Series.
    Return your response as a valid JSON object with this exact structure:

        {{
            "topic": Your Topic,
            "question": "Your question here ending with a question mark?",
            "choices": [
                "A) First option",
                "B) Second option", 
                "C) Third option",
                "D) Fourth option"
            ],
            "answer": "A",
            "explanation": "Brief explanation of why the correct answer is right and why distractors are wrong"
        }}
    """

    response, tl, tm = model.generate_response(
        prompt,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("Single example response:")
    print("Response: ", response)
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
    print("+-------------------------------------------------\n\n")

    # Multi example generation
    prompts = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, tm = model.generate_response(
        prompts,
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print("\nMulti example responses:")
    for i, resp in enumerate(responses):
        print(f"Response {i+1}: {resp}")
    print(
        f"Total tokens: {tl}, Time taken: {tm:.2f} seconds, TGPS: {tl/tm:.2f} tokens/sec"
    )
