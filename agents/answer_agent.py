import re
import json
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions with confidence scoring"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        # The select_prompt1 flag is kept for structural compatibility
        self.select_prompt1 = select_prompt1

    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        This method is now updated with our superior "JSON Machine" and "One-Shot" prompt
        to ensure clean, correctly formatted output from the fine-tuned model.
        """
        sys_prompt = """
        You are a JSON generation machine. Your sole purpose is to solve the following multiple-choice question and generate a single, valid JSON object that follows the specified format.
        Your final output must be ONLY the JSON object and nothing else.
        Do NOT add any conversational text, introductions, or conclusions.
        """
        
        tmpl = (
            "Solve the following question and provide the answer in the specified JSON format.\n\n"
            "Question: {question}\n"
            "Choices: {choices}\n\n"
            "RESPONSE FORMAT: Strictly generate a valid JSON object following this exact example format:\n"
            "```json\n"
            "{{\n"
            '    "answer": "A",\n'
            '    "reasoning": "This is a brief, step-by-step reasoning for why A is the correct answer, written within 50 words."\n'
            "}}\n"
            "```\n\n"
            "IMPORTANT: Your reasoning MUST be concise and you MUST include the closing brace `}}` to complete the JSON object."
        )
        
        choices_str = self._format_choices(question_data.get("choices", []))
        prompt = tmpl.format(question=question_data.get("question", "N/A"), choices=choices_str)
        
        # We return the same powerful prompt regardless of the select_prompt1 flag
        return prompt, sys_prompt

    def answer_question(
        self, question_data: List[Dict], **kwargs
    ) -> Tuple[List[str], int | None, float | None]:
        """
        This method is simplified to handle batching, as our model is optimized for it.
        It prepares a batch of prompts and gets a batch of responses.
        """
        prompts = []
        # A single system prompt is used for the entire batch for efficiency
        _, sp = self.build_prompt({})
        for qd in question_data:
            p, _ = self.build_prompt(qd)
            prompts.append(p)

        # This correctly calls our model, which expects a list of prompts
        resp, tl, gt = self.agent.generate_response(prompts, sp, **kwargs)
        return resp, tl, gt

    def answer_batches(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        """
        This is the original batching loop, now corrected to handle the tuple output
        from our high-performance model.
        """
        all_answers = []
        all_tls, all_gts = [], []
        pbar = tqdm(total=(len(questions) + batch_size - 1) // batch_size, desc="STEPS: ")

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            # `answer_question` now correctly handles batches
            batch_answers_text, tl, gt = self.answer_question(batch_questions, **kwargs)
            
            all_answers.extend(batch_answers_text)
            all_tls.append(tl)
            all_gts.append(gt)
            pbar.update(1)
        pbar.close()
        return all_answers, all_tls, all_gts

    # --- The following original methods are kept for compatibility and structure ---
    def count_tokens_a(self, text: str) -> int:
        return len(self.agent.tokenizer.encode(str(text), add_special_tokens=False))

    def filter_answers(self, ans: List[Dict[str, str]]) -> List[Dict[str, str] | None]:
        def basic_checks(a1: Dict[str, str]) -> bool:
            if "answer" in a1 and isinstance(a1["answer"], str):
                if len(a1["answer"]) == 1 and a1["answer"].upper() in "ABCD":
                    return True
            return False

        filtered = []
        for a in ans:
            if a and basic_checks(a):
                filtered.append(a)
            else:
                filtered.append(None) # Keep placeholders for scoring
        return filtered

    def save_answers(self, answers: List[Any], file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(answers, f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        return " ".join(choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Answering Agent")
    parser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json")
    parser.add_argument("--output_file", type=str, default="outputs/answers.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent()
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    # The main call to the original `answer_batches` method
    raw_outputs, tls, gts = agent.answer_batches(
        questions=sample_questions, batch_size=args.batch_size, **gen_kwargs
    )

    if args.verbose:
        # Verbose output now shows raw model text
        for i, (q, raw_a) in enumerate(zip(sample_questions, raw_outputs)):
            print(f"\n{'='*20} Question {i+1} {'='*20}")
            print(f"Expected: {q.get('answer', 'N/A')}, Model Raw Output:\n{raw_a}")
        if gts:
            total_time = sum(filter(None, gts))
            total_tokens = sum(filter(None, tls))
            if total_time > 0:
                print(f"\n{'='*50}\nTotal Time: {total_time:.3f}s; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f}\n{'='*50}\n")

    # === OUR INTEGRATED 3-LAYER PARSING CASCADE ===
    # This replaces the original script's slow self-correction loop with our robust, multi-layered solution.
    clean_answers = []
    print("\n--- Starting 3-Layer Robust JSON Parsing ---")
    for raw_text in raw_outputs:
        parsed_json = None
        # Layer 1: Direct Parsing
        try:
            parsed_json = json.loads(raw_text)
        except json.JSONDecodeError:
            # Layer 2: Regex Extraction
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try: parsed_json = json.loads(match.group(0))
                except json.JSONDecodeError: pass
            
            # Layer 3: LLM Self-Correction (Last Resort)
            if parsed_json is None:
                try:
                    correction_prompt = f"Extract and return ONLY the valid JSON object from the following text:\n\n```\n{raw_text}\n```"
                    corrected_output, _, _ = agent.agent.generate_response(
                        [correction_prompt], "You are an expert JSON extractor.", temperature=0.0, do_sample=False
                    )
                    parsed_json = json.loads(corrected_output[0])
                except Exception:
                    pass # Final failure

        # Final validation and cleanup
        if parsed_json and "answer" in parsed_json and isinstance(parsed_json.get("answer"), str):
            answer_match = re.search(r'([A-D])', parsed_json["answer"].upper())
            if answer_match:
                parsed_json["answer"] = answer_match.group(1)
                clean_answers.append(parsed_json)
            else:
                clean_answers.append(None) # Append None if answer format is invalid
        else:
            clean_answers.append(None) # Append None if parsing failed entirely

    # The final part of the script uses the original method names for saving and filtering
    agent.save_answers(clean_answers, args.output_file)
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    # We call the original filter function on our cleaned data
    final_filtered_answers = agent.filter_answers(clean_answers)
    agent.save_answers(final_filtered_answers, filtered_file_name)
    
    print(f"\nSaved {len([a for a in clean_answers if a])} clean answers to {args.output_file}!")
    print(f"Saved {len([a for a in final_filtered_answers if a])} filtered answers to {filtered_file_name}!")