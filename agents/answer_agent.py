import re
import json
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

# This correctly imports the Unsloth-powered AAgent from your answer_model.py
from .answer_model import AAgent


def robust_json_parser(raw_text: str) -> dict | None:
    """Robustly extract and parse JSON from model output."""
    # Strategy 1: Try to find JSON between code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Find complete JSON objects (non-greedy)
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(json_pattern, raw_text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match.group(0))
            if "answer" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Extract from first { to last }
    try:
        first_brace = raw_text.find('{')
        if first_brace != -1:
            last_brace = raw_text.rfind('}')
            if last_brace != -1:
                json_str = raw_text[first_brace:last_brace+1]
                return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        pass
    
    return None


def clean_and_validate_answer(parsed_json: dict) -> dict | None:
    """Clean and validate the parsed JSON answer."""
    if not isinstance(parsed_json, dict):
        return None
    
    if "answer" not in parsed_json:
        print(f"❌ 'answer' key missing")
        return None
    
    answer_value = parsed_json.get("answer", "")
    if not isinstance(answer_value, str):
        print(f"❌ 'answer' is not a string")
        return None
    
    # Extract valid answer letter (A-D)
    answer_match = re.search(r'([A-D])', answer_value.upper())
    if not answer_match:
        print(f"❌ No valid answer letter (A-D) in: {answer_value}")
        return None
    
    cleaned = {
        "answer": answer_match.group(1),
        "reasoning": parsed_json.get("reasoning", "").strip()
    }
    
    # Handle empty reasoning
    if not cleaned["reasoning"]:
        cleaned["reasoning"] = "Answer selected based on analysis of the question."
        print(f"⚠️  Empty reasoning, using default")
    
    return cleaned


class AnsweringAgent(object):
    r"""Agent responsible for answering MCQ questions"""

    def __init__(self, **kwargs):
        self.agent = AAgent(**kwargs)

    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        # A forceful "JSON machine" prompt with a one-shot example
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
            "IMPORTANT: You MUST provide non-empty reasoning. Do NOT leave the reasoning field empty."
        )
        
        choices_str = " ".join(question_data.get("choices", []))
        prompt = tmpl.format(question=question_data.get("question", "N/A"), choices=choices_str)
        return prompt, sys_prompt

    def answer_question_batch(
        self, questions: List[Dict], **kwargs
    ) -> Tuple[List[str], int | None, float | None]:
        prompts = []
        _, sp = self.build_prompt({})
        for qd in questions:
            p, _ = self.build_prompt(qd)
            prompts.append(p)

        resp, tl, gt = self.agent.generate_response(prompts, sp, **kwargs)
        return resp, tl, gt

    def answer_all_questions(
        self, questions: List[Dict], batch_size: int = 5, **kwargs
    ) -> Tuple[List[str], float, int]:
        all_answers = []
        total_tokens = 0
        total_time = 0.0
        pbar = tqdm(total=(len(questions) + batch_size - 1) // batch_size, desc="STEPS: ")

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i : i + batch_size]
            answers, tl, gt = self.answer_question_batch(batch_questions, **kwargs)
            all_answers.extend(answers)
            if tl: total_tokens += tl
            if gt: total_time += gt
            pbar.update(1)
        pbar.close()
        return all_answers, total_time, total_tokens

    def save_answers(self, answers: List[Dict], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(answers, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Answering Agent")
    parser.add_argument("--input_file", type=str, default="outputs/filtered_questions.json")
    parser.add_argument("--output_file", type=str, default="outputs/answers.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        questions_to_answer = json.load(f)

    agent = AnsweringAgent()
    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    raw_outputs, total_time, total_tokens = agent.answer_all_questions(
        questions=questions_to_answer, batch_size=args.batch_size, **gen_kwargs
    )

    if args.verbose:
        for i, (q, raw_a) in enumerate(zip(questions_to_answer, raw_outputs)):
            print("\n" + "="*20 + f" Question {i+1} " + "="*20)
            print(f"Question: {q.get('question', 'N/A')}")
            print(f"Expected Answer: {q.get('answer', 'N/A')}")
            print(f"Model Raw Output:\n{raw_a}")
        
        if total_time > 0:
            print("\n" + "=" * 50)
            print(f"Total Time Taken: {total_time:.3f} seconds; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f} tokens/sec\n")
            print("=" * 50 + "\n")

    # === IMPROVED ROBUST JSON PARSING ===
    clean_answers = []
    print("\n--- Starting Robust JSON Parsing ---")
    
    for idx, raw_text in enumerate(raw_outputs, 1):
        print(f"[Q{idx}] ", end="")
        
        # Step 1: Extract JSON
        parsed_json = robust_json_parser(raw_text)
        
        if parsed_json is None:
            print(f"❌ No valid JSON found")
            continue
        
        # Step 2: Clean and validate
        cleaned = clean_and_validate_answer(parsed_json)
        
        if cleaned:
            clean_answers.append(cleaned)
            print(f"✅ Answer: {cleaned['answer']}")
    # =====================================
    
    print(f"\n{'='*50}")
    print(f"Successfully parsed {len(clean_answers)}/{len(raw_outputs)} outputs")
    print(f"{'='*50}")
    
    agent.save_answers(clean_answers, args.output_file)
    print(f"\nSaved {len(clean_answers)} clean answers to {args.output_file}!")