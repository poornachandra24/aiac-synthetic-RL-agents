import re
import json
import yaml
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .question_model import QAgent


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    def build_prompt(self, topic: str, wadvsys: bool = True) -> Tuple[str, str]:
        if wadvsys:
            sys_prompt = """
            You are a JSON generation machine. Your sole purpose is to generate a single, valid JSON object that follows the user's instructions precisely.
            Do NOT add any conversational text, introductions, explanations, or conclusions before or after the JSON object.
            Your entire output must be ONLY the JSON object and nothing else.
            """
        else:
            sys_prompt = "You are an examiner tasked with creating extremely difficult multiple-choice questions"

        tmpl = (
            "Generate a tricky but CONCISE multiple-choice question on the topic: '{topic}'.\n\n"
            "CRITICAL RULES:\n"
            "1.  The question must be short and to the point.\n"
            "2.  The total length of the question, choices, and answer MUST be very short (well under 130 tokens).\n"
            "3.  Your entire response must be a single, valid JSON object following this exact format:\n\n"
            "```json\n"
            "{{\n"
            '    "topic": "Puzzles/Seating Arrangements (Linear, Circular)",\n'
            '    "question": "Eight friends are sitting around a circular table. F sits third to the left of C. Two people are between C and E. G is a neighbor of A, who is second to the right of E. B is third to the right of H. Who sits between A and B (from A\'s left)?",\n'
            '    "choices": ["A) H", "B) F", "C) D", "D) G"],\n'
            '    "answer": "C",\n'
            '    "explanation": "The final order is E, G, A, D, B, C, F, H. Counting from the left of A, D sits between A and B."\n'
            "}}\n"
            "```\n\n"
            "Now, generate a new, unique, tricky, and CONCISE question for the topic: '{topic}'"
        )
        
        prompt = tmpl.format(topic=topic)
        return prompt, sys_prompt

    def generate_question_batch(
        self,
        topics: List[Tuple[str, str]],
        wadvsys: bool,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        prompts = []
        _, sp = self.build_prompt("", wadvsys)
        for t in topics:
            p, _ = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys)
            prompts.append(p)
        resp, tl, gt = self.agent.generate_response(prompts, sp, **gen_kwargs)
        return resp, tl, gt

    def generate_all_questions(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        **kwargs,
    ) -> Tuple[List[str], float, int]:
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        extended_topics = random.choices(all_subtopics, k=num_questions)
        all_questions = []
        total_tokens, total_time = 0, 0.0
        pbar = tqdm(total=(num_questions + batch_size - 1) // batch_size, desc="STEPS: ")
        for i in range(0, num_questions, batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            questions, tl, gt = self.generate_question_batch(batch_topics, wadvsys, **kwargs)
            all_questions.extend(questions)
            if tl: total_tokens += tl
            if gt: total_time += gt
            pbar.update(1)
        pbar.close()
        return all_questions, total_time, total_tokens

    def count_tokens_q(self, text: str) -> int:
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_questions(
        self, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        def basic_checks(q2: Dict[str, Any]) -> bool:
            required_keys = ["topic", "question", "choices", "answer"]
            if not all(key in q2 for key in required_keys): return False
            if not (isinstance(q2.get("choices"), list) and len(q2["choices"]) == 4): return False
            if not all(isinstance(c, str) and len(c) > 2 and c[0].upper() in "ABCD" for c in q2["choices"]): return False
            if not (isinstance(q2.get("answer"), str) and q2["answer"].upper() in "ABCD"): return False
            return True
        return [q for q in questions if basic_checks(q)]

    def save_questions(self, questions: List[Dict], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(questions, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="outputs/questions.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    gen_kwargs = {}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    raw_outputs, total_time, total_tokens = agent.generate_all_questions(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        **gen_kwargs,
    )
    
    print(f"Generated {len(raw_outputs)} raw outputs!")
    if args.verbose and total_time > 0:
        print(f"\n{'='*50}\nTotal Time: {total_time:.3f}s; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f}\n{'='*50}\n")

    # === FINAL, 3-LAYER PARSING CASCADE ===
    clean_questions = []
    print("--- Starting 3-Layer Robust JSON Parsing ---")
    for raw_text in raw_outputs:
        parsed_json = None
        
        # Layer 1: Try direct parsing (fastest)
        try:
            parsed_json = json.loads(raw_text)
            print("✅ Layer 1 Success: Parsed clean JSON directly.")
        except json.JSONDecodeError:
            # Layer 2: Try regex extraction (fast)
            print("⚠️ Layer 1 Failed. Trying Layer 2 (Regex)...")
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    parsed_json = json.loads(match.group(0))
                    print("✅ Layer 2 Success: Extracted JSON with regex.")
                except json.JSONDecodeError:
                    parsed_json = None
            
            # Layer 3: LLM Self-Correction (slow but smart)
            if parsed_json is None:
                print("⚠️ Layer 2 Failed. Trying Layer 3 (LLM Self-Correction)...")
                try:
                    correction_prompt = (
                        "The following text contains a JSON object, but it is formatted incorrectly or has extra text. "
                        "Your task is to extract and return ONLY the valid JSON object and nothing else.\n\n"
                        "Original Text:\n"
                        f"```\n{raw_text}\n```\n\n"
                        "Corrected JSON:"
                    )
                    # Note: We call agent.agent to access the inner model's generate_response
                    corrected_output, _, _ = agent.agent.generate_response(
                        correction_prompt,
                        "You are an expert JSON extractor.",
                        temperature=0.0, # Use greedy decoding for correction
                        do_sample=False,
                    )
                    parsed_json = json.loads(corrected_output)
                    print("✅ Layer 3 Success: Corrected JSON with second LLM call.")
                except Exception as e:
                    print(f"❌ Layer 3 FAILED: Could not self-correct. Error: {e}")
                    
        if parsed_json:
            clean_questions.append(parsed_json)

    agent.save_questions(clean_questions, args.output_file)
    filtered_file = args.output_file.replace("questions.json", "filtered_questions.json")
    filtered_questions = agent.filter_questions(clean_questions)
    agent.save_questions(filtered_questions, filtered_file)
    
    print(f"\nSaved {len(clean_questions)} clean questions to {args.output_file}!")
    print(f"Saved {len(filtered_questions)} filtered questions to {filtered_file}!")