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

        # === FIX: THE ONE-SHOT EXAMPLE PROMPT ===
        # This new prompt is simpler and provides a perfect, concrete example of the desired output.
        # This is the most effective way to force the model to adhere to a specific JSON schema.
        tmpl = (
            "Generate an EXTREMELY DIFFICULT multiple-choice question on the topic: '{topic}'.\n\n"
            "Your entire response must be a single, valid JSON object. Follow this exact format:\n\n"
            "```json\n"
            "{{\n"
            '    "topic": "Puzzles/Seating Arrangements (Linear, Circular)",\n'
            '    "question": "Eight friends—A, B, C, D, E, F, G, and H—are sitting around a circular table, but not necessarily in that order. All are facing the center. F sits third to the left of C. There are two people between C and E. G is an immediate neighbor of A, who sits second to the right of E. B sits third to the right of H. Who sits exactly between A and B when counted from the left of A?",\n'
            '    "choices": [\n'
            '        "A) H",\n'
            '        "B) F",\n'
            '        "C) D",\n'
            '        "D) G"\n'
            '    ],\n'
            '    "answer": "C",\n'
            '    "explanation": "Based on the arrangement, the final order is E, G, A, D, B, C, F, H. Counting from the left of A, D is the only person sitting exactly between A and B."\n'
            "}}\n"
            "```\n\n"
            "Now, generate a new, unique, and EXTREMELY DIFFICULT question for the topic: '{topic}'"
        )
        # =======================================
        
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
        total_tokens = 0
        total_time = 0.0
        pbar = tqdm(total=(num_questions + batch_size - 1) // batch_size, desc="STEPS: ")

        for i in range(0, num_questions, batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            questions, tl, gt = self.generate_question_batch(
                batch_topics, wadvsys, **kwargs
            )
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
    gen_kwargs = {"tgps_show": True}
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
    if args.verbose:
        for output in raw_outputs:
            print(output, flush=True)
        if total_time > 0:
            print("\n" + "=" * 50)
            print(f"Total Time Taken: {total_time:.3f} seconds; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f} tokens/sec\n")
            print("=" * 50 + "\n")

    clean_questions = []
    for raw_text in raw_outputs:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                clean_questions.append(json.loads(json_str))
                print("Successfully extracted and parsed JSON.")
            except json.JSONDecodeError:
                print(f"  -> FAILED to parse extracted JSON from: {json_str}")
        else:
            print(f"  -> FAILED: No JSON object found in raw output.")
    
    agent.save_questions(clean_questions, args.output_file)
    filtered_file = args.output_file.replace("questions.json", "filtered_questions.json")
    filtered_questions = agent.filter_questions(clean_questions)
    agent.save_questions(filtered_questions, filtered_file)
    
    print(f"Saved {len(clean_questions)} clean questions to {args.output_file}!")
    print(f"Saved {len(filtered_questions)} filtered questions to {filtered_file}!")