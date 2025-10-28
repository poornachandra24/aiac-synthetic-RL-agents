import re
import json
import yaml
import random
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Our corrected model class is now the agent's core
from .question_model import QAgent


class QuestioningAgent(object):
    r"""Agent responsible for generating questions"""

    def __init__(self, **kwargs):
        self.agent = QAgent(**kwargs)

    # --- ORIGINAL METHODS - KEPT FOR COMPATIBILITY ---
    def build_inc_samples(self, inc_samples: List[Dict[str, str]], topic: str) -> str:
        # This function is no longer called by our prompt but is kept.
        return ""

    # --- METHODS UPDATED WITH NEW LOGIC ---
    def build_prompt(
        self,
        topic: str,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: List[Dict[str, str]] | None = None,
    ) -> Tuple[str, str]:
        
        if wadvsys:
            sys_prompt = """
            You are a JSON generation machine. Your sole purpose is to generate a single, valid JSON object that follows the user's instructions precisely.
            Do NOT add any conversational text, introductions, explanations, or conclusions before or after the JSON object.
            Your entire output must be ONLY the JSON object and nothing else.
            """
        else:
            sys_prompt = "You are an examiner tasked with creating extremely difficult multiple-choice questions"

        # Using our proven One-Shot Example prompt for conciseness and format adherence
        tmpl = (
            "Generate a tricky but CONCISE multiple-choice question on the topic: '{topic}'.\n\n"
            "CRITICAL RULES:\n"
            "1.  The question must be short and to the point.\n"
            "2.  The total length of the question, choices, and answer MUST be very short (well under 130 tokens).\n"
            "3.  Your entire response must be a single, valid JSON object following this exact format:\n\n"
            "```json\n"
            "{{\n"
            '    "topic": "Puzzles/Seating Arrangements (Linear, Circular)",\n'
            '    "question": "Eight friends sit around a circular table. F is third to the left of C. Two people are between C and E. G is a neighbor of A, who is second to the right of E. B is third to the right of H. Who sits between A and B (from A\'s left)?",\n'
            '    "choices": ["A) H", "B) F", "C) D", "D) G"],\n'
            '    "answer": "C",\n'
            '    "explanation": "The final order is E, G, A, D, B, C, F, H. From A\'s left, D sits between A and B."\n'
            "}}\n"
            "```\n\n"
            "Now, generate a new, unique, tricky, and CONCISE question for the topic: '{topic}'"
        )
        
        prompt = tmpl.format(topic=topic)
        return prompt, sys_prompt

    def generate_question(
        self,
        topic: List[Tuple[str, str]],
        wadvsys: bool,
        wicl: bool,
        inc_samples: Dict[str, List[Dict[str, str]]] | None,
        **gen_kwargs,
    ) -> Tuple[List[str], int | None, float | None]:
        
        prompts = []
        # A single system prompt is used for the entire batch for efficiency
        # The wicl and inc_samples arguments are ignored because our prompt doesn't use them,
        # but we accept them to maintain the original function signature.
        _, sp = self.build_prompt("", wadvsys, wicl, None) 
        for t in topic:
            p, _ = self.build_prompt(f"{t[0]}/{t[1]}", wadvsys, wicl, None)
            prompts.append(p)

        # Call the agent, which correctly expects a list of prompts
        resp, tl, gt = self.agent.generate_response(prompts, sp, **gen_kwargs)
        return resp, tl, gt

    def generate_batches(
        self,
        num_questions: int,
        topics: Dict[str, List[str]],
        batch_size: int = 5,
        wadvsys: bool = True,
        wicl: bool = True,
        inc_samples: Dict[str, List[Dict[str, str]]] | None = None,
        **kwargs,
    ) -> Tuple[List[str], List[int | None], List[float | None]]:
        
        extended_topics = self.populate_topics(topics, num_questions)
        all_questions = []
        all_tls, all_gts = [], []
        pbar = tqdm(total=(len(extended_topics) + batch_size - 1) // batch_size, desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i : i + batch_size]
            # Call the updated `generate_question` which handles batching correctly
            questions, tl, gt = self.generate_question(
                batch_topics, wadvsys, wicl, inc_samples, **kwargs
            )
            all_questions.extend(questions)
            all_tls.append(tl)
            all_gts.append(gt)
            pbar.update(1)
        pbar.close()
        # Return a list of strings and lists of token/time info, as expected
        return all_questions, all_tls, all_gts

    def count_tokens_q(self, text: str) -> int:
        return len(self.agent.tokenizer.encode(str(text), add_special_tokens=False))

    def filter_questions(
        self, questions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # This uses our final, corrected logic
        def basic_checks(q2: Dict[str, Any]) -> bool:
            required_keys = ["topic", "question", "choices", "answer"]
            if not all(key in q2 for key in required_keys): return False
            if not (isinstance(q2.get("choices"), list) and len(q2["choices"]) == 4): return False
            if not all(isinstance(c, str) and len(c) > 2 and c[0].upper() in "ABCD" for c in q2["choices"]): return False
            if not (isinstance(q2.get("answer"), str) and q2["answer"].upper() in "ABCD"): return False
            
            # The original token check from the hackathon guidelines
            check_len = sum(self.count_tokens_q(q2.get(k, "")) for k in ["question", "answer"])
            check_len += sum(self.count_tokens_q(c) for c in q2["choices"]) - 15
            if check_len >= 130: return False

            return True
        
        # This correctly handles a list of dicts now
        correct_format_question = [q for q in questions if basic_checks(q)]

        # The 50% threshold rule from the original script
        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return []

    def save_questions(self, questions: List[Dict], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(
        self, topics: Dict[str, List[str]], num_questions: int
    ) -> List[str]:
        all_subtopics = [(t, st) for t, sublist in topics.items() for st in sublist]
        return random.choices(all_subtopics, k=num_questions)

    @staticmethod
    def load_icl_samples(file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        with open(file_path, "r") as f:
            return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate questions using the QuestioningAgent.")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--output_file", type=str, default="outputs/questions.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # We still load these, even if wicl=False, to maintain the original structure
    inc_samples = QuestioningAgent.load_icl_samples("assets/topics_example.json")
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    # The main function call remains the same, but we now pass wicl=False
    raw_outputs, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=False, # Crucially, we disable the long in-context examples
        inc_samples=inc_samples,
        **gen_kwargs,
    )
    
    print(f"Generated {len(raw_outputs)} raw outputs!")
    if args.verbose and gts:
        total_time = sum(filter(None, gts))
        if total_time > 0:
            total_tokens = sum(filter(None, tls))
            print(f"\n{'='*50}\nTotal Time: {total_time:.3f}s; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f}\n{'='*50}\n")

    # The 3-Layer Parsing Cascade, now integrated into the original structure
    clean_questions = []
    print("\n--- Starting 3-Layer Robust JSON Parsing ---")
    for idx, raw_text in enumerate(raw_outputs, 1):
        print(f"[Q{idx}] ", end="")
        parsed_json = None
        
        # Layer 1: Try direct parsing
        try:
            parsed_json = json.loads(raw_text)
            print("✅ Layer 1 Success: Parsed clean JSON directly.")
        except json.JSONDecodeError:
            # Layer 2: Try regex extraction
            print("⚠️ Layer 1 Failed. Trying Layer 2 (Regex)... ", end="")
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    parsed_json = json.loads(match.group(0))
                    print("✅ Layer 2 Success!")
                except json.JSONDecodeError:
                    parsed_json = None
            
            # Layer 3: LLM Self-Correction
            if parsed_json is None:
                print("⚠️ Layer 2 Failed. Trying Layer 3 (LLM Self-Correction)... ", end="")
                try:
                    correction_prompt = f"Extract and return ONLY the valid JSON object from the following text:\n\n```\n{raw_text}\n```"
                    corrected_output, _, _ = agent.agent.generate_response(
                        [correction_prompt], "You are an expert JSON extractor.", temperature=0.0, do_sample=False
                    )
                    parsed_json = json.loads(corrected_output[0])
                    print("✅ Layer 3 Success!")
                except Exception as e:
                    print(f"❌ Layer 3 FAILED: {e}")
                    
        if parsed_json:
            clean_questions.append(parsed_json)
    
    print(f"\n{'='*50}")
    print(f"Successfully parsed {len(clean_questions)}/{len(raw_outputs)} outputs")
    print(f"{'='*50}")

    agent.save_questions(clean_questions, args.output_file)
    filtered_file = args.output_file.replace("questions.json", "filtered_questions.json")
    filtered_questions = agent.filter_questions(clean_questions)
    agent.save_questions(filtered_questions, filtered_file)
    
    print(f"\nSaved {len(clean_questions)} clean questions to {args.output_file}!")
    print(f"Saved {len(filtered_questions)} filtered questions to {filtered_file}!")