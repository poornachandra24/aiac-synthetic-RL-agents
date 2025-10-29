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
    def build_inc_samples(self, inc_samples: List[Dict[str, str]],
                          topic: str) -> str:
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
            "1.  The question must be short and sensible.\n"
            "2.  The total length of the question, choices, and answer MUST be very short (well under 130 tokens).\n"
            "3.  Your entire response must be **ONLY** a single, valid JSON object following this exact format:\n\n"
            "```json\n"
            "{{\n"
            '    "topic": "Puzzles/Seating Arrangements (Linear, Circular)",\n'
            '    "question": "Eight friends sit around a circular table. F is third to the left of C. Two people are between C and E. G is a neighbor of A, who is second to the right of E. B is third to the right of H. Who sits between A and B (from A\'s left)?",\n'
            '    "choices": ["A) H", "B) F", "C) D", "D) G"],\n'
            '    "answer": "C",\n'
            '    "explanation": "The final order is E, G, A, D, B, C, F, H. From A\'s left, D sits between A and B."\n'
            "}}\n"
            "```\n\n"
            "Now, generate a new, unique, tricky, and CONCISE question for the topic: '{topic} in the expected schema ONLY'"
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
        pbar = tqdm(total=(len(extended_topics) + batch_size - 1) //
                    batch_size,
                    desc="STEPS: ")

        for i in range(0, len(extended_topics), batch_size):
            batch_topics = extended_topics[i:i + batch_size]
            # Call the updated `generate_question` which handles batching correctly
            questions, tl, gt = self.generate_question(batch_topics, wadvsys,
                                                       wicl, inc_samples,
                                                       **kwargs)
            all_questions.extend(questions)
            all_tls.append(tl)
            all_gts.append(gt)
            pbar.update(1)
        pbar.close()
        # Return a list of strings and lists of token/time info, as expected
        return all_questions, all_tls, all_gts

    def count_tokens_q(self, text: str) -> int:
        return len(
            self.agent.tokenizer.encode(str(text), add_special_tokens=False))

    def filter_questions(
            self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        def basic_checks(q2: Dict[str, Any]) -> bool:
            # check required keys
            required_keys = ["topic", "question", "choices", "answer"]
            if all((key in q2) for key in required_keys):
                # check choices format
                checks = all(
                    isinstance(choice, str) and len(choice) > 2
                    and choice[0].upper() in "ABCD"
                    for choice in q2["choices"])
                if (isinstance(q2["choices"], list) and len(q2["choices"]) == 4
                        and checks):
                    # check answer format
                    if isinstance(q2["answer"],
                                  str) and q2["answer"].upper() in "ABCD":
                        # Check token length
                        check_len = sum(
                            self.count_tokens_q(q2[k])
                            for k in ["question", "answer"])
                        check_len += (sum(
                            self.count_tokens_q(choice)
                            for choice in q2["choices"]) - 15)
                        if check_len < 130:
                            if (check_len + self.count_tokens_q(
                                    q2.get("explanation", "None")) <= 1024):
                                return True
            return False

        correct_format_question = []
        for i, q in enumerate(questions):
            if isinstance(q, dict):
                if basic_checks(q):
                    correct_format_question.append(q)
            else:
                continue

        if len(correct_format_question) >= 0.5 * len(questions):
            return correct_format_question
        return list()

    def save_questions(self, questions: List[Dict], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(questions, f, indent=4)

    def populate_topics(self, topics: Dict[str, List[str]],
                        num_questions: int) -> List[str]:
        all_subtopics = [(t, st) for t, sublist in topics.items()
                         for st in sublist]
        return random.choices(all_subtopics, k=num_questions)

    @staticmethod
    def load_icl_samples(
            file_path: str | Path) -> Dict[str, List[Dict[str, str]]]:
        with open(file_path, "r") as f:
            return json.load(f)

    def extract_json_from_text_batch(
            self,
            texts: List[str],
            batch_size: int = 10) -> List[Dict[str, Any] | None]:
        """Batch process JSON extraction with improved LLM correction"""

        results = [None] * len(texts)
        remaining_indices = list(range(len(texts)))

        # Layer 1: Direct parsing
        new_remaining = []
        for idx in remaining_indices:
            try:
                results[idx] = json.loads(texts[idx])
                print(
                    f"[Q{idx+1}] ✅ Layer 1 Success: Parsed clean JSON directly."
                )
            except json.JSONDecodeError:
                new_remaining.append(idx)
        remaining_indices = new_remaining

        # Layer 2: Improved regex extraction for remaining
        new_remaining = []
        for idx in remaining_indices:
            parsed = self._try_regex_extraction(texts[idx])
            if parsed:
                results[idx] = parsed
                print(f"[Q{idx+1}] ✅ Layer 2 Success!")
            else:
                new_remaining.append(idx)
                print(f"[Q{idx+1}] ⚠️ Layer 2 Failed.")

        remaining_indices = new_remaining

        # Layer 3: Improved batch LLM correction
        if remaining_indices:
            print(
                f"\n🔧 Batch correcting {len(remaining_indices)} failed questions with LLM (batch_size={batch_size})..."
            )
            self._batch_llm_correction(texts, results, remaining_indices, 1 ,batch_size)

        return results

    def _try_regex_extraction(self, text: str) -> Dict[str, Any] | None:
        """Improved regex extraction with multiple patterns and cleaning"""
        # First, try to extract JSON with various patterns
        json_patterns = [
            r'\{[^{}]*\{[^{}]*\}[^{}]*\}',  # Nested objects
            r'\{.*\}',  # Simple object
            r'\[\{.*\}\]',  # Array of objects
        ]

        for pattern in json_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    json_str = match.group(0)
                    # Clean common issues before parsing
                    json_str = self._clean_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        return None

    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues"""
        # Remove trailing commas
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        # Fix single quotes to double quotes
        json_str = re.sub(r"'([^']*)'", r'"\1"', json_str)
        # Remove extra whitespace
        json_str = re.sub(r'\s+', ' ', json_str)
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)
        return json_str.strip()

    def _batch_llm_correction(self, texts: List[str], results: List,
                              indices: List[int], max_attempts: int,
                              batch_size: int):
        """Batch process LLM correction for multiple failed items"""

        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch_texts = [texts[idx] for idx in batch_indices]

            # Preprocess texts to remove conversational wrappers
            preprocessed_texts = [
                self._remove_conversational_wrappers(text)
                for text in batch_texts
            ]

            correction_prompt_template = """
                EXTRACT AND RETURN ONLY VALID JSON from the text below. Follow these rules STRICTLY:
                
                1. Return ONLY the JSON object, no other text
                2. Ensure proper JSON format with double quotes
                3. If multiple JSON objects exist, pick the most complete one
                4. If no valid JSON, return {{}}
                
                TEXT:
                {text}
                
                JSON OUTPUT:
                """

            batch_prompts = [
                correction_prompt_template.format(text=text)
                for text in preprocessed_texts
            ]

            try:
                # Use more conservative generation parameters
                corrected_outputs, _, _ = self.agent.generate_response(
                    batch_prompts,
                    "You are a JSON extraction expert. Return ONLY valid JSON, no explanations.",
                    temperature=0.0,  # More deterministic
                    do_sample=False,
                    max_new_tokens=300,
                    pad_token_id=self.agent.tokenizer.eos_token_id)

                for i, (original_idx, corrected_text) in enumerate(
                        zip(batch_indices, corrected_outputs)):
                    parsed_json = self._parse_and_validate_corrected_output(
                        corrected_text)
                    if parsed_json:
                        results[original_idx] = parsed_json
                        print(f"[Q{original_idx+1}] ✅ Layer 3 Success!")
                    else:
                        # Try one more time with more aggressive cleaning
                        retry_parsed = self._aggressive_json_recovery(
                            batch_texts[i])
                        if retry_parsed:
                            results[original_idx] = retry_parsed
                            print(
                                f"[Q{original_idx+1}] ✅ Layer 3 Success (after retry)!"
                            )
                        else:
                            print(f"[Q{original_idx+1}] ❌ Layer 3 FAILED")

            except Exception as e:
                print(f"❌ Batch correction failed: {e}")
                # Try individual processing for this batch
                for idx in batch_indices:
                    individual_result = self._individual_json_recovery(
                        texts[idx])
                    if individual_result:
                        results[idx] = individual_result
                        print(
                            f"[Q{idx+1}] ✅ Layer 3 Success (individual recovery)!"
                        )
                    else:
                        print(f"[Q{idx+1}] ❌ Layer 3 FAILED (batch error)")
                        

    def _remove_conversational_wrappers(self, text: str) -> str:
        """Remove common conversational wrappers around JSON"""
        # Remove common prefixes
        prefixes = [
            "Here is", "Here's", "Here is your", "Generated JSON:",
            "JSON output:", "The JSON is:", "```json", "```"
        ]

        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()

        # Remove common suffixes
        suffixes = ["```", "End of JSON", "That's it", "This is the JSON"]
        for suffix in suffixes:
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()

        return text

    def _parse_and_validate_corrected_output(
            self, corrected_text: str) -> Dict[str, Any] | None:
        """Parse and validate LLM-corrected output with better validation"""
        try:
            # Clean the output
            cleaned_text = corrected_text.strip()
            cleaned_text = re.sub(r'^```json\s*', '', cleaned_text)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            cleaned_text = cleaned_text.strip()

            # Skip empty objects
            if not cleaned_text or cleaned_text == "{}":
                return None

            # Try to parse
            parsed = json.loads(cleaned_text)

            # Validate it's a proper question object
            if (isinstance(parsed, dict) and parsed.get("question")
                    and parsed.get("choices") and parsed.get("answer")):
                return parsed

        except (json.JSONDecodeError, KeyError, IndexError):
            return None

        return None

    def _aggressive_json_recovery(self, text: str) -> Dict[str, Any] | None:
        """More aggressive JSON recovery for difficult cases"""
        try:
            # Try to find anything that looks like structured data
            patterns = [
                r'{\s*"topic"[^}]+"question"[^}]+"choices"[^}]+"answer"[^}]*}',
                r'{\s*"question"[^}]+"choices"[^}]+"answer"[^}]*}',
                r'{\s*[\w"][^}]*}',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    try:
                        # Clean and try to parse
                        cleaned = self._clean_json_string(match)
                        parsed = json.loads(cleaned)
                        if isinstance(parsed, dict) and parsed:
                            return parsed
                    except:
                        continue
        except:
            pass

        return None

    def _individual_json_recovery(self, text: str) -> Dict[str, Any] | None:
        """Individual recovery for stubborn cases"""
        # Try to manually construct from obvious patterns
        try:
            # Look for question pattern
            question_match = re.search(r'"question"\s*:\s*"([^"]+)"', text)
            choices_match = re.search(r'"choices"\s*:\s*\[([^\]]+)\]', text)
            answer_match = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
            topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', text)

            if question_match and choices_match and answer_match:
                # Try to build a basic question object
                question_obj = {
                    "topic":
                    topic_match.group(1) if topic_match else "Unknown",
                    "question": question_match.group(1),
                    "choices":
                    self._parse_choices_string(choices_match.group(1)),
                    "answer": answer_match.group(1)
                }
                return question_obj
        except:
            pass

        return None

    def _parse_choices_string(self, choices_str: str) -> List[str]:
        """Parse choices from string representation"""
        try:
            # Try to parse as JSON first
            return json.loads(f"[{choices_str}]")
        except:
            # Fallback: split by commas and clean
            choices = re.findall(r'["\']([^"\']+)["\']', choices_str)
            if choices:
                return choices
            # Last resort: split by commas
            return [
                choice.strip() for choice in choices_str.split(',')
                if choice.strip()
            ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate questions using the QuestioningAgent.")
    parser.add_argument("--num_questions", type=int, default=10)
    parser.add_argument("--output_file",
                        type=str,
                        default="outputs/questions.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # We still load these, even if wicl=False, to maintain the original structure
    inc_samples = QuestioningAgent.load_icl_samples(
        "assets/topics_example.json")
    with open("assets/topics.json") as f:
        topics = json.load(f)

    agent = QuestioningAgent()
    gen_kwargs = {"tgps_show": True}
    with open("qgen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    # The main function call remains the same, but we now pass wicl=False
    question, tls, gts = agent.generate_batches(
        num_questions=args.num_questions,
        topics=topics,
        batch_size=args.batch_size,
        wadvsys=True,
        wicl=False,  # Crucially, we disable the long in-context examples
        inc_samples=inc_samples,
        **gen_kwargs,
    )

    print(f"Generated {len(question)} raw questions!")
    if args.verbose and gts:
        for i, q in enumerate(question):
            print(f"Question ID: {i+1}", flush=True)
            print(q, flush=True)
            print("\n" + "=" * 50 + "\n\n")
        total_time = sum(filter(None, gts))
        if total_time > 0:
            total_tokens = sum(filter(None, tls))
            print(
                f"\n{'='*50}\nTotal Time: {total_time:.3f}s; Total Tokens: {total_tokens}; TGPS: {total_tokens/total_time:.3f}\n{'='*50}\n"
            )

    # The 3-Layer Parsing Cascade, now integrated into the original structure
    # The 3-Layer Batch Parsing Cascade
    print("\n--- Starting 3-Layer Robust JSON Parsing ---")


    # Batch process all questions at once
    parsed_results = agent.extract_json_from_text_batch(
        question, batch_size=args.batch_size)

    # Filter out None results and collect successful parses
    clean_questions = [
        result for result in parsed_results if result is not None
    ]
    failed_count = len(question) - len(clean_questions)
    print(f"\n{'='*50}")
    print(
        f"Successfully parsed {len(clean_questions)}/{len(question)} outputs")
    print(f"Failed: {failed_count} questions")
    print(f"{'='*50}")

    agent.save_questions(clean_questions, args.output_file)
    filtered_file = args.output_file.replace("questions.json",
                                             "filtered_questions.json")
    filtered_questions = agent.filter_questions(clean_questions)
    agent.save_questions(filtered_questions, filtered_file)

    print(
        f"\nSaved {len(clean_questions)} clean questions to {args.output_file}!"
    )
    print(
        f"Saved {len(filtered_questions)} filtered questions to {filtered_file}!"
    )
