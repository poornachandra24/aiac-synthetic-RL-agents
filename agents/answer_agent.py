import re
import json
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Dict, Any

from .answer_model import AAgent


class AnsweringAgent(object):
    r"""Hybrid Agent - Best of both worlds for answering MCQ questions"""

    def __init__(self, select_prompt1: bool = True, **kwargs):
        self.agent = AAgent(**kwargs)
        self.select_prompt1 = select_prompt1

    def build_prompt(self, question_data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Uses the superior "JSON Machine" prompt strategy for cleaner outputs
        """
        sys_prompt2 = """You are a JSON generation machine. Your sole purpose is to solve the following multiple-choice question and generate a single, valid JSON object that follows the specified format.
Your final output must be ONLY the JSON object and nothing else.
Do NOT add any conversational text, introductions, or conclusions."""

        sys_prompt1 = (
            "You are an expert answer agent specializing in solving multiple-choice questions (MCQs) that test "
            "quantitative aptitude skills, as seen in top-tier competitive exams. "
            "You have a deep understanding of logical reasoning, puzzles, and analytical problem-solving under exam conditions. "
            "For each question, think step by step using a clear chain-of-thought approach. "
            "Break down the problem, analyze all options, eliminate distractors, and then confidently select the correct answer. "
            "Always explain your reasoning before finalizing your choice. "
            "Your output must be ONLY a valid JSON object with no additional text."
        )

        tmpl = (
            "Solve the following question and provide the answer in the specified JSON format.\n\n"
            "Question: {question}\n"
            "Choices: {choices}\n\n"
            "RESPONSE FORMAT: Strictly generate a valid JSON object following this exact example format:\n"
            "```json\n"
            "{{\n"
            '    "answer": "A",\n'
            '    "reasoning": "This is a brief, step-by-step reasoning for why A is the correct answer, written within 75 words."\n'
            "}}\n"
            "```\n\n"
            "IMPORTANT: Your reasoning MUST be concise (max 75 words) and you MUST include the closing brace `}}` to complete the JSON object."
        )

        choices_str = self._format_choices(question_data.get("choices", []))
        prompt = tmpl.format(question=question_data.get("question", "N/A"),
                             choices=choices_str)

        return prompt, sys_prompt1 if self.select_prompt1 else sys_prompt2

    def answer_question(
            self, question_data: Dict | List[Dict],
            **kwargs) -> Tuple[List[Dict], int | None, float | None]:
        """Generate answer(s) for the given question(s)"""
        if isinstance(question_data, list):
            prompt = []
            for qd in question_data:
                p, sp = self.build_prompt(qd)
                prompt.append(p)
        else:
            prompt, sp = self.build_prompt(question_data)

        resp, tl, gt = self.agent.generate_response(prompt, sp, **kwargs)

        if (isinstance(resp, list) and all(isinstance(r, str)
                                           for r in resp)) or isinstance(
                                               resp, str):
            return resp, tl, gt
        else:
            return (
                "",
                tl,
                gt if not isinstance(resp, list) else [""] * len(resp),
                tl,
                gt,
            )

    def answer_batches(
            self,
            questions: List[Dict],
            batch_size: int = 5,
            **kwargs
    ) -> Tuple[List[Dict], List[int | None], List[float | None]]:
        """Answer questions in batches with proper remainder handling"""
        answers = []
        tls, gts = [], []
        total_batches = (len(questions) + batch_size - 1) // batch_size
        pbar = tqdm(total=total_batches, desc="STEPS: ", unit="batch")

        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers, tl, gt = self.answer_question(
                batch_questions, **kwargs)
            answers.extend(batch_answers)
            tls.append(tl)
            gts.append(gt)
            pbar.update(1)

        # Handle last batch with less than batch_size (CRITICAL: from original)
        if len(questions) % batch_size != 0:
            batch_questions = questions[-(len(questions) % batch_size):]
            batch_answers = self.answer_question(batch_questions, **kwargs)
            answers.extend(batch_answers[0])
            tls.append(batch_answers[1])
            gts.append(batch_answers[2])
            pbar.update(1)
        pbar.close()
        return answers, tls, gts

    def extract_single_letter(self, answer_text: str,
                              choices: List[str]) -> str:
        """Extract single letter (A-D) from answer text using multiple methods"""
        answer_text = answer_text.strip()

        # Method 1: Direct single letter check
        if len(answer_text) == 1 and answer_text.upper() in 'ABCD':
            return answer_text.upper()

        # Method 2: Pattern matching for common formats
        patterns = [
            r'^([A-D])\)',  # A), B), etc.
            r'^\(([A-D])\)',  # (A), (B), etc.
            r'^Option\s+([A-D])',  # Option A, Option B
            r'^Answer:\s*([A-D])',  # Answer: A
            r'^The\s+answer\s+is\s+([A-D])',  # The answer is A
            r'\b([A-D])\b(?:\)|\.|\s|$)',  # Any single letter A-D with boundary
        ]

        for pattern in patterns:
            match = re.search(pattern, answer_text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Method 3: Use LLM extraction if no clear letter found
        if len(answer_text) > 3:
            try:
                sys_prompt = "You are a precise option extractor. Extract ONLY the single letter (A, B, C, or D) from the answer. Return just the letter, nothing else."
                extraction_result = self.agent.generate_response(
                    self._option_extractor_prompt(answer_text, choices),
                    sys_prompt)[0]

                # Clean the extraction result
                if isinstance(extraction_result, str):
                    extraction_result = extraction_result.strip().upper()
                    if len(extraction_result
                           ) == 1 and extraction_result in 'ABCD':
                        return extraction_result
            except Exception as e:
                print(f"LLM extraction failed: {e}")

        # Fallback: mark as extraction failed
        return "X"

    def parse_json_cascade(self,
                           raw_text: str,
                           question_id: str = "N/A") -> Dict[str, str] | None:
        """
        3-Layer cascading JSON parser with question ID tracking
        Layer 1: Direct parsing
        Layer 2: Regex extraction
        Layer 3: LLM self-correction
        """
        parsed_json = None

        # Layer 1: Direct Parsing
        try:
            parsed_json = json.loads(raw_text)
            print(
                f"[Question {question_id}] ✓ Layer 1: Direct JSON parsing successful"
            )
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"[Question {question_id}] ✗ Layer 1 failed: {str(e)[:50]}")

        # Layer 2: Regex Extraction
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
                print(
                    f"[Question {question_id}] ✓ Layer 2: Regex extraction successful"
                )
                return parsed_json
            except json.JSONDecodeError as e:
                print(
                    f"[Question {question_id}] ✗ Layer 2 failed: {str(e)[:50]}"
                )

        # Layer 3: LLM Self-Correction (Last Resort)
        try:
            correction_prompt = (
                "Extract and return ONLY the valid JSON object from the following text. "
                "The JSON must have 'answer' and 'reasoning' keys.\n\n"
                f"```\n{raw_text}\n```")
            corrected_output, _, _ = self.agent.generate_response(
                [correction_prompt],
                "You are an expert JSON extractor.",
                temperature=0.0,
                do_sample=False)
            parsed_json = json.loads(corrected_output[0])
            print(
                f"[Question {question_id}] ✓ Layer 3: LLM self-correction successful"
            )
            return parsed_json
        except Exception as e:
            print(f"[Question {question_id}] ✗ Layer 3 failed: {str(e)[:50]}")

        print(
            f"[Question {question_id}] ✗✗✗ ALL LAYERS FAILED - Cannot parse JSON"
        )
        return None

    def _option_extractor_prompt(self, answer_string: str,
                                 choices: List[str]) -> str:
        """Generate prompt for extracting option letter from answer string"""
        tmpl = """You are an advanced LLM specialized in extracting the single correct option letter (A, B, C, D) from answers to multiple-choice questions. Your job is to parse the provided answer text, remove unnecessary prefixes, and identify which letter best matches the answer. If no direct letter is apparent, use textual comparison with the provided choices. If still no match is found, return "X".

EXTRACTION RULES:
1. Direct Letter Matching (e.g., "A", "(B)", or "The correct answer is C").
2. Prefix Removal (e.g., "The answer is", "Best option:", etc.).
3. Text Matching: If no letter is found, compare the answer text with the option contents.

OUTPUT FORMAT: Return only the correct option letter (A, B, C, or D), with no additional text.

Example 1:
Answer String: "The correct answer is B"
Choices: ["A) 21", "B) 44", "C) 15", "D) 68"]
Output: B

Example 2:
Answer String: "Photosynthesis happens in chloroplasts"
Choices: ["A) Chloroplasts are responsible for photosynthesis", "B) Photosynthesis occurs in mitochondria", "C) The photosynthetic process is responsible for oxygen release", "D) Light absorption enables this process"]
Output: A

Answer String: {}
Choices: {}
Output:"""
        return tmpl.format(answer_string, json.dumps(choices))

    def count_tokens_a(self, text: str) -> int:
        """Count the number of tokens in the text using the agent's tokenizer"""
        if not hasattr(self.agent, "tokenizer"):
            raise AttributeError(
                "The agent does not have a tokenizer attribute.")
        return len(self.agent.tokenizer.encode(text, add_special_tokens=False))

    def filter_answers(
            self, ans: List[str | Dict[str, str]]) -> List[Dict[str, str]]:
        r"""Filter answers to ensure they are in the correct format (comprehensive validation)"""

        def basic_checks(a1: Dict[str, str]) -> bool:
            # check required keys
            required_keys = ["answer"]
            if all((key in a1) and isinstance(a1[key], str)
                   for key in required_keys):
                if len(a1["answer"]) == 1 and (a1["answer"] not in "ABCDabcd"):
                    return False
                check_len = self.count_tokens_a(a1["answer"])
                if check_len < 50:
                    check_len += self.count_tokens_a(
                        a1.get("reasoning", "None"))
                    if check_len < 512:
                        if len(a1['answer']) == 1 and a1['answer'].upper() in 'ABCD':
                            return True
            return False

        filtered_answers = []
        for i, a in enumerate(ans):
            if isinstance(a, dict):
                if basic_checks(a):
                    filtered_answers.append(a)
                else:
                    filtered_answers.append(None)
                    print(f"Skipping invalid answer at index {i}: {a}")
            elif isinstance(a, str):
                # Basic checks: at least with correct JSON format
                try:
                    a1 = json.loads(a)
                    if basic_checks(a1):
                        filtered_answers.append(a1)
                    else:
                        filtered_answers.append(None)
                        print(f"Skipping invalid answer at index {i}: {a}")
                except json.JSONDecodeError:
                    # If JSON decoding fails, skip this answer
                    print(f"Skipping invalid JSON at index {i}: {a}")
                    filtered_answers.append(None)
                    continue
            else:
                # If the answer is neither a dict nor a str, skip it
                print(f"Skipping unsupported type at index {i}: {type(a)}")
                filtered_answers.append(None)
        return filtered_answers

    def save_answers(self, answers: List[str], file_path: str | Path) -> None:
        """Save generated answers to a JSON file"""
        # check for existence of dir
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump([a for a in answers], f, indent=4)

    def _format_choices(self, choices: List[str]) -> str:
        r"""Format the choices for better readability"""
        formatted = []
        for choice in choices:
            # Ensure each choice starts with a letter if not already formatted
            if not re.match(r"^[A-D]\)", choice.strip()):
                # Extract letter from existing format or assign based on position
                letter = chr(65 + len(formatted))  # A, B, C, D
                formatted.append(f"{letter}) {choice.strip()}")
            else:
                formatted.append(choice.strip())
        return " ".join(formatted)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Hybrid Answering Agent")
    parser.add_argument(
        "--input_file",
        type=str,
        default="outputs/filtered_questions.json",
        help="Path to the input JSON file with questions",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/answers.json",
        help="Path to save the answers",
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=5,
                        help="Batch size for processing questions")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose output")
    args = parser.parse_args()

    SELECT_PROMPT1 = True  # Use the JSON Machine prompt

    # Load sample questions
    with open(args.input_file, "r") as f:
        sample_questions = json.load(f)

    agent = AnsweringAgent(select_prompt1=SELECT_PROMPT1)

    gen_kwargs = {"tgps_show": True}
    with open("agen.yaml", "r") as f:
        gen_kwargs.update(yaml.safe_load(f))

    # Generate answers in batches
    print("\n" + "=" * 60)
    print("STARTING BATCH ANSWER GENERATION")
    print("=" * 60 + "\n")

    raw_outputs, tls, gts = agent.answer_batches(questions=sample_questions,
                                                 batch_size=args.batch_size,
                                                 **gen_kwargs)

    # === HYBRID 3-LAYER PARSING CASCADE WITH ROBUST EXTRACTION ===
    print("\n" + "=" * 60)
    print("STARTING 3-LAYER JSON PARSING CASCADE")
    print("=" * 60 + "\n")

    clean_answers = []
    for idx, (q, raw_text) in enumerate(zip(sample_questions, raw_outputs)):
        question_id = q.get("id", q.get("question_id", idx + 1))

        if args.verbose:
            print(f"\n{'='*20} Question {question_id} {'='*20}")
            print(f"Question: {q.get('question', 'N/A')}...")
            print(f"Expected: {q.get('answer', 'N/A')}")
            print(f"Raw Output: {raw_text[:200]}...")

        # Parse JSON using 3-layer cascade
        parsed_json = agent.parse_json_cascade(raw_text, str(question_id))

        if parsed_json and "answer" in parsed_json:
            # Extract and clean the answer letter
            answer_text = parsed_json["answer"].strip()
            extracted_letter = agent.extract_single_letter(
                answer_text, q.get("choices", []))

            if extracted_letter and extracted_letter != "X":
                parsed_json["answer"] = extracted_letter
                clean_answers.append(parsed_json)
                print(
                    f"[Question {question_id}] ✓ Final Answer: {extracted_letter}"
                )
            else:
                print(
                    f"[Question {question_id}] ✗ Failed to extract valid option letter"
                )
                parsed_json["answer"] = "X"
                clean_answers.append(None)
        else:
            print(f"[Question {question_id}] ✗ Failed to parse JSON")
            clean_answers.append(None)

        if args.verbose:
            print(f"{'='*50}\n")

    # === METRICS AND STATISTICS (PRESERVED FROM ORIGINAL) ===
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)

    if gen_kwargs.get("tgps_show", False) and gts:
        for idx, (tl, gt) in enumerate(zip(tls, gts)):
            if tl is not None and gt is not None:
                print(
                    f"BATCH {idx + 1}: Tokens={tl}, Time={gt:.3f}s, TGPS={tl/gt:.3f}"
                )

        # Total metrics
        total_tls = [tl for tl in tls if tl is not None]
        total_gts = [gt for gt in gts if gt is not None]

        if total_gts:
            total_time = sum(total_gts)
            total_tokens = sum(total_tls)
            avg_tgps = total_tokens / total_time if total_time > 0 else 0

            print("\n" + "=" * 60)
            print(f"TOTAL TIME: {total_time:.3f} seconds")
            print(f"TOTAL TOKENS: {total_tokens}")
            print(f"AVERAGE TGPS: {avg_tgps:.3f} tokens/second")
            print("=" * 60 + "\n")

    # === SAVE RESULTS ===
    successful_answers = len([a for a in clean_answers if a])
    total_questions = len(sample_questions)

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Total Questions: {total_questions}")
    print(f"   Successfully Parsed: {successful_answers}")
    print(f"   Failed: {total_questions - successful_answers}")
    print(
        f"   Success Rate: {(successful_answers/total_questions)*100:.2f}%\n")

    # Save raw answers
    agent.save_answers(clean_answers, args.output_file)
    print(f"✓ Saved {successful_answers} answers to {args.output_file}")

    # Filter and save filtered answers
    filtered_file_name = args.output_file.replace("answers.json", "filtered_answers.json")
    final_filtered_answers = agent.filter_answers(clean_answers)
    agent.save_answers(final_filtered_answers, filtered_file_name)

    filtered_count = len([a for a in final_filtered_answers if a])
    print(f"✓ Saved {filtered_count} filtered answers to {filtered_file_name}")
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60 + "\n")
