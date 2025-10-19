"""Sample script that asks OpenAI GPT-4o-mini to solve simple math/graph problems.

Updates (2025-10-01):
    - Do not print the model output to stdout; save it to a text file with a random filename instead.
    - Only print the saved file path to stdout (the answer itself remains hidden).

Usage:
    1. Set your API key via the OPENAI_API_KEY environment variable:
             export OPENAI_API_KEY=sk-xxxx
    2. Example:
             python tools/openai_api.py --question "Solve for x in 2x + 3 = 11"

    If the question is omitted, the default calculation is (12 * (7 - 2)).

Dependencies (openai is already listed in environment.yml):
        conda env update -f environment.yml

Notes:
    - Never commit your API key to the repository.
    - Retries requests with an exponential-backoff-like approach when network or rate limits occur.
    - Automatically saves answers under tools/outputs/answers/ (creates the directory if missing).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI  # New client introduced in openai>=1.0.0


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solve simple math problems with GPT-4o-mini")
    p.add_argument(
        "--subset",
        type=str,
        choices=["edge_count", "triangle_counting", "cycle_check"],
        default="edge_count",
        help="The subset of GraphQA to solve",
    )
    return p.parse_args()


def get_prompt(subset: str) -> str:
    prompt_data = {
        "edge_count": (
            "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
            "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, and 17. "
            "The edges in G are: (0, 1) (0, 2) (0, 3) (0, 6) (0, 7) (0, 11) (0, 12) (0, 13) (0, 14) (0, 15) "
            "(0, 16) (0, 17) (1, 2) (1, 5) (1, 6) (1, 7) (1, 8) (1, 10) (1, 11) (1, 13) (1, 14) (1, 16) (2, 3) "
            "(2, 4) (2, 5) (2, 6) (2, 8) (2, 10) (2, 11) (2, 12) (2, 13) (2, 14) (2, 15) (2, 16) (2, 17) (3, 4) "
            "(3, 5) (3, 6) (3, 7) (3, 8) (3, 9) (3, 10) (3, 16) (3, 17) (4, 5) (4, 6) (4, 7) (4, 8) (4, 9) "
            "(4, 11) (4, 12) (4, 13) (4, 14) (4, 15) (4, 16) (4, 17) (5, 7) (5, 8) (5, 9) (5, 10) (5, 11) "
            "(5, 12) (5, 14) (5, 15) (5, 16) (5, 17) (6, 8) (6, 9) (6, 10) (6, 11) (6, 12) (6, 14) (6, 15) "
            "(6, 16) (7, 8) (7, 9) (7, 11) (7, 13) (7, 15) (7, 16) (7, 17) (8, 9) (8, 11) (8, 12) (8, 13) "
            "(8, 16) (8, 17) (9, 10) (9, 12) (9, 13) (9, 16) (9, 17) (10, 11) (10, 12) (10, 13) (10, 14) "
            "(10, 15) (10, 16) (10, 17) (11, 13) (11, 14) (11, 15) (11, 17) (12, 13) (12, 14) (12, 15) (13, 14) "
            "(13, 15) (13, 16) (13, 17) (14, 15) (14, 16) (14, 17) (15, 17) (16, 17). "
            "Q: How many edges are in this graph? A:"
            # Expected answer: 115
        ),
        # "edge_count": (
        #     "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
        #     "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10. "
        #     "The edges in G are: (0, 3) (0, 5) (0, 6) (0, 9) (1, 2) (1, 3) (1, 4) (1, 6) (1, 8) (1, 10) (2, 3) "
        #     "(2, 6) (2, 7) (2, 9) (2, 10) (3, 4) (3, 8) (3, 9) (3, 10) (4, 5) (4, 6) (4, 9) (4, 10) (5, 7) "
        #     "(5, 10) (6, 7) (6, 8) (6, 9) (6, 10) (7, 8) (8, 9) (9, 10). "
        #     "Q: How many edges are in this graph? A:"
        #     # Expected answer: 32
        # ),
        "triangle_counting": (
            "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
            "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, and 7. "
            "The edges in G are: (0, 1) (0, 2) (0, 5) (0, 7) (1, 2) (1, 4) (1, 5) (1, 7) (2, 5) (3, 5) (5, 7). "
            "Q: How many triangles are in this graph?"
        ),
        "cycle_check": (
            "In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge. "
            "G describes a graph among nodes 0, 1, 2, 3, 4, 5, 6, and 7. "
            "The edges in G are: (0, 1) (0, 2) (0, 5) (0, 7) (1, 2) (1, 4) (1, 5) (1, 7) (2, 5) (3, 5) (5, 7). "
            "Q: Is there a cycle in this graph? A:"
        ),
    }
    return prompt_data[subset]


def solve_math(prompt: str, max_new_tokens: int, max_retries: int = 3, retry_wait: float = 2.0) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment. Please export OPENAI_API_KEY=...")

    client = OpenAI(api_key=api_key)

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
                temperature=0.2,
                max_output_tokens=max_new_tokens,
            )
            if hasattr(response, "output_text") and response.output_text:  # Convenience helper provided by the SDK
                return response.output_text.strip()
            # Fallback: manually concatenate the content
            parts: list[str] = []
            for item in getattr(response, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in {"output_text", "text"}:
                        txt = getattr(c, "text", None)
                        if txt and hasattr(txt, "value"):
                            parts.append(txt.value)
            if parts:
                return "".join(parts).strip()
            return str(response)
        except Exception as e:  # pragma: no cover
            last_error = e
            if attempt == max_retries:
                break
            wait = retry_wait * attempt
            print(
                f"[WARN] Call failed (attempt {attempt}/{max_retries}): {e}. Retrying after {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"OpenAI call failed: {last_error}")


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the portion after the literal 'Answer:' marker from the output; return None if it is missing."""
    import re

    m = re.search(r"Answer\s*[:：]\s*(.+)$", text.strip(), re.MULTILINE)
    if m:
        return m.group(1).strip()
    return None


def main() -> None:  # pragma: no cover
    args = build_args()

    try:
        prompt = get_prompt(args.subset)
        result = solve_math(prompt, max_new_tokens=1024)
    except Exception as e:  # pragma: no cover
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Create the output directory (tools/outputs/answers)
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Generate a random filename
    out_path = out_dir / f"{args.subset}.txt"
    # Write to the file (UTF-8)
    try:
        out_path.write_text(result, encoding="utf-8")
    except Exception as e:  # If writing fails, report the error and exit
        print(f"[ERROR] Failed to save the answer to a file: {e}", file=sys.stderr)
        sys.exit(1)

    print(result)
    print(f"Answer saved to: {out_path}")


if __name__ == "__main__":
    main()
