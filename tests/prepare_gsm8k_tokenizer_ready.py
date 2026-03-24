import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


ANSWER_RE = re.compile(r"####\s*([-0-9.,]+)")


def convert_split(split: str, out_path: Path, limit: int | None, use_boxed: bool) -> None:
    ds = load_dataset("gsm8k", "main", split=split if limit is None else f"{split}[:{limit}]")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    suffix = "and put the final answer in \\boxed{...}." if use_boxed else "and end with only the final answer."

    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            match = ANSWER_RE.search(row["answer"])
            if match is None:
                continue

            record = {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            "Solve the following GSM8K problem. Show your reasoning if needed, "
                            f"{suffix}\n\n{row['question'].strip()}"
                        ),
                    }
                ],
                "label": match.group(1).replace(",", "").strip(),
                "metadata": {"source": "gsm8k"},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--boxed", action="store_true")
    args = parser.parse_args()

    convert_split("train", args.train_out, args.train_limit, args.boxed)
    if args.test_out is not None:
        convert_split("test", args.test_out, args.test_limit, args.boxed)


if __name__ == "__main__":
    main()
