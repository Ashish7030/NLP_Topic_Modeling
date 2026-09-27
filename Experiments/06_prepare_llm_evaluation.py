import os
import json
import numpy as np


# PROJECT PATHS

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "llm_evaluation"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# INPUT FILES

METHOD_FILES = {
    "LDA": os.path.join(
        RESULTS_DIR,
        "lda",
        "topics.txt"
    ),

    "NMF": os.path.join(
        RESULTS_DIR,
        "nmf",
        "topics.txt"
    ),

    "BERT Document": os.path.join(
        RESULTS_DIR,
        "bert_document",
        "topics.txt"
    ),

    "BERT Token": os.path.join(
        RESULTS_DIR,
        "bert_token_final",
        "topics.txt"
    ),
}


# OUTPUT FILE

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "topics_for_llm_evaluation.json"
)


# PRINT PATHS

print("=" * 70)
print("Preparing topics for LLM evaluation")
print("=" * 70)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print(f"\nOutput file:")
print(OUTPUT_FILE)


# CHECK INPUT FILES

print("\nChecking topic files...")

missing_files = []

for method, filepath in METHOD_FILES.items():

    if os.path.exists(filepath):
        print(f"✓ {method}: {filepath}")
    else:
        print(f"✗ {method}: NOT FOUND")
        missing_files.append((method, filepath))


if missing_files:

    print("\nERROR: The following topic files are missing:\n")

    for method, filepath in missing_files:
        print(f"{method}:")
        print(f"  {filepath}")

    raise FileNotFoundError(
        "One or more topic files are missing. "
        "Check that the previous experiments have been completed."
    )


# PARSE TOPICS

def parse_topics(filepath):
    """
    Read topics.txt.

    Expected format:

    Topic 0: word1, word2, word3, ...
    Topic 1: word1, word2, word3, ...
    """

    topics = []

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue


            if ":" not in line:
                continue

            topic_name, words_text = line.split(":", 1)

            words = [
                word.strip()
                for word in words_text.split(",")
                if word.strip()
            ]

            if len(words) == 0:
                continue

            topics.append({
                "topic_id": len(topics),
                "words": words
            })

    return topics


# LOAD ALL TOPICS

all_topics = []

for method, filepath in METHOD_FILES.items():

    print(f"\nReading {method}...")

    topics = parse_topics(filepath)

    print(f"Found {len(topics)} topics")

    for topic in topics:

        all_topics.append({
            "method": method,
            "topic_id": topic["topic_id"],
            "words": topic["words"]
        })


# CHECK EXPECTED NUMBER

print("\n" + "=" * 70)
print("Topic summary")
print("=" * 70)

method_counts = {}

for topic in all_topics:

    method = topic["method"]

    method_counts[method] = (
        method_counts.get(method, 0) + 1
    )


for method, count in method_counts.items():

    print(f"{method:20s}: {count} topics")


print(f"\nTotal topics: {len(all_topics)}")


# SAVE JSON

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_topics,
        f,
        indent=2,
        ensure_ascii=False
    )


# VERIFY OUTPUT

if not os.path.exists(OUTPUT_FILE):

    raise RuntimeError(
        "Output file was not created."
    )


file_size = os.path.getsize(OUTPUT_FILE)


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"\nCreated:")
print(OUTPUT_FILE)

print(f"\nFile size:")
print(f"{file_size / 1024:.1f} KB")

print(f"\nNumber of topics:")
print(len(all_topics))

print("\nMethods:")

for method, count in method_counts.items():
    print(f"  {method}: {count}")

print("\nYou can now run:")
print(
    "./.venv/bin/python "
    "Experiments/06_llm_evaluation.py"
)

print("=" * 70)