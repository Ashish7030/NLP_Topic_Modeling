import json
import os
import re
from collections import Counter

import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


# CONFIGURATION

DATA_PATH = "data/wikipedia/processed/articles.jsonl"

TOPIC_WORDS = 15

RESULTS = {
    "LDA": "results/lda/topics.txt",
    "NMF": "results/nmf/topics.txt",
    "BERT Document": "results/bert_document/topics.txt",
    "BERT Token": "results/bert_token_final/topics.txt",
}

OUTPUT_DIR = "results/common_evaluation"


# LOAD DOCUMENTS

def load_documents(path):

    documents = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)

            if text.strip():
                documents.append(text.strip())

    return documents


# TOKENIZE CORPUS

def tokenize_documents(documents):

    tokenized = []

    for document in documents:

        tokens = re.findall(
            r"[A-Za-z]+",
            document.lower()
        )

        tokenized.append(tokens)

    return tokenized


# LOAD TOPICS

def load_topics(path):

    topics = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                continue

            _, words = line.split(
                ":",
                1
            )

            topic_words = [
                word.strip().lower()
                for word in words.split(",")
                if word.strip()
            ]

            topic_words = topic_words[
                :TOPIC_WORDS
            ]

            if topic_words:
                topics.append(
                    topic_words
                )

    return topics


# C_V COHERENCE

def calculate_coherence(
    topics,
    tokenized_documents,
    dictionary
):

    valid_topics = []

    for topic in topics:

        valid_words = [
            word
            for word in topic
            if word in dictionary.token2id
        ]

        if len(valid_words) >= 2:

            valid_topics.append(
                valid_words
            )

    if not valid_topics:
        return 0.0

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v"
    )

    return float(
        coherence_model.get_coherence()
    )


# TOPIC DIVERSITY

def calculate_diversity(topics):

    all_words = []

    for topic in topics:

        all_words.extend(topic)

    if not all_words:
        return 0.0

    unique_words = len(
        set(all_words)
    )

    return unique_words / len(
        all_words
    )


# TOPIC EXCLUSIVITY

def calculate_exclusivity(topics):

    word_topic_count = Counter()

    for topic in topics:

        for word in set(topic):

            word_topic_count[word] += 1

    scores = []

    for topic in topics:

        for word in topic:

            count = word_topic_count[
                word
            ]

            if count > 0:

                scores.append(
                    1.0 / count
                )

    if not scores:
        return 0.0

    return float(
        np.mean(scores)
    )


# TOPIC OVERLAP

def calculate_topic_overlap(topics):

    if len(topics) < 2:
        return 0.0

    overlaps = []

    for i in range(len(topics)):

        set_i = set(topics[i])

        for j in range(i + 1, len(topics)):

            set_j = set(topics[j])

            intersection = (
                len(set_i & set_j)
            )

            union = (
                len(set_i | set_j)
            )

            if union > 0:

                jaccard = (
                    intersection / union
                )

                overlaps.append(
                    jaccard
                )

    if not overlaps:
        return 0.0

    return float(
        np.mean(overlaps)
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("COMMON TOPIC MODEL EVALUATION")
    print("=" * 70)

    # Load corpus

    print()
    print("Loading corpus...")

    documents = load_documents(
        DATA_PATH
    )

    print(
        f"Documents: {len(documents)}"
    )

    # Tokenize once

    print(
        "Tokenizing corpus..."
    )

    tokenized_documents = (
        tokenize_documents(
            documents
        )
    )

    dictionary = Dictionary(
        tokenized_documents
    )

    print(
        f"Vocabulary: {len(dictionary)}"
    )

    # Evaluate each model

    results = []

    for method, topic_path in RESULTS.items():

        print()
        print("-" * 70)
        print(f"Evaluating: {method}")
        print("-" * 70)

        if not os.path.exists(
            topic_path
        ):

            print(
                f"WARNING: Missing file: "
                f"{topic_path}"
            )

            continue

        topics = load_topics(
            topic_path
        )

        print(
            f"Topics found: {len(topics)}"
        )

        if len(topics) == 0:

            print(
                "WARNING: No topics found."
            )

            continue

        coherence = calculate_coherence(
            topics,
            tokenized_documents,
            dictionary
        )

        diversity = calculate_diversity(
            topics
        )

        exclusivity = calculate_exclusivity(
            topics
        )

        overlap = calculate_topic_overlap(
            topics
        )

        result = {
            "method": method,
            "n_topics": len(topics),
            "n_words_per_topic": TOPIC_WORDS,
            "coherence_cv": coherence,
            "topic_diversity": diversity,
            "topic_exclusivity": exclusivity,
            "mean_topic_overlap_jaccard": overlap
        }

        results.append(
            result
        )

        print(
            f"c_v coherence:     {coherence:.4f}"
        )

        print(
            f"Topic diversity:   {diversity:.4f}"
        )

        print(
            f"Topic exclusivity: {exclusivity:.4f}"
        )

        print(
            f"Topic overlap:     {overlap:.4f}"
        )

    # Save JSON

    json_path = os.path.join(
        OUTPUT_DIR,
        "common_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # Save CSV

    csv_path = os.path.join(
        OUTPUT_DIR,
        "common_metrics.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Method,Topics,WordsPerTopic,"
            "CoherenceCV,Diversity,"
            "Exclusivity,MeanTopicOverlap\n"
        )

        for result in results:

            f.write(
                f"{result['method']},"
                f"{result['n_topics']},"
                f"{result['n_words_per_topic']},"
                f"{result['coherence_cv']:.4f},"
                f"{result['topic_diversity']:.4f},"
                f"{result['topic_exclusivity']:.4f},"
                f"{result['mean_topic_overlap_jaccard']:.4f}\n"
            )

    # Print final comparison

    print()
    print("=" * 70)
    print("FINAL COMMON EVALUATION")
    print("=" * 70)

    print()

    print(
        f"{'Method':<20}"
        f"{'c_v':>10}"
        f"{'Diversity':>12}"
        f"{'Exclusivity':>14}"
        f"{'Overlap':>12}"
    )

    print("-" * 70)

    for result in results:

        print(
            f"{result['method']:<20}"
            f"{result['coherence_cv']:>10.4f}"
            f"{result['topic_diversity']:>12.4f}"
            f"{result['topic_exclusivity']:>14.4f}"
            f"{result['mean_topic_overlap_jaccard']:>12.4f}"
        )

    print()
    print(
        f"Saved: {json_path}"
    )

    print(
        f"Saved: {csv_path}"
    )


if __name__ == "__main__":
    main()