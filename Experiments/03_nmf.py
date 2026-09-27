import json
import os
import time

import joblib
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


# Configuration

DATA_PATH = "data/wikipedia/processed/articles.jsonl"
OUTPUT_DIR = "results/nmf"

N_TOPICS = 20
N_TOP_WORDS = 15

RANDOM_STATE = 42
MAX_ITER = 200

MIN_DF = 5
MAX_DF = 0.95
MAX_FEATURES = 20000


# Load corpus

def load_documents(path):
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            documents.append(item["text"])

    return documents


# Topic metrics

def topic_diversity(topics):
    all_words = [word for topic in topics for word in topic]
    return len(set(all_words)) / len(all_words)


def topic_exclusivity(topics):
    word_counts = {}

    for topic in topics:
        for word in topic:
            word_counts[word] = word_counts.get(word, 0) + 1

    topic_scores = []

    for topic in topics:
        unique_words = sum(
            1 for word in topic
            if word_counts[word] == 1
        )

        topic_scores.append(unique_words / len(topic))

    return sum(topic_scores) / len(topic_scores)


# Main

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading documents...")
    documents = load_documents(DATA_PATH)

    print(f"Documents: {len(documents)}")

    # TF-IDF representation

    print("\nBuilding TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES
    )

    X = vectorizer.fit_transform(documents)

    print(f"TF-IDF matrix shape: {X.shape}")

    feature_names = vectorizer.get_feature_names_out()

    # Train NMF

    print("\nTraining NMF...")

    start_time = time.time()

    nmf = NMF(
        n_components=N_TOPICS,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=MAX_ITER
    )

    document_topic = nmf.fit_transform(X)

    training_time = time.time() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    # Extract topics

    topics = []

    for topic_idx, topic in enumerate(nmf.components_):

        top_indices = topic.argsort()[-N_TOP_WORDS:][::-1]

        top_words = [
            feature_names[i]
            for i in top_indices
        ]

        topics.append(top_words)

    # Print topics

    print("\n" + "=" * 70)
    print("NMF TOPICS")
    print("=" * 70)

    for i, topic in enumerate(topics, start=1):
        print(f"\nTopic {i}:")
        print(", ".join(topic))

    # Prepare documents for coherence

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    dictionary = Dictionary(tokenized_documents)

    # c_v coherence

    print("\nCalculating c_v coherence...")

    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v"
    )

    coherence = coherence_model.get_coherence()

    # Diversity

    diversity = topic_diversity(topics)

    # Exclusivity

    exclusivity = topic_exclusivity(topics)

    # Print metrics

    print("\n" + "=" * 70)
    print("NMF RESULTS")
    print("=" * 70)

    print(f"Training time:      {training_time:.2f} seconds")
    print(f"c_v coherence:      {coherence:.4f}")
    print(f"Topic diversity:    {diversity:.4f}")
    print(f"Topic exclusivity:  {exclusivity:.4f}")

    # Save model

    joblib.dump(
        {
            "model": nmf,
            "vectorizer": vectorizer
        },
        os.path.join(OUTPUT_DIR, "nmf_model.joblib")
    )

    # Save document-topic distributions

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "document_topic_distributions.npy"
        ),
        document_topic
    )

    # Save topics

    with open(
        os.path.join(OUTPUT_DIR, "topics.txt"),
        "w",
        encoding="utf-8"
    ) as f:

        for i, topic in enumerate(topics, start=1):
            f.write(
                f"Topic {i}: "
                + ", ".join(topic)
                + "\n"
            )

    # Save metrics

    metrics = {
        "n_documents": len(documents),
        "n_topics": N_TOPICS,
        "n_top_words": N_TOP_WORDS,
        "matrix_shape": list(X.shape),
        "training_time_seconds": training_time,
        "coherence_cv": coherence,
        "topic_diversity": diversity,
        "topic_exclusivity": exclusivity,
        "random_state": RANDOM_STATE,
        "max_iter": MAX_ITER,
        "min_df": MIN_DF,
        "max_df": MAX_DF,
        "max_features": MAX_FEATURES
    }

    with open(
        os.path.join(OUTPUT_DIR, "nmf_metrics.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()