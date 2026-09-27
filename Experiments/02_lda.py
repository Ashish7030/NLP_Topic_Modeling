import json
import time
from pathlib import Path

import joblib
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


# CONFIGURATION

CORPUS_FILE = Path(
    "data/wikipedia/processed/articles.jsonl"
)

OUTPUT_DIR = Path("results/lda")

N_TOPICS = 20
N_TOP_WORDS = 15

RANDOM_STATE = 42
MAX_ITER = 20

MIN_DF = 5
MAX_DF = 0.95

MAX_FEATURES = 20000


# LOAD CORPUS

def load_documents():

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(
            f"Corpus not found:\n{CORPUS_FILE}\n\n"
            "Run 01_prepare_wikitext.py first."
        )

    documents = []
    titles = []

    with CORPUS_FILE.open(
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            record = json.loads(line)

            titles.append(record["title"])
            documents.append(record["text"])

    return titles, documents


# TOP WORDS

def get_top_words(model, feature_names):

    topics = []

    for topic_id, topic in enumerate(
        model.components_
    ):

        top_indices = topic.argsort()[
            ::-1
        ][:N_TOP_WORDS]

        words = [
            feature_names[index]
            for index in top_indices
        ]

        topics.append(words)

    return topics


# TOPIC DIVERSITY

def topic_diversity(topics):

    words = [
        word
        for topic in topics
        for word in topic
    ]

    return len(set(words)) / len(words)


# TOPIC EXCLUSIVITY

def topic_exclusivity(topics):

    word_counts = {}

    for topic in topics:

        for word in topic:

            word_counts[word] = (
                word_counts.get(word, 0) + 1
            )

    scores = []

    for topic in topics:

        if not topic:
            continue

        exclusive = sum(
            1
            for word in topic
            if word_counts[word] == 1
        )

        scores.append(
            exclusive / len(topic)
        )

    return float(np.mean(scores))


# COHERENCE

def calculate_coherence(
    documents,
    topics
):

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    dictionary = Dictionary(
        tokenized_documents
    )

    # Remove extremely rare tokens from the coherence dictionary.
    dictionary.filter_extremes(
        no_below=2
    )

    valid_topics = []

    for topic in topics:

        filtered_topic = [
            word
            for word in topic
            if word in dictionary.token2id
        ]

        if len(filtered_topic) >= 2:
            valid_topics.append(
                filtered_topic
            )

    if not valid_topics:
        return None

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    )

    return float(
        coherence_model.get_coherence()
    )


# SAVE TOPICS

def save_topics(topics):

    topics_file = (
        OUTPUT_DIR / "topics.txt"
    )

    with topics_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for topic_id, words in enumerate(
            topics
        ):

            f.write(
                f"Topic {topic_id + 1}: "
            )

            f.write(
                ", ".join(words)
            )

            f.write("\n")

    return topics_file


def main():

    print("=" * 65)
    print("Experiment 1 — LDA Baseline")
    print("=" * 65)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nLoading corpus:\n{CORPUS_FILE}"
    )

    titles, documents = load_documents()

    print(
        f"Documents loaded: "
        f"{len(documents):,}"
    )

    # Vectorization

    print("\nCreating document-term matrix...")

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES
    )

    X = vectorizer.fit_transform(
        documents
    )

    feature_names = (
        vectorizer.get_feature_names_out()
    )

    print(
        f"Matrix shape: {X.shape}"
    )

    # Train LDA

    print(
        f"\nTraining LDA with "
        f"{N_TOPICS} topics..."
    )

    start_time = time.perf_counter()

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=MAX_ITER,
        learning_method="batch",
        random_state=RANDOM_STATE,
        evaluate_every=-1,
        n_jobs=-1
    )

    lda.fit(X)

    runtime = (
        time.perf_counter()
        - start_time
    )

    print(
        f"LDA training time: "
        f"{runtime:.2f} seconds"
    )

    # Topics

    topics = get_top_words(
        lda,
        feature_names
    )

    # Metrics

    print("\nCalculating topic diversity...")

    diversity = topic_diversity(
        topics
    )

    print(
        f"Topic diversity: "
        f"{diversity:.4f}"
    )

    print("\nCalculating topic exclusivity...")

    exclusivity = topic_exclusivity(
        topics
    )

    print(
        f"Topic exclusivity: "
        f"{exclusivity:.4f}"
    )

    print("\nCalculating c_v coherence...")

    coherence = calculate_coherence(
        documents,
        topics
    )

    if coherence is not None:

        print(
            f"c_v coherence: "
            f"{coherence:.4f}"
        )

    else:

        print(
            "c_v coherence could not be calculated."
        )

    # Document-topic distributions

    print(
        "\nCalculating document-topic distributions..."
    )

    document_topics = (
        lda.transform(X)
    )

    np.save(
        OUTPUT_DIR /
        "document_topic_distributions.npy",
        document_topics
    )

    # Save model

    joblib.dump(
        {
            "model": lda,
            "vectorizer": vectorizer,
        },
        OUTPUT_DIR /
        "lda_model.joblib"
    )

    # Save topics

    topics_file = save_topics(
        topics
    )

    # Save metrics

    metrics = {

        "experiment":
            "LDA baseline",

        "dataset":
            "WikiText-103",

        "documents":
            len(documents),

        "n_topics":
            N_TOPICS,

        "top_words_per_topic":
            N_TOP_WORDS,

        "max_iter":
            MAX_ITER,

        "random_state":
            RANDOM_STATE,

        "min_df":
            MIN_DF,

        "max_df":
            MAX_DF,

        "max_features":
            MAX_FEATURES,

        "vocabulary_size":
            len(feature_names),

        "document_term_matrix_rows":
            X.shape[0],

        "document_term_matrix_columns":
            X.shape[1],

        "runtime_seconds":
            runtime,

        "topic_coherence_cv":
            coherence,

        "topic_diversity":
            diversity,

        "topic_exclusivity":
            exclusivity
    }

    metrics_file = (
        OUTPUT_DIR /
        "lda_metrics.json"
    )

    with metrics_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # Print topics

    print(
        "\n" + "=" * 65
    )

    print("DISCOVERED TOPICS")

    print(
        "=" * 65
    )

    for topic_id, words in enumerate(
        topics
    ):

        print(
            f"\nTopic {topic_id + 1}: "
            + ", ".join(words)
        )

    # Final output

    print(
        "\n" + "=" * 65
    )

    print("LDA EXPERIMENT COMPLETE")

    print(
        "=" * 65
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nTopics:"
        f"\n  {topics_file}"
    )

    print(
        f"\nMetrics:"
        f"\n  {metrics_file}"
    )

    print(
        "\nModel:"
        f"\n  {OUTPUT_DIR / 'lda_model.joblib'}"
    )


if __name__ == "__main__":
    main()
