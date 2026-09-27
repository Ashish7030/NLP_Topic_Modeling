import json
import os
import time

import joblib
import numpy as np
import torch

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel


# Configuration

DATA_PATH = "data/wikipedia/processed/articles.jsonl"
OUTPUT_DIR = "results/bert_document"

MODEL_NAME = "bert-base-uncased"

N_TOPICS = 20
N_TOP_WORDS = 15

RANDOM_STATE = 42

MAX_LENGTH = 256
BATCH_SIZE = 16

MIN_DF = 5
MAX_DF = 0.95
MAX_FEATURES = 20000


# Load documents

def load_documents(path):

    documents = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)
            documents.append(item["text"])

    return documents


# Metrics

def topic_diversity(topics):

    all_words = [
        word
        for topic in topics
        for word in topic
    ]

    return len(set(all_words)) / len(all_words)


def topic_exclusivity(topics):

    word_counts = {}

    for topic in topics:

        for word in topic:
            word_counts[word] = word_counts.get(word, 0) + 1

    scores = []

    for topic in topics:

        unique_words = sum(
            1
            for word in topic
            if word_counts[word] == 1
        )

        scores.append(
            unique_words / len(topic)
        )

    return sum(scores) / len(scores)


# Mean pooling

def mean_pooling(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1).expand(
        hidden_states.size()
    ).float()

    summed = torch.sum(
        hidden_states * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# Main

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load corpus

    print("Loading documents...")

    documents = load_documents(DATA_PATH)

    print(f"Documents: {len(documents)}")

    # Load BERT

    print("\nLoading BERT...")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    model.to(device)
    model.eval()

    # Generate document embeddings

    print("\nGenerating document embeddings...")

    embeddings = []

    start_time = time.time()

    for start in range(
        0,
        len(documents),
        BATCH_SIZE
    ):

        batch_documents = documents[
            start:start + BATCH_SIZE
        ]

        encoded = tokenizer(
            batch_documents,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            outputs = model(
                **encoded
            )

        batch_embeddings = mean_pooling(
            outputs.last_hidden_state,
            encoded["attention_mask"]
        )

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

        if (start // BATCH_SIZE) % 10 == 0:

            print(
                f"Processed "
                f"{min(start + BATCH_SIZE, len(documents))}"
                f"/{len(documents)} documents"
            )

    embeddings = np.vstack(embeddings)

    embedding_time = time.time() - start_time

    print(
        f"\nEmbedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding time: "
        f"{embedding_time:.2f} seconds"
    )

    # K-Means clustering

    print("\nRunning K-Means...")

    start_time = time.time()

    kmeans = KMeans(
        n_clusters=N_TOPICS,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        embeddings
    )

    clustering_time = time.time() - start_time

    print(
        f"Clustering time: "
        f"{clustering_time:.2f} seconds"
    )

    # TF-IDF for interpreting clusters

    print("\nExtracting topic words...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES
    )

    X = vectorizer.fit_transform(
        documents
    )

    feature_names = vectorizer.get_feature_names_out()

    topics = []

    for cluster_id in range(N_TOPICS):

        indices = np.where(
            cluster_labels == cluster_id
        )[0]

        if len(indices) == 0:
            topics.append([])
            continue

        cluster_matrix = X[indices]

        mean_tfidf = np.asarray(
            cluster_matrix.mean(axis=0)
        ).ravel()

        top_indices = mean_tfidf.argsort()[
            -N_TOP_WORDS:
        ][::-1]

        top_words = [
            feature_names[i]
            for i in top_indices
        ]

        topics.append(top_words)

    # Print topics

    print("\n" + "=" * 70)
    print("BERT DOCUMENT-LEVEL TOPICS")
    print("=" * 70)

    for i, topic in enumerate(topics, start=1):

        print(f"\nTopic {i}:")

        print(", ".join(topic))

    # Prepare coherence data

    tokenized_documents = [
        document.lower().split()
        for document in documents
    ]

    dictionary = Dictionary(
        tokenized_documents
    )

    # Remove empty topics if any
    valid_topics = [
        topic
        for topic in topics
        if topic
    ]

    # c_v coherence

    print("\nCalculating c_v coherence...")

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v"
    )

    coherence = coherence_model.get_coherence()

    # Diversity

    diversity = topic_diversity(
        valid_topics
    )

    # Exclusivity

    exclusivity = topic_exclusivity(
        valid_topics
    )

    total_time = (
        embedding_time +
        clustering_time
    )

    # Results

    print("\n" + "=" * 70)
    print("BERT DOCUMENT-LEVEL RESULTS")
    print("=" * 70)

    print(
        f"Embedding time:    "
        f"{embedding_time:.2f} seconds"
    )

    print(
        f"Clustering time:   "
        f"{clustering_time:.2f} seconds"
    )

    print(
        f"Total model time:   "
        f"{total_time:.2f} seconds"
    )

    print(
        f"c_v coherence:      "
        f"{coherence:.4f}"
    )

    print(
        f"Topic diversity:    "
        f"{diversity:.4f}"
    )

    print(
        f"Topic exclusivity:  "
        f"{exclusivity:.4f}"
    )

    # Save embeddings

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "document_embeddings.npy"
        ),
        embeddings
    )

    # Save cluster labels

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "cluster_labels.npy"
        ),
        cluster_labels
    )

    # Save model

    joblib.dump(
        kmeans,
        os.path.join(
            OUTPUT_DIR,
            "kmeans_model.joblib"
        )
    )

    # Save topics

    with open(
        os.path.join(
            OUTPUT_DIR,
            "topics.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        for i, topic in enumerate(
            topics,
            start=1
        ):

            f.write(
                f"Topic {i}: "
                + ", ".join(topic)
                + "\n"
            )

    # Save metrics

    metrics = {

        "model": MODEL_NAME,

        "n_documents": len(documents),

        "n_topics": N_TOPICS,

        "n_top_words": N_TOP_WORDS,

        "embedding_dimension": int(
            embeddings.shape[1]
        ),

        "embedding_time_seconds":
            embedding_time,

        "clustering_time_seconds":
            clustering_time,

        "total_model_time_seconds":
            total_time,

        "coherence_cv":
            coherence,

        "topic_diversity":
            diversity,

        "topic_exclusivity":
            exclusivity,

        "max_length":
            MAX_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "random_state":
            RANDOM_STATE
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            "bert_document_metrics.json"
        ),
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