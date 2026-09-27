import json
import os
import re
import time
from collections import Counter, defaultdict

import joblib
import numpy as np
import torch

from sklearn.decomposition import IncrementalPCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans

from transformers import BertTokenizer, BertModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


# CONFIGURATION

DATA_PATH = "data/wikipedia/processed/articles.jsonl"
OUTPUT_DIR = "results/bert_token_final"

MODEL_NAME = "bert-base-uncased"

N_TOPICS = 20
N_TOP_WORDS = 15

MAX_LENGTH = 512
BATCH_SIZE = 8
MAX_TOKENS = 100_000

MIN_DOC_FREQ = 5
MAX_DOC_FREQ_RATIO = 0.25

PCA_COMPONENTS = 100

RANDOM_STATE = 42

KMEANS_MAX_ITER = 20
KMEANS_TOL = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# UTILITY FUNCTIONS

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_documents(path):
    

    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)

            if isinstance(item, dict):
                text = item.get("text", "")
            else:
                text = str(item)

            if text and text.strip():
                documents.append(text.strip())

    return documents


# DOCUMENT-FREQUENCY VOCABULARY

def basic_word_tokens(text):
    

    return re.findall(r"[A-Za-z]+", text.lower())


def build_document_frequency_vocabulary(documents):
    

    print()
    print("Building document-frequency vocabulary...")

    doc_count = len(documents)

    max_doc_freq = int(MAX_DOC_FREQ_RATIO * doc_count)

    df_counter = Counter()

    for document in documents:
        words = set(basic_word_tokens(document))

        for word in words:
            if len(word) <= 2:
                continue

            if word in ENGLISH_STOP_WORDS:
                continue

            df_counter[word] += 1

    vocabulary = {
        word
        for word, df in df_counter.items()
        if MIN_DOC_FREQ <= df <= max_doc_freq
    }

    print(f"Documents: {doc_count}")
    print(f"Vocabulary after DF filtering: {len(vocabulary)}")
    print(f"Minimum document frequency: {MIN_DOC_FREQ}")
    print(f"Maximum document frequency: {max_doc_freq}")

    return vocabulary, df_counter


# WORDPIECE RECONSTRUCTION

def reconstruct_words(
    input_ids,
    attention_mask,
    tokenizer,
    allowed_vocabulary
):


    words = []
    current_word = []
    current_indices = []

    special_ids = set(tokenizer.all_special_ids)

    for idx, (token_id, mask) in enumerate(
        zip(input_ids.tolist(), attention_mask.tolist())
    ):

        if mask == 0:
            continue

        if token_id in special_ids:
            continue

        token = tokenizer.convert_ids_to_tokens(token_id)

        if token.startswith("##"):

            if current_word:
                current_word.append(token[2:])
                current_indices.append(idx)

        else:

            # Finish previous word
            if current_word:

                word = "".join(current_word).lower()

                if (
                    word in allowed_vocabulary
                    and word.isalpha()
                    and len(word) > 2
                ):
                    words.append(
                        (
                            word,
                            current_indices.copy()
                        )
                    )

            # Start new word
            current_word = [token]
            current_indices = [idx]

    # Finish final word
    if current_word:

        word = "".join(current_word).lower()

        if (
            word in allowed_vocabulary
            and word.isalpha()
            and len(word) > 2
        ):
            words.append(
                (
                    word,
                    current_indices.copy()
                )
            )

    return words


# CONTEXTUAL WORD EMBEDDINGS

def get_token_embeddings(
    documents,
    tokenizer,
    model,
    vocabulary
):

    print()
    print("Generating contextual word embeddings...")

    all_embeddings = []
    all_words = []

    total_tokens = 0

    start_time = time.time()

    # Pending batches

    pending_input_ids = []
    pending_attention_masks = []

    pending_word_info = []

    # Process one batch

    def process_batch():

        nonlocal total_tokens

        if not pending_input_ids:
            return

        batch_input_ids = torch.stack(
            pending_input_ids
        ).to(DEVICE)

        batch_attention_masks = torch.stack(
            pending_attention_masks
        ).to(DEVICE)

        with torch.no_grad():

            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_masks
            )


        hidden_states = outputs.last_hidden_state

        for batch_index in range(
            len(pending_word_info)
        ):

            word_info = pending_word_info[
                batch_index
            ]

            for word, indices in word_info:

                if total_tokens >= MAX_TOKENS:
                    break

                vectors = hidden_states[
                    batch_index,
                    indices,
                    :
                ]

                # Mean-pool WordPieces
                word_embedding = vectors.mean(
                    dim=0
                )

                all_embeddings.append(
                    word_embedding.cpu().numpy()
                )

                all_words.append(word)

                total_tokens += 1

            if total_tokens >= MAX_TOKENS:
                break

        pending_input_ids.clear()
        pending_attention_masks.clear()
        pending_word_info.clear()

    # Process documents

    for doc_index, document in enumerate(documents):

        if total_tokens >= MAX_TOKENS:
            break


        encoded = tokenizer(
            document,
            truncation=True,
            max_length=MAX_LENGTH,
            return_overflowing_tokens=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"]
        attention_masks = encoded["attention_mask"]

        number_of_chunks = input_ids.shape[0]

        for chunk_index in range(number_of_chunks):

            if total_tokens >= MAX_TOKENS:
                break

            chunk_input_ids = input_ids[
                chunk_index
            ]

            chunk_attention_mask = attention_masks[
                chunk_index
            ]

            word_info = reconstruct_words(
                chunk_input_ids,
                chunk_attention_mask,
                tokenizer,
                vocabulary
            )

            if not word_info:
                continue

            pending_input_ids.append(
                chunk_input_ids
            )

            pending_attention_masks.append(
                chunk_attention_mask
            )

            pending_word_info.append(
                word_info
            )

            if len(pending_input_ids) >= BATCH_SIZE:
                process_batch()

        if (doc_index + 1) % 100 == 0:

            print(
                f"Processed documents: "
                f"{doc_index + 1}/{len(documents)} "
                f"| tokens: {total_tokens}"
            )

    # Process remaining chunks
    process_batch()

    elapsed = time.time() - start_time

    embeddings = np.asarray(
        all_embeddings,
        dtype=np.float32
    )

    print()
    print(
        f"Generated tokens: {len(all_words)}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Embedding time: {elapsed:.2f} seconds"
    )

    return embeddings, all_words, elapsed


# INCREMENTAL PCA

def apply_pca(embeddings):

    print()
    print(
        f"Applying Incremental PCA "
        f"({PCA_COMPONENTS} dimensions)..."
    )

    start_time = time.time()

    n_components = min(
        PCA_COMPONENTS,
        embeddings.shape[0],
        embeddings.shape[1]
    )

    # Batch size for IncrementalPCA
    batch_size = min(
        4096,
        embeddings.shape[0]
    )

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=batch_size
    )

    # Fit PCA
    pca.fit(embeddings)

    # Transform
    reduced = pca.transform(
        embeddings
    )

    elapsed = time.time() - start_time

    print(
        f"PCA output shape: {reduced.shape}"
    )

    print(
        f"PCA time: {elapsed:.2f} seconds"
    )

    return reduced, pca


# L2 NORMALIZATION

def normalize_embeddings(embeddings):

    print()
    print("L2-normalizing embeddings...")

    normalized = normalize(
        embeddings,
        norm="l2"
    )

    return normalized.astype(
        np.float32
    )


# SPHERICAL K-MEANS

def spherical_kmeans(
    embeddings,
    n_clusters,
    random_state=42,
    max_iter=20,
    tol=1e-4
):
    """
    Spherical K-Means.

    Since embeddings are L2-normalized, cosine similarity
    can be optimized by normalizing cluster centroids after
    every update.
    """

    print()
    print(
        f"Running spherical K-Means "
        f"(K={n_clusters})..."
    )

    start_time = time.time()

    rng = np.random.RandomState(
        random_state
    )

    n_samples = embeddings.shape[0]

    # Random initialization
    initial_indices = rng.choice(
        n_samples,
        size=n_clusters,
        replace=False
    )

    centers = embeddings[
        initial_indices
    ].copy()

    centers = normalize(
        centers,
        norm="l2"
    )

    labels = np.full(
        n_samples,
        -1,
        dtype=np.int32
    )

    for iteration in range(max_iter):

        # Cosine similarity because vectors are normalized

        similarities = (
            embeddings @ centers.T
        )

        new_labels = similarities.argmax(
            axis=1
        ).astype(np.int32)

        if np.array_equal(
            labels,
            new_labels
        ):
            print(
                f"Converged at iteration "
                f"{iteration + 1}"
            )
            labels = new_labels
            break

        if np.any(labels >= 0):

            changed = np.mean(
                labels != new_labels
            )

            if changed <= tol:

                print(
                    f"Converged at iteration "
                    f"{iteration + 1}"
                )

                labels = new_labels
                break

        labels = new_labels

        # Recompute centroids

        new_centers = np.zeros_like(
            centers
        )

        for cluster_id in range(
            n_clusters
        ):

            members = embeddings[
                labels == cluster_id
            ]

            if len(members) == 0:

                # Empty cluster:
                # choose a random point
                random_index = rng.randint(
                    n_samples
                )

                new_centers[
                    cluster_id
                ] = embeddings[
                    random_index
                ]

            else:

                center = members.mean(
                    axis=0
                )

                norm = np.linalg.norm(
                    center
                )

                if norm > 0:

                    center = center / norm

                new_centers[
                    cluster_id
                ] = center

        centers = new_centers

    elapsed = time.time() - start_time

    print(
        f"Clustering time: {elapsed:.2f} seconds"
    )

    return (
        labels,
        centers,
        elapsed
    )


# TOPIC WORDS

def get_topic_words(
    embeddings,
    words,
    labels,
    centers,
    n_topics,
    n_top_words
):

    print()
    print("Extracting representative topic words...")

    topic_words = []

    words = np.asarray(
        words
    )

    for topic_id in range(
        n_topics
    ):

        indices = np.where(
            labels == topic_id
        )[0]

        if len(indices) == 0:

            topic_words.append([])

            continue

        topic_embeddings = embeddings[
            indices
        ]

        center = centers[
            topic_id
        ]

        similarities = (
            topic_embeddings @ center
        )

        ranked = np.argsort(
            similarities
        )[::-1]

        selected = []

        seen = set()

        for rank_index in ranked:

            word = words[
                indices[rank_index]
            ]

            if word in seen:
                continue

            selected.append(
                word
            )

            seen.add(word)

            if len(selected) >= n_top_words:
                break

        topic_words.append(
            selected
        )

    return topic_words


# PRINT TOPICS

def print_topics(topic_words):

    print()
    print("=" * 70)
    print("DISCOVERED TOPICS")
    print("=" * 70)

    for topic_id, words in enumerate(
        topic_words,
        start=1
    ):

        print(
            f"Topic {topic_id:02d}: "
            + ", ".join(words)
        )


# EVALUATION

def calculate_coherence(
    topic_words,
    documents
):

    tokenized_documents = []

    for document in documents:

        tokens = [
            word.lower()
            for word in re.findall(
                r"[A-Za-z]+",
                document
            )
        ]

        tokenized_documents.append(
            tokens
        )

    dictionary = Dictionary(
        tokenized_documents
    )

    # Keep only topics containing valid dictionary words
    valid_topics = []

    for topic in topic_words:

        valid_words = [
            word
            for word in topic
            if word in dictionary.token2id
        ]

        if valid_words:
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


def calculate_topic_diversity(
    topic_words
):

    all_words = []

    for topic in topic_words:
        all_words.extend(topic)

    if not all_words:
        return 0.0

    return (
        len(set(all_words))
        / len(all_words)
    )


def calculate_topic_exclusivity(
    topic_words
):

    word_topic_count = Counter()

    for topic in topic_words:

        for word in set(topic):

            word_topic_count[word] += 1

    scores = []

    for topic in topic_words:

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


# SAVE TOPICS

def save_topics(topic_words):

    path = os.path.join(
        OUTPUT_DIR,
        "topics.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for topic_id, words in enumerate(
            topic_words,
            start=1
        ):

            f.write(
                f"Topic {topic_id:02d}: "
                + ", ".join(words)
                + "\n"
            )

    print(
        f"Saved topics to: {path}"
    )


# SAVE METRICS

def save_metrics(
    coherence,
    diversity,
    exclusivity,
    embedding_time,
    clustering_time,
    total_time,
    n_documents,
    n_tokens,
    embedding_dimension
):

    metrics = {
        "method": "BERT token-level clustering",
        "model": MODEL_NAME,
        "bert_layer": "final",
        "n_documents": n_documents,
        "n_tokens": n_tokens,
        "embedding_dimension": embedding_dimension,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "min_doc_freq": MIN_DOC_FREQ,
        "max_doc_freq_ratio": MAX_DOC_FREQ_RATIO,
        "pca_components": PCA_COMPONENTS,
        "n_topics": N_TOPICS,
        "n_top_words": N_TOP_WORDS,
        "random_state": RANDOM_STATE,
        "coherence_cv": coherence,
        "topic_diversity": diversity,
        "topic_exclusivity": exclusivity,
        "embedding_time_seconds": embedding_time,
        "clustering_time_seconds": clustering_time,
        "total_time_seconds": total_time
    }

    path = os.path.join(
        OUTPUT_DIR,
        "bert_token_metrics.json"
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        f"Saved metrics to: {path}"
    )


# MAIN

def main():

    overall_start = time.time()

    ensure_output_dir()

    # Load documents

    print("Loading documents...")

    documents = load_documents(
        DATA_PATH
    )

    print(
        f"Documents: {len(documents)}"
    )

    if len(documents) == 0:

        raise ValueError(
            "No documents found."
        )

    # Build vocabulary

    vocabulary, df_counter = (
        build_document_frequency_vocabulary(
            documents
        )
    )

    # Save vocabulary

    vocabulary_path = os.path.join(
        OUTPUT_DIR,
        "vocabulary.json"
    )

    with open(
        vocabulary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            sorted(vocabulary),
            f,
            indent=2
        )

    # Load BERT

    print()
    print("Loading BERT...")
    print(f"Device: {DEVICE}")

    tokenizer = BertTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = BertModel.from_pretrained(
        MODEL_NAME
    )

    model.to(
        DEVICE
    )

    model.eval()

    # Generate embeddings

    embeddings, words, embedding_time = (
        get_token_embeddings(
            documents,
            tokenizer,
            model,
            vocabulary
        )
    )

    if len(embeddings) == 0:

        raise ValueError(
            "No token embeddings were generated."
        )

    # Save raw embeddings

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "token_embeddings_raw.npy"
        ),
        embeddings
    )

    with open(
        os.path.join(
            OUTPUT_DIR,
            "token_words.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            words,
            f
        )

    # PCA

    reduced_embeddings, pca = (
        apply_pca(
            embeddings
        )
    )

    joblib.dump(
        pca,
        os.path.join(
            OUTPUT_DIR,
            "incremental_pca.joblib"
        )
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "token_embeddings_100d.npy"
        ),
        reduced_embeddings
    )

    # Free memory
    del embeddings

    # Normalize

    normalized_embeddings = (
        normalize_embeddings(
            reduced_embeddings
        )
    )

    # Spherical K-Means

    labels, centers, clustering_time = (
        spherical_kmeans(
            normalized_embeddings,
            n_clusters=N_TOPICS,
            random_state=RANDOM_STATE,
            max_iter=KMEANS_MAX_ITER,
            tol=KMEANS_TOL
        )
    )

    # Save clustering results

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "cluster_labels.npy"
        ),
        labels
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "cluster_centers.npy"
        ),
        centers
    )

    # Topic words

    topic_words = get_topic_words(
        normalized_embeddings,
        words,
        labels,
        centers,
        N_TOPICS,
        N_TOP_WORDS
    )

    print_topics(
        topic_words
    )

    save_topics(
        topic_words
    )

    # Evaluation

    print()
    print("=" * 70)
    print("EVALUATION")
    print("=" * 70)

    print("Calculating c_v coherence...")

    coherence = calculate_coherence(
        topic_words,
        documents
    )

    diversity = calculate_topic_diversity(
        topic_words
    )

    exclusivity = calculate_topic_exclusivity(
        topic_words
    )

    total_time = (
        time.time()
        - overall_start
    )

    print()
    print(
        f"c_v coherence:      {coherence:.4f}"
    )

    print(
        f"Topic diversity:    {diversity:.4f}"
    )

    print(
        f"Topic exclusivity:  {exclusivity:.4f}"
    )

    print(
        f"Embedding time:     {embedding_time:.2f}s"
    )

    print(
        f"Clustering time:    {clustering_time:.2f}s"
    )

    print(
        f"Total time:         {total_time:.2f}s"
    )

    # Save metrics

    save_metrics(
        coherence=coherence,
        diversity=diversity,
        exclusivity=exclusivity,
        embedding_time=embedding_time,
        clustering_time=clustering_time,
        total_time=total_time,
        n_documents=len(documents),
        n_tokens=len(words),
        embedding_dimension=PCA_COMPONENTS
    )

    # Final summary

    print()
    print("=" * 70)
    print("BERT TOKEN-LEVEL EXPERIMENT COMPLETE")
    print("=" * 70)

    print(
        f"Documents:          {len(documents)}"
    )

    print(
        f"Tokens clustered:   {len(words)}"
    )

    print(
        f"Topics:             {N_TOPICS}"
    )

    print(
        f"c_v coherence:      {coherence:.4f}"
    )

    print(
        f"Topic diversity:    {diversity:.4f}"
    )

    print(
        f"Topic exclusivity:  {exclusivity:.4f}"
    )

    print()
    print(
        f"Results saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()