import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


# CONFIGURATION

DATA_DIR = Path("data/wikipedia")

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

OUTPUT_DIR = DATA_DIR / "processed"

N_ARTICLES = 1000
SEED = 42

MIN_TOKENS = 100


# WIKITEXT HEADINGS

ARTICLE_HEADING = re.compile(
    r"^\s*=\s+(.+?)\s+=\s*$"
)


# READ PARQUET

def iter_rows():

    for path in TRAIN_FILES:

        if not path.exists():

            raise FileNotFoundError(
                f"\nFile not found:\n{path}\n\n"
                "Expected files:\n"
                "data/wikipedia/"
                "train-00000-of-00002.parquet\n"
                "data/wikipedia/"
                "train-00001-of-00002.parquet"
            )

        print(f"Reading: {path}")

        parquet = pq.ParquetFile(path)

        for batch in parquet.iter_batches(
            batch_size=10000,
            columns=["text"]
        ):

            for text in batch.column(
                "text"
            ).to_pylist():

                yield text


# RECONSTRUCT DOCUMENTS

def iter_documents():

    current_title = None
    current_lines = []

    for text in iter_rows():

        if text is None:
            continue

        line = text.rstrip()

        match = ARTICLE_HEADING.match(
            line
        )

        if match:

            # Save previous document
            if current_title is not None:

                body = "\n".join(
                    current_lines
                ).strip()

                tokens = body.split()

                if len(tokens) >= MIN_TOKENS:

                    yield (
                        current_title,
                        body
                    )

            # Start new document
            current_title = (
                match.group(1).strip()
            )

            current_lines = []

        else:

            if current_title is not None:

                current_lines.append(line)

    # Save final document
    if current_title is not None:

        body = "\n".join(
            current_lines
        ).strip()

        if len(body.split()) >= MIN_TOKENS:

            yield (
                current_title,
                body
            )


# RANDOM SAMPLING

def select_documents():

    rng = random.Random(SEED)

    reservoir = []

    total_documents = 0

    for document in iter_documents():

        total_documents += 1

        if len(reservoir) < N_ARTICLES:

            reservoir.append(document)

        else:

            index = rng.randrange(
                total_documents
            )

            if index < N_ARTICLES:

                reservoir[index] = document

    return (
        reservoir,
        total_documents
    )


# STATISTICS

def get_statistics(documents):

    lengths = []

    vocabulary = set()

    for title, text in documents:

        tokens = text.split()

        lengths.append(
            len(tokens)
        )

        vocabulary.update(
            tokens
        )

    total_tokens = sum(lengths)

    return {

        "documents":
            len(documents),

        "tokens":
            total_tokens,

        "vocabulary":
            len(vocabulary),

        "average_tokens_per_document":
            total_tokens / len(documents),

        "minimum_tokens_per_document":
            min(lengths),

        "maximum_tokens_per_document":
            max(lengths),

        "median_tokens_per_document":
            sorted(lengths)[
                len(lengths) // 2
            ],

        "minimum_document_length":
            MIN_TOKENS,

        "random_seed":
            SEED
    }


# SAVE DATASET

def save_dataset(
    documents,
    total_available
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # JSONL

    jsonl_file = (
        OUTPUT_DIR /
        "articles.jsonl"
    )

    with jsonl_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for i, (title, text) in enumerate(
            documents
        ):

            record = {

                "id": i,

                "title":
                    title,

                "text":
                    text
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    # Plain text

    text_file = (
        OUTPUT_DIR /
        "articles.txt"
    )

    with text_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for title, text in documents:

            f.write(
                f"= {title} =\n"
            )

            f.write(text)

            f.write("\n\n")

    # Statistics

    stats = get_statistics(
        documents
    )

    stats.update({

        "dataset":
            "WikiText-103",

        "configuration":
            "wikitext-103-raw-v1",

        "split":
            "train",

        "requested_documents":
            N_ARTICLES,

        "available_documents_after_filter":
            total_available,

        "selection":
            "random reservoir sampling",

        "source_files":
            [
                str(x)
                for x in TRAIN_FILES
            ]
    })

    stats_file = (
        OUTPUT_DIR /
        "corpus_stats.json"
    )

    with stats_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    return (
        jsonl_file,
        text_file,
        stats_file,
        stats
    )


def main():

    print("=" * 65)
    print("WikiText-103 Dataset Preparation")
    print("=" * 65)

    print(
        f"\nTarget documents: {N_ARTICLES}"
    )

    print(
        f"Minimum document length: "
        f"{MIN_TOKENS} tokens"
    )

    print(
        f"Random seed: {SEED}"
    )

    print(
        "\nReconstructing documents..."
    )

    documents, total_available = (
        select_documents()
    )

    print(
        f"\nValid documents detected: "
        f"{total_available:,}"
    )

    print(
        f"Selected documents: "
        f"{len(documents):,}"
    )

    if len(documents) < N_ARTICLES:

        raise RuntimeError(
            "\nNot enough valid documents "
            "were found."
        )

    # Stable ordering after sampling
    documents.sort(
        key=lambda x:
        x[0].lower()
    )

    (
        jsonl_file,
        text_file,
        stats_file,
        stats
    ) = save_dataset(
        documents,
        total_available
    )

    print(
        "\n" + "=" * 65
    )

    print("DATASET READY")

    print(
        "=" * 65
    )

    print(
        f"\nJSONL:\n{jsonl_file}"
    )

    print(
        f"\nText corpus:\n{text_file}"
    )

    print(
        f"\nStatistics:\n{stats_file}"
    )

    print(
        "\nCorpus statistics:"
    )

    print(
        f"  Documents: "
        f"{stats['documents']:,}"
    )

    print(
        f"  Tokens: "
        f"{stats['tokens']:,}"
    )

    print(
        f"  Vocabulary: "
        f"{stats['vocabulary']:,}"
    )

    print(
        f"  Average tokens/document: "
        f"{stats['average_tokens_per_document']:.2f}"
    )

    print(
        f"  Median tokens/document: "
        f"{stats['median_tokens_per_document']:,}"
    )

    print(
        f"  Minimum tokens/document: "
        f"{stats['minimum_tokens_per_document']:,}"
    )

    print(
        f"  Maximum tokens/document: "
        f"{stats['maximum_tokens_per_document']:,}"
    )


if __name__ == "__main__":
    main()