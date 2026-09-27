import os
import json
import csv
import time
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


# CONFIGURATION

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

MAX_NEW_TOKENS = 220
DO_SAMPLE = False
MAX_RETRIES = 3


# PROJECT PATHS

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "llm_evaluation",
    "topics_for_llm_evaluation.json"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "llm_evaluation"
)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR,
    "llm_topic_scores_strict.json"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "llm_topic_scores_strict.csv"
)

OUTPUT_AVERAGES = os.path.join(
    OUTPUT_DIR,
    "llm_method_averages_strict.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# DEVICE

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


# HEADER

print("=" * 70)
print("STRICT LOCAL LLM TOPIC EVALUATION")
print("=" * 70)

print(f"Model:  {MODEL_NAME}")
print(f"Device: {DEVICE}")

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# CHECK INPUT

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nTopic evaluation file was not found:\n{INPUT_FILE}\n\n"
        "Run 06_prepare_llm_evaluation.py first."
    )


# LOAD TOPICS

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    topics = json.load(f)


print(f"\nTopics to evaluate: {len(topics)}")


# LOAD MODEL

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.to(DEVICE)
model.eval()

print("Model loaded.")


# STRICT EVALUATION PROMPT

SYSTEM_PROMPT = """
You are a highly critical NLP researcher evaluating topic-model outputs.

Your task is to AUDIT the quality of a topic represented by its top words.

IMPORTANT:
Do NOT try to invent a creative or broad umbrella label simply to make
the topic appear coherent.

Judge the WORDS THEMSELVES.

A topic should receive a high score only when its words genuinely form
a coherent and reasonably specific semantic group.

------------------------------------------------------------
STRICT PENALIZATION RULES
------------------------------------------------------------

1. GENERIC / FUNCTIONAL WORDS

Words such as:

result, fact, thing, time, used, main, new, possible,
particularly, greatly, significantly, eventually, quickly,
usually, generally

are weak topic indicators.

If a topic contains many such words, reduce coherence and specificity.

2. MIXED SEMANTIC FIELDS

If words belong to clearly different domains, penalize coherence.

For example:

schools, railway, film, station, education

should NOT be rescued by inventing a label such as:

"Transportation and Education"

Instead, recognize that the topic contains multiple semantic fields.

3. NUMERICAL / FORMAT NOISE

Tokens such as:

000, 2011, ft, km, mm, sr

should generally be treated as noise unless they clearly contribute to
a coherent domain.

4. GENERIC UMBRELLA LABELS

Do not create broad labels merely to justify a mixed list.

For example, do not turn:

film, school, railway, city, station

into:

"Urban Life and Culture"

5. PARTIAL COHERENCE

If only some words clearly belong together while several others do not,
the topic should receive a moderate or low score.

6. SPECIFICITY

A topic containing a recognizable but very broad domain should not
automatically receive 5.

For example:

"people, work, time, life, world"

is not a specific topic.

------------------------------------------------------------
RATING SCALE
------------------------------------------------------------

5 = Excellent

Nearly all words strongly belong to one clear and specific semantic
topic. Very little or no noise.

4 = Good

A clear topic is present. Most words fit, with only minor noise or
generic terms.

3 = Moderate

A recognizable topic exists, but several words are generic, broad,
or weakly related.

2 = Weak

The topic contains substantial noise, weak semantic relationships,
or multiple semantic fields.

1 = Very poor

The words form a largely incoherent collection, contain severe domain
mixing, or consist mainly of generic/non-topical words.

------------------------------------------------------------
CALIBRATION EXAMPLES
------------------------------------------------------------

Example 1 — GOOD:

[
"hurricane",
"cyclone",
"storm",
"landfall",
"winds",
"rainfall"
]

Expected assessment:

coherence = 5
specificity = 5

Reason:
The words form a highly coherent and specific weather-related topic.

------------------------------------------------------------

Example 2 — POOR:

[
"fact",
"result",
"greatly",
"particularly",
"eventually",
"addition",
"point",
"quickly"
]

Expected assessment:

coherence = 1 or 2
specificity = 1

Reason:
The words are mostly generic or functional terms and do not identify
a concrete semantic topic.

------------------------------------------------------------

Example 3 — MIXED:

[
"000",
"ft",
"station",
"million",
"city",
"film",
"schools",
"railway"
]

Expected assessment:

coherence = 2
specificity = 2

Reason:
The words mix numerical noise with transportation, urban, film,
and education-related concepts.

------------------------------------------------------------

Example 4 — GOOD:

[
"league",
"football",
"team",
"match",
"goal",
"player",
"coach"
]

Expected assessment:

coherence = 5
specificity = 4 or 5

Reason:
The words form a clear football-related topic.

------------------------------------------------------------

IMPORTANT:
Do not assume that every topic has a meaningful topic.
It is acceptable, and sometimes preferable, to classify a topic as
"Non-Topic" or "Mixed Domain Noise".

------------------------------------------------------------
OUTPUT FORMAT
------------------------------------------------------------

Return ONLY a valid JSON object.

Use exactly this structure:

{
  "topic_label": "short topic label",
  "coherence": 1,
  "interpretability": 1,
  "specificity": 1,
  "overall": 1,
  "problematic_words": [],
  "brief_reason": "One sentence explaining the evaluation."
}

Rules:

- coherence: integer from 1 to 5
- interpretability: integer from 1 to 5
- specificity: integer from 1 to 5
- overall: integer from 1 to 5
- problematic_words: list of words that do not fit well
- brief_reason: exactly one concise sentence
"""


# PROMPT BUILDER

def build_prompt(words):

    word_string = ", ".join(words)

    return f"""
Evaluate the following topic words STRICTLY.

Topic words:
[{word_string}]

Remember:

- Judge the words themselves.
- Do not invent a broad umbrella topic to rescue mixed words.
- Penalize generic words.
- Penalize numerical/format noise.
- Penalize unrelated semantic fields.
- Identify problematic words explicitly.
- A topic does not need to receive a high score simply because a
  plausible label can be invented.

Return ONLY the JSON object.
"""


# JSON EXTRACTION

def extract_json(text):

    text = text.strip()

    # Remove markdown fences
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    # Direct JSON
    try:

        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except Exception:
        pass

    # Search for JSON object inside response
    start = text.find("{")

    if start == -1:
        return None

    depth = 0

    for i in range(start, len(text)):

        if text[i] == "{":
            depth += 1

        elif text[i] == "}":

            depth -= 1

            if depth == 0:

                candidate = text[start:i + 1]

                try:

                    result = json.loads(candidate)

                    if isinstance(result, dict):
                        return result

                except Exception:
                    return None

    return None


# VALIDATION

def validate_result(result):

    if not isinstance(result, dict):
        return False

    required = [
        "topic_label",
        "coherence",
        "interpretability",
        "specificity",
        "overall",
        "problematic_words",
        "brief_reason"
    ]

    for key in required:

        if key not in result:
            return False

    score_fields = [
        "coherence",
        "interpretability",
        "specificity",
        "overall"
    ]

    for key in score_fields:

        value = result[key]

        if not isinstance(value, int):
            return False

        if value < 1 or value > 5:
            return False

    if not isinstance(
        result["problematic_words"],
        list
    ):
        return False

    if not isinstance(
        result["brief_reason"],
        str
    ):
        return False

    return True


# EVALUATE ONE TOPIC

def evaluate_topic(words):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": build_prompt(words)
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    result = extract_json(response)

    return result, response


# MAIN EVALUATION

results = []

start_time = time.time()


for index, topic in enumerate(
    topics,
    start=1
):

    method = topic["method"]
    topic_id = topic["topic_id"]
    words = topic["words"]

    print()
    print(
        f"[{index}/{len(topics)}] "
        f"Auditing {method} topic {topic_id}..."
    )

    result = None
    raw_response = ""

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            result, raw_response = evaluate_topic(
                words
            )

            if validate_result(result):
                break

        except Exception as e:

            print(
                f"  Attempt {attempt} failed: {e}"
            )

    # FALLBACK

    if not validate_result(result):

        print("  WARNING: Could not parse valid JSON.")

        result = {
            "topic_label": "Unparseable",
            "coherence": 1,
            "interpretability": 1,
            "specificity": 1,
            "overall": 1,
            "problematic_words": words[:3],
            "brief_reason":
                "The model output could not be parsed into the required format."
        }

    # ADD METADATA

    result["method"] = method
    result["topic_id"] = topic_id
    result["words"] = words
    result["raw_response"] = raw_response

    results.append(result)

    # PRINT RESULT

    print(
        f"  Label: {result['topic_label']}"
    )

    print(
        f"  Scores -> "
        f"Coh={result['coherence']} | "
        f"Interp={result['interpretability']} | "
        f"Spec={result['specificity']} | "
        f"Overall={result['overall']}"
    )

    print(
        f"  Problematic words: "
        f"{result['problematic_words']}"
    )

    print(
        f"  Critique: "
        f"{result['brief_reason']}"
    )


# METHOD AVERAGES

methods = {}

for result in results:

    method = result["method"]

    if method not in methods:
        methods[method] = []

    methods[method].append(result)


averages = []


for method, method_results in methods.items():

    def average(field):

        return round(
            sum(
                r[field]
                for r in method_results
            )
            /
            len(method_results),
            3
        )

    averages.append({

        "method": method,

        "n_topics": len(method_results),

        "coherence":
            average("coherence"),

        "interpretability":
            average("interpretability"),

        "specificity":
            average("specificity"),

        "overall":
            average("overall")
    })


# SAVE JSON

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


# SAVE CSV

csv_fields = [
    "method",
    "topic_id",
    "topic_label",
    "coherence",
    "interpretability",
    "specificity",
    "overall",
    "problematic_words",
    "brief_reason"
]


with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=csv_fields
    )

    writer.writeheader()

    for result in results:

        row = {
            "method":
                result["method"],

            "topic_id":
                result["topic_id"],

            "topic_label":
                result["topic_label"],

            "coherence":
                result["coherence"],

            "interpretability":
                result["interpretability"],

            "specificity":
                result["specificity"],

            "overall":
                result["overall"],

            "problematic_words":
                ", ".join(
                    result["problematic_words"]
                ),

            "brief_reason":
                result["brief_reason"]
        }

        writer.writerow(row)


# SAVE METHOD AVERAGES

average_fields = [
    "method",
    "n_topics",
    "coherence",
    "interpretability",
    "specificity",
    "overall"
]


with open(
    OUTPUT_AVERAGES,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=average_fields
    )

    writer.writeheader()
    writer.writerows(averages)


# FINAL SUMMARY

elapsed = time.time() - start_time


print()
print("=" * 70)
print("STRICT LLM EVALUATION COMPLETE")
print("=" * 70)

print(
    f"Evaluated topics: {len(results)}"
)

print(
    f"Time: {elapsed:.2f} seconds"
)

print()
print("Method averages:")
print()


for row in averages:

    print(
        f"{row['method']:20s} "
        f"Coh={row['coherence']:.2f} | "
        f"Interp={row['interpretability']:.2f} | "
        f"Spec={row['specificity']:.2f} | "
        f"Overall={row['overall']:.2f}"
    )


print()
print("Saved:")

print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_AVERAGES)

print()
print("=" * 70)