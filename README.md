# CONQUER: A Construction Query Interface for Corpus Search

## Installation

CONQUER can be installed from the project repository and imported as a Python package.

```bash
pip install git+https://github.com/joyennn/CONQUER.git
```

```python
from conquer import *

set_api_key("YOUR_API_KEY")
```

---

## Corpus Processing

Users first convert raw text corpora into dependency-parsed representations.  
The `dp()` function processes an input corpus and stores the parsed representation for later query execution.

```python
corpus = dp("corpus.txt")
```

Users can inspect the processed corpus using the preview function, which displays parsed sentences with their token-level annotations.

```python
corpus.preview(15)
```

If the input corpus has already been processed, CONQUER automatically reuses the cached parsing results. Users can force the corpus to be parsed again when necessary.

```python
corpus = dp("corpus.txt", force_reparse=True)
```

The parser cache can also be removed manually.

```python
corpus.cleanup_cache()
```

Previously parsed corpora can be loaded directly without repeating the dependency parsing process.

```python
corpus = load_parsed("parsed/corpus.parquet")
```

---

## Natural Language Query Planning

Users describe target linguistic constraints using natural language.  
The `plan_query()` function converts each description into a structured query representation.

```python
plan1 = plan_query(
    "One token has lemma by."
)

plan2 = plan_query(
    "The ROOT token has an aux:pass dependent."
)

plan3 = plan_query(
    "The ROOT token has an obl:agent dependent."
)
```

CONQUER also supports user-provided external resources. Referenced files can be incorporated into the generated query representation.

```python
plan = plan_query(
    "Find the verb in the verb_list.txt.",
    files=["verb_list.txt"]
)
```

Multiple query plans can be combined into a single reusable construction definition.

```python
passive_construction = plan_set(
    name="passive",
    plans=[plan1, plan2, plan3]
)
```

Users can inspect generated query plans.

```python
show(passive_construction)
```

Generated query plans can be saved for later use.

```python
save(
    passive_construction,
    "plans/passive.json"
)
```

Previously saved query plans can be loaded directly.

```python
loaded = load("plans/passive.json")
```

---

## Query Validation

Before execution, CONQUER validates the generated query representation.

```python
report = validate(passive_construction)

show_report(report)
```

---

## Query Compilation

The validated query plan is compiled into executable search operations.

```python
compiled = compile_plan(
    corpus,
    passive_construction
)

show_code(compiled)
```

---

## Query Execution

The compiled query is applied to the parsed corpus using `apply()`.

```python
results = apply(corpus, compiled)
```

---

## Result Management

The result object stores both sentence-level and token-level retrieval outputs.

Users can obtain a compact summary of the retrieval results.

```python
results.summary()
```

Matched sentences can be inspected directly.

```python
results.preview()

results.df
```

Token-level annotations for matched sentences can also be accessed.

```python
results.tokens()
```

Results can be exported for downstream corpus analysis. Sentence-level outputs, token-level outputs, and plain-text sentence lists can be saved separately.

```python
results.save_csv(
    "outputs/passive_construction_sentences.csv"
)

results.save_tokens_csv(
    "outputs/passive_construction_tokens.csv"
)

results.save_txt(
    "outputs/passive_construction_sentences.txt"
)
```
