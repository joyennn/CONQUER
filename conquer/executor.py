"""Execution layer for compiled DP-GPT queries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .compiler import CompiledQuery


# ---------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------

@dataclass
class QueryResults:
    """Results returned by apply()."""

    df: pd.DataFrame
    token_df: pd.DataFrame
    matched_ids: list[int]
    query: CompiledQuery

    @property
    def n_matches(self) -> int:
        """Return the number of matched sentences."""
        return len(self.matched_ids)

    def preview(self, n: int = 10) -> pd.DataFrame:
        """Return the first n matched sentences."""
        return self.df.head(n).copy()

    def tokens(self) -> pd.DataFrame:
        """Return all token rows for matched sentences."""
        return self.token_df.copy()

    def save_csv(self, path: str | Path) -> None:
        """Save matched sentence-level results as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(path, index=False)

    def save_tokens_csv(self, path: str | Path) -> None:
        """Save token-level results as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.token_df.to_csv(path, index=False)

    def save_txt(self, path: str | Path) -> None:
        """Save matched sentences as a plain-text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if "sentence" not in self.df.columns:
            raise ValueError("Sentence-level results do not contain a 'sentence' column.")

        path.write_text(
            "\n".join(self.df["sentence"].astype(str).tolist()),
            encoding="utf-8",
        )
    
    def summary(self, top_n: int = 10) -> dict:
        """Return summary statistics for the extraction results."""
        total_sentences = None

        if "sent_id" in self.query.plan:
            total_sentences = None

        matched_sentences = self.n_matches
        matched_tokens = int(len(self.token_df))

        avg_sentence_length = None
        if not self.token_df.empty and "sent_id" in self.token_df.columns:
            sent_lengths = self.token_df.groupby("sent_id").size()
            avg_sentence_length = float(sent_lengths.mean())

        top_lemmas = []
        if "lemma" in self.token_df.columns:
            top_lemmas = (
                self.token_df["lemma"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .to_dict()
            )

        top_upos = []
        if "upos" in self.token_df.columns:
            top_upos = (
                self.token_df["upos"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .to_dict()
            )

        top_deprels = []
        if "deprel" in self.token_df.columns:
            top_deprels = (
                self.token_df["deprel"]
                .dropna()
                .astype(str)
                .value_counts()
                .head(top_n)
                .to_dict()
            )

        return {
            "matched_sentences": matched_sentences,
            "matched_tokens": matched_tokens,
            "avg_sentence_length": avg_sentence_length,
            "top_lemmas": top_lemmas,
            "top_upos": top_upos,
            "top_deprels": top_deprels,
        }


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_dataframe(corpus_or_df: Any) -> pd.DataFrame:
    """Accept either a Corpus object or a pandas DataFrame."""
    if isinstance(corpus_or_df, pd.DataFrame):
        return corpus_or_df

    if hasattr(corpus_or_df, "df"):
        df = corpus_or_df.df
        if isinstance(df, pd.DataFrame):
            return df

    raise TypeError(
        "apply() expects a pandas DataFrame or a Corpus object with a .df attribute."
    )


def _sentence_results(df: pd.DataFrame, matched_ids: list[int]) -> pd.DataFrame:
    """Return one row per matched sentence."""
    if "sentence" in df.columns:
        out = (
            df.loc[df["sent_id"].isin(matched_ids), ["sent_id", "sentence"]]
            .drop_duplicates("sent_id")
            .reset_index(drop=True)
        )
        return out

    return pd.DataFrame({"sent_id": matched_ids})


# ---------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------

def apply(
    corpus_or_df: Any,
    compiled: CompiledQuery,
    return_tokens: bool = False,
) -> QueryResults | pd.DataFrame:
    """Apply a compiled query to a parsed corpus.

    Parameters
    ----------
    corpus_or_df:
        Corpus object returned by dp(), or a pandas DataFrame.
    compiled:
        CompiledQuery object returned by compile_plan().
    return_tokens:
        If True, return the token-level DataFrame directly for backward-style use.
        If False, return a QueryResults object.

    Returns
    -------
    QueryResults or pandas.DataFrame
        By default, returns a QueryResults object. If return_tokens=True,
        returns token rows for matched sentences.
    """
    if not isinstance(compiled, CompiledQuery):
        raise TypeError(
            "apply() now expects a CompiledQuery. "
            "Run compiled = compile_plan(corpus, plan) before apply()."
        )

    df = _get_dataframe(corpus_or_df)

    if "sent_id" not in df.columns:
        raise ValueError("Parsed corpus must contain a 'sent_id' column.")

    matched_ids: list[int] = []

    for sent_id, sent_df in df.groupby("sent_id", sort=True):
        include_ok = all(predicate(sent_df) for predicate in compiled.include)
        exclude_ok = not any(predicate(sent_df) for predicate in compiled.exclude)

        if include_ok and exclude_ok:
            matched_ids.append(sent_id)

    token_df = df.loc[df["sent_id"].isin(matched_ids)].copy()
    sentence_df = _sentence_results(df, matched_ids)

    if return_tokens:
        return token_df

    return QueryResults(
        df=sentence_df,
        token_df=token_df,
        matched_ids=matched_ids,
        query=compiled,
    )


def show_results(results: QueryResults, n: int = 10, top_n: int = 5) -> None:
    """Print a compact result summary."""
    if not isinstance(results, QueryResults):
        raise TypeError("show_results() expects a QueryResults object.")

    summary = results.summary(top_n=top_n)

    print("=" * 60)
    print("DP-GPT Query Results")
    print("=" * 60)
    print(f"Matched sentences : {summary['matched_sentences']}")
    print(f"Matched tokens    : {summary['matched_tokens']}")

    avg_len = summary.get("avg_sentence_length")
    if avg_len is not None:
        print(f"Avg sent length   : {avg_len:.2f}")

    print("-" * 60)

    print("Top lemmas:")
    if summary["top_lemmas"]:
        for lemma, count in summary["top_lemmas"].items():
            print(f"  {lemma}: {count}")
    else:
        print("  None")

    print()
    print("Top UPOS:")
    if summary["top_upos"]:
        for upos, count in summary["top_upos"].items():
            print(f"  {upos}: {count}")
    else:
        print("  None")

    print()
    print("Top dependency labels:")
    if summary["top_deprels"]:
        for deprel, count in summary["top_deprels"].items():
            print(f"  {deprel}: {count}")
    else:
        print("  None")

    print("-" * 60)

    print(f"Preview: first {n} matched sentences")
    print("-" * 60)

    if results.n_matches == 0:
        print("No matched sentences.")
        print("=" * 60)
        return

    preview = results.preview(n)

    if "sentence" in preview.columns:
        for _, row in preview.iterrows():
            print(f"[{row['sent_id']}] {row['sentence']}")
    else:
        print(preview)

    print("=" * 60)