"""Dependency parsing utilities.

This module handles corpus preprocessing only:

plain-text corpus
    -> Stanza dependency parsing
    -> batch-level checkpoint cache
    -> final parquet output
    -> Corpus object

Key features
------------
- One non-empty line is treated as one sentence.
- dp() is the main entry point.
- If a parsed parquet file already exists, dp() loads it automatically.
- If parsing was interrupted, dp() resumes from cached checkpoints.
- Large corpora are parsed in configurable batches.
- Corpus identity is tracked using SHA256 file hash.
- Different input files get separate workspaces.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd


DEFAULT_COLUMNS = [
    "sent_id",
    "sentence",
    "id",
    "text",
    "lemma",
    "upos",
    "xpos",
    "feats",
    "head",
    "deprel",
    "start_char",
    "end_char",
]


# ---------------------------------------------------------------------
# Corpus object
# ---------------------------------------------------------------------

@dataclass
class Corpus:
    """Parsed corpus object returned by dp()."""

    df: pd.DataFrame
    source_path: Optional[Path] = None
    output_path: Optional[Path] = None
    workspace_dir: Optional[Path] = None
    metadata: Optional[dict[str, Any]] = None

    def preview(
        self,
        sent_id: int,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Return the parsed tokens for one sentence."""
        if columns is None:
            columns = ["id", "text", "lemma", "upos", "head", "deprel"]

        missing = [col for col in columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Unknown column(s): {missing}")

        return self.df.loc[self.df["sent_id"] == sent_id, columns].copy()

    def size(self) -> int:
        """Return the number of parsed sentences."""
        if self.df.empty:
            return 0
        return int(self.df["sent_id"].nunique())

    def save(self, path: str | Path | None = None) -> None:
        """Save the parsed DataFrame as a parquet file."""
        target = Path(path) if path is not None else self.output_path
        if target is None:
            raise ValueError("No output path specified.")

        target.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(target, index=False)
        self.output_path = target

    def cleanup_cache(self) -> None:
        """Delete the workspace cache for this corpus."""
        if self.workspace_dir is None:
            raise ValueError("No workspace_dir is associated with this Corpus.")
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def dp(
    path: str | Path,
    lang: str = "en",
    processors: str = "tokenize,pos,lemma,depparse",
    batch_size: int = 10_000,
    use_gpu: bool = False,
    stanza_pipeline=None,
    output_path: str | Path | None = None,
    workspace_root: str | Path = ".conquer",
    resume: bool = True,
    show_progress: bool = True,
    auto_cleanup: bool = False,
    force_reparse: bool = False,
    use_existing: bool = True,
) -> Corpus:
    """Prepare a parsed corpus.

    dp() automatically chooses among three modes:

    1. Load:
       If the final parquet file already exists, load it directly.

    2. Resume:
       If checkpoint files exist but final parquet does not, continue parsing
       from the first missing batch.

    3. Parse:
       If no previous output exists, start dependency parsing from scratch.

    Parameters
    ----------
    path:
        Plain-text corpus. Each non-empty line is treated as one sentence.
    lang:
        Stanza language code, e.g. "en".
    processors:
        Stanza processors to load.
    batch_size:
        Number of sentences processed per batch.
    use_gpu:
        Whether to let Stanza use GPU.
    stanza_pipeline:
        Optional preloaded Stanza pipeline.
    output_path:
        Final parquet output path. If None, defaults to:
        parsed/<input_stem>.parquet
    workspace_root:
        Root directory for hash-based parser workspaces.
    resume:
        If True, reuse completed checkpoint batches.
    show_progress:
        If True, show progress bar when tqdm is installed.
    auto_cleanup:
        If True, delete workspace cache after successful final output creation.
        Default is False.
    force_reparse:
        If True, ignore existing parquet/checkpoints and parse from scratch.
    use_existing:
        If True, load existing final parquet if available.
        If False, do not auto-load final parquet.

    Returns
    -------
    Corpus
        Parsed corpus object containing the token-level DataFrame.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {source_path}")

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if output_path is None:
        output_path_obj = Path("parsed") / f"{source_path.stem}.parquet"
    else:
        output_path_obj = Path(output_path)

    output_path_obj = output_path_obj.expanduser().resolve()

    file_hash = _file_sha256(source_path)
    short_hash = file_hash[:12]

    workspace_root_obj = Path(workspace_root).expanduser().resolve()
    workspace_dir = _workspace_dir(workspace_root_obj, file_hash)
    checkpoints_dir = workspace_dir / "checkpoints"
    metadata_path = workspace_dir / "metadata.json"

    # ------------------------------------------------------------
    # Mode 1: load existing final parquet
    # ------------------------------------------------------------
    if (
        use_existing
        and output_path_obj.exists()
        and not force_reparse
    ):
        _print_load_message(output_path_obj)
        return load_parsed(
            output_path_obj,
            source_path=source_path,
            workspace_dir=workspace_dir,
            metadata_path=metadata_path,
        )

    # ------------------------------------------------------------
    # Force reparse
    # ------------------------------------------------------------
    if force_reparse:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        if output_path_obj.exists():
            output_path_obj.unlink()

    # ------------------------------------------------------------
    # Parse or resume
    # ------------------------------------------------------------
    sentences = _read_sentences(source_path)
    total_sentences = len(sentences)

    workspace_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    metadata = _make_metadata(
        source_path=source_path,
        output_path=output_path_obj,
        file_hash=file_hash,
        lang=lang,
        processors=processors,
        batch_size=batch_size,
        total_sentences=total_sentences,
    )

    old_metadata = _read_metadata(metadata_path)

    if old_metadata is not None:
        _check_metadata_compatibility(
            old_metadata=old_metadata,
            new_metadata=metadata,
            metadata_path=metadata_path,
        )

    if total_sentences == 0:
        empty_df = pd.DataFrame(columns=DEFAULT_COLUMNS)
        corpus = Corpus(
            df=empty_df,
            source_path=source_path,
            output_path=output_path_obj,
            workspace_dir=workspace_dir,
            metadata=metadata,
        )
        corpus.save(output_path_obj)
        _write_metadata(metadata_path, metadata | {"completed": True})
        return corpus

    batch_starts = list(range(0, total_sentences, batch_size))
    completed_before = _completed_batch_starts(checkpoints_dir)

    _write_metadata(
        metadata_path,
        metadata
        | {
            "completed": False,
            "completed_batches": completed_before,
        },
    )

    _print_start_message(
        source_path=source_path,
        output_path=output_path_obj,
        workspace_dir=workspace_dir,
        total_sentences=total_sentences,
        batch_size=batch_size,
        short_hash=short_hash,
        resume=resume,
        completed_batches=len(completed_before),
        total_batches=len(batch_starts),
    )

    stanza_pipeline = _load_stanza_pipeline(
        lang=lang,
        processors=processors,
        use_gpu=use_gpu,
        stanza_pipeline=stanza_pipeline,
    )

    iterator = _progress_iterator(
        batch_starts,
        show_progress=show_progress,
        desc="Parsing corpus",
    )

    t0 = time.time()

    for start in iterator:
        end = min(start + batch_size, total_sentences)
        checkpoint_file = _checkpoint_path(checkpoints_dir, start, end)

        if resume and checkpoint_file.exists():
            continue

        batch = sentences[start:end]
        batch_df = _parse_batch(
            batch=batch,
            start_sent_id=start,
            stanza_pipeline=stanza_pipeline,
        )

        batch_df.to_parquet(checkpoint_file, index=False)

        current_metadata = _read_metadata(metadata_path) or metadata
        completed = sorted(set(current_metadata.get("completed_batches", [])) | {start})
        current_metadata["completed_batches"] = completed
        current_metadata["last_completed_batch"] = start
        current_metadata["updated_at"] = _now()
        _write_metadata(metadata_path, current_metadata)

    df = _combine_checkpoints(checkpoints_dir, expected_starts=batch_starts)

    corpus = Corpus(
        df=df,
        source_path=source_path,
        output_path=output_path_obj,
        workspace_dir=workspace_dir,
        metadata=metadata,
    )

    corpus.save(output_path_obj)

    final_metadata = _read_metadata(metadata_path) or metadata
    final_metadata["completed"] = True
    final_metadata["completed_batches"] = batch_starts
    final_metadata["output_path"] = str(output_path_obj)
    final_metadata["elapsed_seconds"] = round(time.time() - t0, 3)
    final_metadata["updated_at"] = _now()
    _write_metadata(metadata_path, final_metadata)

    if auto_cleanup:
        shutil.rmtree(workspace_dir)

    return corpus


def load_parsed(
    path: str | Path,
    source_path: str | Path | None = None,
    workspace_dir: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Corpus:
    """Load parsed corpus as a Corpus object."""
    path = Path(path).expanduser().resolve()
    df = pd.read_parquet(path)

    metadata = None
    if metadata_path is not None:
        metadata = _read_metadata(Path(metadata_path))

    return Corpus(
        df=df,
        source_path=Path(source_path).expanduser().resolve() if source_path else None,
        output_path=path,
        workspace_dir=Path(workspace_dir).expanduser().resolve() if workspace_dir else None,
        metadata=metadata,
    )


def cleanup_workspace(
    path: str | Path,
    workspace_root: str | Path = ".conquer",
) -> None:
    """Delete the hash-based workspace for a given input corpus."""
    path = Path(path).expanduser().resolve()
    file_hash = _file_sha256(path)
    workspace_dir = _workspace_dir(Path(workspace_root).expanduser().resolve(), file_hash)

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)


# ---------------------------------------------------------------------
# Internal helpers: reading, hashing, metadata
# ---------------------------------------------------------------------

def _read_sentences(path: str | Path) -> list[str]:
    """Read a line-separated corpus."""
    path = Path(path)
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    path = Path(path)
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def _workspace_dir(workspace_root: Path, file_hash: str) -> Path:
    """Return hash-based workspace directory."""
    return workspace_root / file_hash[:16]


def _make_metadata(
    source_path: Path,
    output_path: Path,
    file_hash: str,
    lang: str,
    processors: str,
    batch_size: int,
    total_sentences: int,
) -> dict[str, Any]:
    """Create parser metadata."""
    stat = source_path.stat()

    return {
        "tool": "conquer",
        "module": "parser",
        "parser": "stanza",
        "source_path": str(source_path),
        "source_name": source_path.name,
        "output_path": str(output_path),
        "file_hash": file_hash,
        "file_size_bytes": stat.st_size,
        "file_modified_time": stat.st_mtime,
        "lang": lang,
        "processors": processors,
        "batch_size": batch_size,
        "total_sentences": total_sentences,
        "completed_batches": [],
        "completed": False,
        "created_at": _now(),
        "updated_at": _now(),
    }


def _read_metadata(path: Path) -> Optional[dict[str, Any]]:
    """Read metadata JSON if it exists."""
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _check_metadata_compatibility(
    old_metadata: dict[str, Any],
    new_metadata: dict[str, Any],
    metadata_path: Path,
) -> None:
    """Ensure existing workspace belongs to the same parsing configuration."""
    keys_to_check = [
        "file_hash",
        "lang",
        "processors",
        "batch_size",
        "total_sentences",
    ]

    incompatible = []
    for key in keys_to_check:
        if old_metadata.get(key) != new_metadata.get(key):
            incompatible.append(
                {
                    "key": key,
                    "old": old_metadata.get(key),
                    "new": new_metadata.get(key),
                }
            )

    if incompatible:
        msg = [
            "Existing parser workspace has incompatible settings.",
            f"Metadata path: {metadata_path}",
            "Use force_reparse=True or change workspace_root.",
            "Differences:",
        ]

        for item in incompatible:
            msg.append(f"- {item['key']}: old={item['old']} / new={item['new']}")

        raise ValueError("\n".join(msg))


def _now() -> str:
    """Return current local timestamp."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------
# Internal helpers: Stanza parsing
# ---------------------------------------------------------------------

def _load_stanza_pipeline(
    lang: str,
    processors: str,
    use_gpu: bool,
    stanza_pipeline=None,
):
    """Load or reuse a Stanza pipeline."""
    if stanza_pipeline is not None:
        return stanza_pipeline

    try:
        import stanza
    except ImportError as exc:
        raise ImportError(
            "stanza is required for dp(). Install it with: pip install stanza"
        ) from exc

    try:
        return stanza.Pipeline(
            lang=lang,
            processors=processors,
            use_gpu=use_gpu,
            tokenize_pretokenized=False,
            verbose=False,
        )
    except Exception:
        stanza.download(lang, processors=processors, verbose=False)
        return stanza.Pipeline(
            lang=lang,
            processors=processors,
            use_gpu=use_gpu,
            tokenize_pretokenized=False,
            verbose=False,
        )


def _parse_batch(
    batch: list[str],
    start_sent_id: int,
    stanza_pipeline,
) -> pd.DataFrame:
    """Parse one batch of sentences and return a token-level DataFrame."""
    rows = []

    doc = stanza_pipeline("\n".join(batch))

    for local_i, sent in enumerate(doc.sentences):
        sent_id = start_sent_id + local_i
        original = batch[local_i] if local_i < len(batch) else sent.text

        for word in sent.words:
            rows.append(
                {
                    "sent_id": sent_id,
                    "sentence": original,
                    "id": int(word.id),
                    "text": word.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "xpos": word.xpos,
                    "feats": word.feats,
                    "head": int(word.head) if word.head is not None else None,
                    "deprel": word.deprel,
                    "start_char": getattr(word, "start_char", None),
                    "end_char": getattr(word, "end_char", None),
                }
            )

    return pd.DataFrame(rows, columns=DEFAULT_COLUMNS)


# ---------------------------------------------------------------------
# Internal helpers: checkpoints and progress
# ---------------------------------------------------------------------

def _checkpoint_path(checkpoints_dir: Path, start: int, end: int) -> Path:
    """Return checkpoint path for a batch."""
    return checkpoints_dir / f"batch_{start:09d}_{end:09d}.parquet"


def _completed_batch_starts(checkpoints_dir: Path) -> list[int]:
    """Return starts of existing checkpoint batches."""
    starts = []

    for file in checkpoints_dir.glob("batch_*_*.parquet"):
        try:
            start = int(file.name.split("_")[1])
            starts.append(start)
        except Exception:
            continue

    return sorted(set(starts))


def _combine_checkpoints(
    checkpoints_dir: Path,
    expected_starts: list[int],
) -> pd.DataFrame:
    """Combine checkpoint parquet files into one DataFrame."""
    files = []

    for start in expected_starts:
        pattern = f"batch_{start:09d}_*.parquet"
        matches = sorted(checkpoints_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Missing checkpoint for batch starting at sentence {start}."
            )

        files.append(matches[0])

    dfs = [pd.read_parquet(file) for file in files]

    if not dfs:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    return pd.concat(dfs, ignore_index=True)


def _progress_iterator(
    items: list[int],
    show_progress: bool,
    desc: str,
):
    """Return tqdm iterator when available."""
    if not show_progress:
        return items

    try:
        from tqdm.auto import tqdm

        return tqdm(items, desc=desc, unit="batch")
    except ImportError:
        return items


# ---------------------------------------------------------------------
# Internal helpers: console messages
# ---------------------------------------------------------------------

def _print_load_message(output_path: Path) -> None:
    """Print message when existing parsed corpus is loaded."""
    print("=" * 58)
    print("DP-GPT Parser")
    print("=" * 58)
    print("Existing parsed corpus found.")
    print(f"Loading      : {output_path}")
    print("=" * 58)


def _print_start_message(
    source_path: Path,
    output_path: Path,
    workspace_dir: Path,
    total_sentences: int,
    batch_size: int,
    short_hash: str,
    resume: bool,
    completed_batches: int,
    total_batches: int,
) -> None:
    """Print parser start/resume message."""
    mode = "Resume" if resume and completed_batches > 0 else "Parse"

    print("=" * 58)
    print("DP-GPT Parser")
    print("=" * 58)
    print(f"Mode        : {mode}")
    print(f"Corpus      : {source_path}")
    print(f"Sentences   : {total_sentences:,}")
    print(f"Batch size  : {batch_size:,}")
    print(f"Batches     : {completed_batches:,} / {total_batches:,} completed")
    print(f"Output      : {output_path}")
    print(f"Workspace   : {workspace_dir}")
    print(f"Hash        : {short_hash}")
    print(f"Resume      : {resume}")
    print("=" * 58)
