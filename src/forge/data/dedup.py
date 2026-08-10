"""MinHash-based semantic deduplication.

Uses MinHash + LSH (Locality-Sensitive Hashing) for near-duplicate
detection. This catches paraphrases and near-copies that exact
dedup misses, which is critical for training data quality.

Requires: pip install 'forge-llm[data]' (datasketch + scikit-learn)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from rich.console import Console
from rich.progress import Progress

console = Console()

# Default shingle (n-gram) size for MinHash
DEFAULT_SHINGLE_SIZE = 5
# Default similarity threshold
DEFAULT_THRESHOLD = 0.8
# Default number of MinHash permutations
DEFAULT_NUM_PERM = 128


def shinglize(text: str, k: int = DEFAULT_SHINGLE_SIZE) -> Set[str]:
    """Convert text into a set of character k-shingles.

    Normalizes whitespace and lowercases before shingling.
    """
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def find_near_duplicates(
    records: List[Dict[str, Any]],
    text_field: str = "text",
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_size: int = DEFAULT_SHINGLE_SIZE,
) -> List[Tuple[int, int, float]]:
    """Find near-duplicate pairs using MinHash LSH.

    Args:
        records: List of data records.
        text_field: Which field to use for similarity.
        threshold: Jaccard similarity threshold (0-1).
        num_perm: Number of MinHash permutations.
        shingle_size: Size of character n-grams.

    Returns:
        List of (idx_a, idx_b, similarity) tuples.
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        console.print(
            "[red]✗ datasketch not installed.[/red]\n"
            "  Install with: pip install 'forge-llm[data]'"
        )
        return []

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: Dict[int, MinHash] = {}

    with Progress() as progress:
        task = progress.add_task("[cyan]Computing MinHash signatures...", total=len(records))

        for idx, record in enumerate(records):
            text = _extract_text(record, text_field)
            if not text:
                progress.advance(task)
                continue

            shingles = shinglize(text, shingle_size)
            m = MinHash(num_perm=num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))

            minhashes[idx] = m

            try:
                lsh.insert(str(idx), m)
            except ValueError:
                pass  # Duplicate key — already a known duplicate

            progress.advance(task)

    # Query for near-duplicates
    duplicates: List[Tuple[int, int, float]] = []
    seen_pairs: Set[Tuple[int, int]] = set()

    for idx, m in minhashes.items():
        candidates = lsh.query(m)
        for cand_str in candidates:
            cand_idx = int(cand_str)
            if cand_idx == idx:
                continue
            pair = (min(idx, cand_idx), max(idx, cand_idx))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            sim = minhashes[idx].jaccard(minhashes[cand_idx])
            if sim >= threshold:
                duplicates.append((pair[0], pair[1], round(sim, 4)))

    return sorted(duplicates, key=lambda x: -x[2])


def deduplicate(
    records: List[Dict[str, Any]],
    text_field: str = "text",
    threshold: float = DEFAULT_THRESHOLD,
    strategy: str = "keep_first",
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Remove near-duplicates from a list of records.

    Args:
        records: Input records.
        text_field: Field to compare.
        threshold: Similarity threshold.
        strategy: 'keep_first' or 'keep_longest'.

    Returns:
        (deduplicated_records, removed_indices).
    """
    duplicates = find_near_duplicates(records, text_field, threshold)

    # Build set of indices to remove
    remove_indices: Set[int] = set()
    for idx_a, idx_b, sim in duplicates:
        if strategy == "keep_longest":
            text_a = _extract_text(records[idx_a], text_field)
            text_b = _extract_text(records[idx_b], text_field)
            # Remove the shorter one
            if len(text_a or "") >= len(text_b or ""):
                remove_indices.add(idx_b)
            else:
                remove_indices.add(idx_a)
        else:
            # keep_first — remove the later occurrence
            remove_indices.add(idx_b)

    deduped = [r for i, r in enumerate(records) if i not in remove_indices]
    removed = sorted(remove_indices)

    console.print(
        f"[green]✓[/green] Dedup: {len(records)} → {len(deduped)} "
        f"(removed {len(removed)} near-duplicates at threshold {threshold})"
    )

    return deduped, removed


def _extract_text(record: Dict[str, Any], text_field: str) -> Optional[str]:
    """Extract the primary text from a record for dedup comparison."""
    # Direct field match
    if text_field in record:
        val = record[text_field]
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            # Handle messages/conversations
            return " ".join(
                str(item.get("content", item.get("value", "")))
                for item in val
                if isinstance(item, dict)
            )

    # Fallback: concatenate all string values
    parts = []
    for v in record.values():
        if isinstance(v, str):
            parts.append(v)
    return " ".join(parts) if parts else None
