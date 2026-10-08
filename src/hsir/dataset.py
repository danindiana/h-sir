"""Corpus loading, privileged-field guard, deduplication, document-level split.

Only the `directions` field is read. Titles, ingredient lists, NER tags and
links are never passed to the parser (they would be privileged information
unavailable to the byte baseline).
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

FIELDS_USED = ["directions"]


@dataclass
class Doc:
    key: str
    x: bytes


def _steps(raw: str) -> list[str]:
    raw = raw.strip()
    if raw.startswith("["):
        for f in (json.loads, ast.literal_eval):
            try:
                v = f(raw)
                if isinstance(v, list):
                    return [str(s) for s in v]
            except Exception:
                pass
    return [raw]


def load(path: str, limit: int | None = None) -> tuple[list[Doc], dict]:
    """RecipeNLG CSV (column `directions`) or JSONL with `directions`."""
    docs: list[Doc] = []
    audit = Counter()
    if path.endswith(".jsonl"):
        with open(path, "rb") as f:
            for i, line in enumerate(f):
                if limit and len(docs) >= limit:
                    break
                audit["rows"] += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    audit["bad_json"] += 1
                    continue
                if rec.get("synthetic"):
                    audit["synthetic_records"] += 1
                d = rec.get("directions")
                if d is None:
                    audit["missing_directions"] += 1
                    continue
                steps = d if isinstance(d, list) else _steps(str(d))
                x = "\n".join(map(str, steps)).encode("utf-8", "surrogatepass")
                if not x.strip():
                    audit["empty"] += 1
                    continue
                docs.append(Doc(str(rec.get("link", i)), x))
    else:
        csv.field_size_limit(sys.maxsize)
        with open(path, newline="", encoding="utf-8", errors="surrogateescape") as f:
            rd = csv.DictReader(f)
            if "directions" not in (rd.fieldnames or []):
                raise ValueError(f"no 'directions' column in {path}: "
                                 f"{rd.fieldnames}")
            for i, row in enumerate(rd):
                if limit and len(docs) >= limit:
                    break
                audit["rows"] += 1
                steps = _steps(row["directions"] or "")
                x = "\n".join(steps).encode("utf-8", "surrogateescape")
                if not x.strip():
                    audit["empty"] += 1
                    continue
                docs.append(Doc(row.get("link") or str(i), x))
    audit["loaded"] = len(docs)
    return docs, dict(audit)


# ------------------------------------------------------------------ dedupe
_WS = re.compile(rb"\s+")
_WORD = re.compile(rb"[a-z0-9]+")
_P = (1 << 31) - 1  # Mersenne prime; a*v < 2^62 fits uint64


def _norm(x: bytes) -> bytes:
    return _WS.sub(b" ", x.lower()).strip()


def _shingles(x: bytes, k: int = 5) -> set[int]:
    w = _WORD.findall(x.lower())
    if len(w) < k:
        grams = [b" ".join(w)] if w else [x]
    else:
        grams = [b" ".join(w[i:i + k]) for i in range(len(w) - k + 1)]
    return {int.from_bytes(hashlib.blake2b(g, digest_size=8).digest(), "big") % _P
            for g in grams}


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def dedupe(docs: list[Doc], seed: int, threshold: float = 0.8,
           perms: int = 64, bands: int = 16) -> tuple[list[Doc], dict]:
    """Exact (normalized) + MinHash-LSH near-duplicate removal.

    Keeps one representative (lowest index) per cluster, so near-duplicates
    can never straddle the train/test split.
    """
    n = len(docs)
    uf = _UF(n)
    seen: dict[bytes, int] = {}
    exact = 0
    for i, d in enumerate(docs):
        h = hashlib.sha256(_norm(d.x)).digest()
        if h in seen:
            uf.union(seen[h], i)
            exact += 1
        else:
            seen[h] = i
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P, perms, dtype=np.uint64)[:, None]
    b = rng.integers(0, _P, perms, dtype=np.uint64)[:, None]
    rows = perms // bands
    sh = [_shingles(d.x) for d in docs]
    buckets: dict[tuple, list[int]] = {}
    for i, s in enumerate(sh):
        v = np.fromiter(s, dtype=np.uint64, count=len(s))[None, :]
        sig = ((a * v + b) % np.uint64(_P)).min(axis=1).tolist()
        for bd in range(bands):
            key = (bd, tuple(sig[bd * rows:(bd + 1) * rows]))
            buckets.setdefault(key, []).append(i)
    near = 0
    checked = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        m0 = members[0]
        for j in members[1:]:
            if (m0, j) in checked or uf.find(m0) == uf.find(j):
                continue
            checked.add((m0, j))
            inter = len(sh[m0] & sh[j])
            if inter / max(1, len(sh[m0] | sh[j])) >= threshold:
                uf.union(m0, j)
                near += 1
    keep = [docs[i] for i in range(n) if uf.find(i) == i]
    return keep, {"input_docs": n, "exact_duplicates_merged": exact,
                  "near_duplicates_merged": near, "kept_docs": len(keep),
                  "near_dup_jaccard_threshold": threshold,
                  "minhash_perms": perms, "lsh_bands": bands}


def split(docs: list[Doc], test_frac: float, seed: int):
    train, test = [], []
    for d in docs:
        h = hashlib.sha256(f"{seed}:{d.key}".encode()).digest()
        u = int.from_bytes(h[:8], "big") / 2 ** 64
        (test if u < test_frac else train).append(d)
    return train, test
