"""Nonsemantic control: reversible byte-level BPE, length-matched to H-SIR.

Pre-tokenization splits bytes into chunks (leading whitespace attached), so
concatenating decoded tokens always reproduces the input exactly. Merges are
learned on train documents only, stopping when mean symbols per source byte
reaches the hybrid representation's value.
"""
from __future__ import annotations

import re
from collections import Counter

_CHUNK = re.compile(rb"\s*\S+|\s+")


class BPE:
    def __init__(self):
        self.merges: list[tuple[int, int]] = []
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._rank: dict[tuple[int, int], int] = {}
        self._cache: dict[bytes, list[int]] = {}

    def fit(self, docs: list[bytes], target_symbols_per_byte: float,
            max_merges: int = 8000, max_chunks: int = 50000) -> dict:
        freq = Counter()
        for x in docs:
            freq.update(_CHUNK.findall(x))
        total_bytes = sum(len(c) * f for c, f in freq.items())
        words = [(list(c), f) for c, f in freq.most_common(max_chunks)]
        tail_bytes = total_bytes - sum(len(w) * f for w, f in words)
        symbols = total_bytes
        ratio = 1.0
        while (len(self.merges) < max_merges
               and symbols / max(1, total_bytes) > target_symbols_per_byte):
            pairs = Counter()
            for w, f in words:
                for p in zip(w, w[1:]):
                    pairs[p] += f
            if not pairs:
                break
            (l, r), cnt = max(pairs.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if cnt < 2:
                break
            new = 256 + len(self.merges)
            self.merges.append((l, r))
            self._rank[(l, r)] = len(self.merges) - 1
            self.vocab[new] = self.vocab[l] + self.vocab[r]
            saved = 0
            for w, f in words:
                i = 0
                while i < len(w) - 1:
                    if w[i] == l and w[i + 1] == r:
                        w[i:i + 2] = [new]
                        saved += f
                    i += 1
            symbols -= saved
        ratio = symbols / max(1, total_bytes)
        return {"merges": len(self.merges), "train_symbols_per_byte": ratio,
                "target_symbols_per_byte": target_symbols_per_byte,
                "untrained_tail_bytes_fraction": tail_bytes / max(1, total_bytes)}

    def _encode_chunk(self, c: bytes) -> list[int]:
        hit = self._cache.get(c)
        if hit is not None:
            return hit
        w = list(c)
        while len(w) > 1:
            best = None
            for i, p in enumerate(zip(w, w[1:])):
                rk = self._rank.get(p)
                if rk is not None and (best is None or rk < best[0]):
                    best = (rk, i)
            if best is None:
                break
            rk, i = best
            w[i:i + 2] = [256 + rk]
        if len(self._cache) < 200000:
            self._cache[c] = w
        return w

    def encode(self, x: bytes) -> list[int]:
        out: list[int] = []
        for c in _CHUNK.findall(x):
            out.extend(self._encode_chunk(c))
        return out

    def decode(self, ids: list[int]) -> bytes:
        return b"".join(self.vocab[i] for i in ids)
