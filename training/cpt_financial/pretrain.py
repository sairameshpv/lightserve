"""Continued pretraining: Llama-3-8B-Instruct reads part of the Corpus-Prep library (SEC 10-K
text, training/corpus_prep/) with LoRA, so it gets more fluent in financial writing.

The library is flat uint32 token files with <|end_of_text|> between reports. Training reads
fixed-length "pages" (windows) from random places in them.
"""

import numpy as np


class PageDataset:
    """`num_pages` windows of `page_len` tokens, from random places across `paths` (a file's chance
    is proportional to its size), fixed by `seed` so a run can be repeated. Files are memory-mapped:
    nothing is read until a page is asked for. A page may run across the end of one report into the
    next; the end-of-document token between them marks the boundary (the standard approach)."""

    def __init__(self, paths, page_len: int, num_pages: int, seed: int):
        self.files = [np.memmap(p, dtype=np.uint32, mode="r") for p in paths]
        self.page_len = page_len
        room = np.array([len(f) - page_len + 1 for f in self.files])  # possible start positions
        rng = np.random.default_rng(seed)
        self.which = rng.choice(len(self.files), size=num_pages, p=room / room.sum())
        self.start = rng.integers(0, room[self.which])

    def __len__(self):
        return len(self.which)

    def __getitem__(self, i):
        s = self.start[i]
        ids = np.asarray(self.files[self.which[i]][s:s + self.page_len], dtype=np.int64)
        return {"input_ids": ids, "labels": ids.copy()}  # the model shifts labels by one itself
