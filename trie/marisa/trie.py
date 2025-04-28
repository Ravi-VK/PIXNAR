import marisa_trie
import numpy as np
from typing import List, Optional, Tuple

class TrieWithContinuations:
    def __init__(self, path_prefix, eos_token_id):
        self.values = np.load(f"{path_prefix}.marisa.values.npy")
        self.offsets = np.load(f"{path_prefix}.marisa.offsets.npy")
        self.eos_token_id = eos_token_id
        self.trie = marisa_trie.RecordTrie("<Q")
        self.trie.load(f"{path_prefix}.marisa")

    def valid_continuations_direct(self, key):
        index = self.trie[key][0][0]
        return self.values[self.offsets[index] : self.offsets[index + 1]]

    def __getattr__(self, name):
        return getattr(self.trie, name)