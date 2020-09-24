# -*- coding: utf-8 -*-

import torch
import numpy as np


class Embedding(object):

    def __init__(self, tokens, vectors, lang, unk=None):
        super(Embedding, self).__init__()

        self.tokens = tokens
        self.vectors = torch.tensor(vectors)
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

        # fasttext embedding transform for multilingul
        transform = "multilingual_trans/alignment_matrices/" + lang + ".txt"
        transmat = torch.tensor(np.loadtxt(transform), dtype=torch.float) if isinstance(transform, str) else transform
        # print(self.vectors)
        self.vectors = torch.matmul(self.vectors, transmat)

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    @property
    def dim(self):
        return self.vectors.size(1)

    @property
    def unk_index(self):
        if self.unk is not None:
            return self.tokens.index(self.unk)
        else:
            raise AttributeError

    @classmethod
    def load(cls, path, lang, unk=None):
        with open(path, 'r') as f:
            lines = [line for line in f]
                
        splits = [line.split() for line in lines]
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
                                for s in splits])

        return cls(tokens, vectors, lang, unk=unk)
