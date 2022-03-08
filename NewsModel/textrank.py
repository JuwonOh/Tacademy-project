import re
from collections import Counter, defaultdict
from operator import itemgetter

import nltk
import numpy as np
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

## textrank는 lovit의 https://lovit.github.io/nlp/2019/04/30/textrank/에서 가져왔습니다.


def nltk_tagger(input_string):
    pos_output = pos_tag(word_tokenize(input_string))
    nounpos_output = [
        each_pos
        for each_pos in pos_output
        if len(re.findall(r"NN", each_pos[1])) > 0
    ]
    return list(map(itemgetter(0), nounpos_output))


def dict_to_mat(d, n_rows, n_cols):
    rows, cols, data = [], [], []
    for (i, j), v in d.items():
        rows.append(i)
        cols.append(j)
        data.append(v)
    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols))


def cooccurrence(tokens, vocab_to_idx, window=2, min_cooccurrence=2):
    counter = defaultdict(int)
    for s, tokens_i in enumerate(tokens):
        vocabs = [vocab_to_idx[w] for w in tokens_i if w in vocab_to_idx]
        n = len(vocabs)
        for i, v in enumerate(vocabs):
            if window <= 0:
                b, e = 0, n
            else:
                b = max(0, i - window)
                e = min(i + window, n)
            for j in range(b, e):
                if i == j:
                    continue
                counter[(v, vocabs[j])] += 1
                counter[(vocabs[j], v)] += 1
    counter = {k: v for k, v in counter.items() if v >= min_cooccurrence}
    n_vocabs = len(vocab_to_idx)
    return dict_to_mat(counter, n_vocabs, n_vocabs)


def scan_vocabulary(sents, tokenize, min_count=2):
    counter = Counter(w for sent in sents for w in tokenize(sent))
    counter = {w: c for w, c in counter.items() if c >= min_count}
    idx_to_vocab = [w for w, _ in sorted(counter.items(), key=lambda x: -x[1])]
    vocab_to_idx = {vocab: idx for idx, vocab in enumerate(idx_to_vocab)}
    return idx_to_vocab, vocab_to_idx


def word_graph(
    sents, tokenize=None, min_count=2, window=2, min_cooccurrence=2
):
    idx_to_vocab, vocab_to_idx = scan_vocabulary(sents, tokenize, min_count)
    tokens = [tokenize(sent) for sent in sents]
    g = cooccurrence(tokens, vocab_to_idx, window, min_cooccurrence)
    return g, idx_to_vocab


def pagerank(x, df=0.85, max_iter=30):
    assert 0 < df < 1

    # initialize
    A = normalize(x, axis=0, norm="l1")
    R = np.ones(A.shape[0]).reshape(-1, 1)
    bias = (1 - df) * np.ones(A.shape[0]).reshape(-1, 1)

    # iteration
    for _ in range(max_iter):
        R = df * (A * R) + bias

    return R


def textrank_keyword(
    input_text,
    tokenize=nltk_tagger,
    min_count=2,
    window=5,
    min_cooccurrence=2,
    df=0.85,
    max_iter=30,
    topk=30,
):
    sents = sent_tokenize(input_text)
    g, idx_to_vocab = word_graph(
        sents, tokenize, min_count, window, min_cooccurrence
    )
    R = pagerank(g, df, max_iter).reshape(-1)
    idxs = R.argsort()[-topk:]
    keywords = [(idx_to_vocab[idx], R[idx]) for idx in reversed(idxs)]
    return keywords


from operator import itemgetter


def textrank_list_keywords(input_text, topk_length=10):
    tupled_keywords = textrank_keyword(
        sents=sent_tokenize(input_text),
        tokenize=nltk_tagger,
        min_count=2,
        window=5,
        min_cooccurrence=2,
        topk=topk_length,
    )
    return list(map(itemgetter(0), tupled_keywords))
