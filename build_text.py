"""Extract GloVe-300 vectors for the description vocabulary and build sentence features."""
import gzip, re, numpy as np
from descriptions import DESC, LABEL_TEXT
tok = lambda s: re.findall(r"[a-z]+", s.lower())
vocab = set(w for d in DESC for s in d for w in tok(s)) | set(w for s in LABEL_TEXT for w in tok(s))
vec = {}
with gzip.open('assets/glove300.gz', 'rt', encoding='utf8') as f:
    first = f.readline()
    for ln in f:
        w, rest = ln.split(' ', 1)
        if w in vocab:
            vec[w] = np.array(rest.split(), np.float32)
missing = sorted(vocab - set(vec)); print('vocab', len(vocab), 'missing', missing)
STOP = {'the', 'a', 'an', 'and', 'of', 'to', 'in', 'with', 'is', 'as', 'at', 'it', 'from', 'then', 'while', 'one', 'after', 'other', 'each', 'if'}
def sent(s):
    ws = [w for w in tok(s) if w in vec]
    cw = [w for w in ws if w not in STOP] or ws
    M = np.stack([vec[w] for w in cw])
    return np.concatenate([M.mean(0), M.max(0)])
T = np.array([[sent(s) for s in d] for d in DESC], np.float32)  # (22,4,600)
L = np.array([sent(s) for s in LABEL_TEXT], np.float32)          # (22,600)
np.savez_compressed('cache/text.npz', T=T, L=L)
np.savez_compressed('assets/glove_subset.npz', words=np.array(sorted(vec)), vecs=np.stack([vec[w] for w in sorted(vec)]))
print(T.shape, L.shape)
# cosine similarity of class-mean text features for mirrored pairs
Tm = T.mean(1); Tm = Tm/np.linalg.norm(Tm,axis=1,keepdims=True); S = Tm@Tm.T
print('R/L high wave', S[0,1].round(3), 'R high vs R horiz', S[0,2].round(3), 'clap vs bend', S[14,15].round(3))
# token-level arrays for the attention-pooled text encoder
LMAX = 16
def toks(s):
    ws = [w for w in tok(s) if w in vec and w not in STOP][:LMAX]
    M = np.zeros((LMAX, 300), np.float32); m = np.zeros(LMAX, np.float32)
    for i, w in enumerate(ws): M[i] = vec[w]; m[i] = 1
    return M, m
TT = np.zeros((22, 4, LMAX, 300), np.float32); TM = np.zeros((22, 4, LMAX), np.float32)
for c, d in enumerate(DESC):
    for j, s in enumerate(d):
        TT[c, j], TM[c, j] = toks(s)
LT = np.zeros((22, 1, LMAX, 300), np.float32); LM = np.zeros((22, 1, LMAX), np.float32)
for c, s in enumerate(LABEL_TEXT):
    LT[c, 0], LM[c, 0] = toks(s)
np.savez_compressed('cache/text_tok.npz', TT=TT, TM=TM, LT=LT, LM=LM)
print('max tokens', TM.sum(-1).max(), LM.sum(-1).max())
