"""CPU latency (single sequence, after warm-up) and parameter counts for every model."""
import os, time, json
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import numpy as np, jax, jax.numpy as jnp
import data as Dt, train as TR, run as RN

bank = Dt.Bank()
tr, va, te = Dt.loso(bank, 0, 0)
out = {}
for cfg in ['stgcn', 'dcl', 'depth', 'early', 'attn', 'xmtf', 'actionmae', 'langbind', 'opensense']:
    trn, kind = RN.build(cfg, bank, np.arange(22), 0)
    trn.norm = TR.Norm(bank, tr)
    X = trn.norm(bank.batch(te[:1]))
    if kind == 'sup':
        key = jax.random.PRNGKey(0)
        xb = TR.to_j(X)
        p = (trn.model.init({'params': key, 'dropout': key}, xb[trn.uni])['params'] if trn.uni else
             trn.model.init({'params': key, 'dropout': key}, xb, jnp.ones((1, 3)))['params'])
        f = jax.jit(lambda p, x: trn._apply(p, x, jnp.ones((1, 3)), False, key))
        f(p, xb).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(50):
            f(p, xb).block_until_ready()
        lat = (time.perf_counter() - t0) / 50
    else:
        p = trn.init(X)
        T = jnp.asarray(trn.T[trn.seen]); Mk = jnp.asarray(trn.Mk[trn.seen])
        mc = 8 if cfg == 'opensense' else 1
        g = lambda p, x: trn._fwd(p, x, jnp.ones((1, 3)), T, Mk, False, mc, jax.random.PRNGKey(0))['logits']
        f = jax.jit(g)
        xb = TR.to_j(X)
        f(p, xb).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(50):
            f(p, xb).block_until_ready()
        lat = (time.perf_counter() - t0) / 50
    out[cfg] = {'params': TR.n_params(p), 'latency_ms': 1000 * lat}
    print(cfg, out[cfg], flush=True)
json.dump(out, open('results/bench.json', 'w'), indent=1)
