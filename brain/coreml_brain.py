"""
brain/coreml_brain.py — batched recurrent neural forward pass
==============================================================
Elman RNN — each organism carries a hidden state across ticks:

    inputs  (MAX_POP, N_INPUTS)
    W1      (MAX_POP, N_INPUTS, N_HIDDEN)
    W2      (MAX_POP, N_HIDDEN, N_OUTPUTS)
    h_prev  (MAX_POP, N_HIDDEN)            ← recurrent state
    →
    h_new   (MAX_POP, N_HIDDEN)            ← stored back into pop
    out     (MAX_POP, N_OUTPUTS)

h_new = tanh(inputs @ W1 + h_prev)   recurrent connection
out   = tanh(h_new @ W2)

The weights evolve what to write into h and how to read from it.
Fear, momentum, hunger anticipation — whatever helps survival emerges.
Falls back to numpy einsum if CoreML unavailable.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    _HAS_CT = True
except ImportError:
    _HAS_CT = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH   = PROJECT_ROOT / "build" / "brain.mlpackage"
META_PATH    = PROJECT_ROOT / "build" / "brain_meta.json"

_model      = None
_max_pop    = 0
_n_in       = 0
_n_hid      = 0
_n_out      = 0
_use_coreml = False


def init_brain(max_pop: int, n_inputs: int, n_hidden: int, n_outputs: int) -> bool:
    global _model, _max_pop, _n_in, _n_hid, _n_out, _use_coreml

    _max_pop = max_pop
    _n_in    = n_inputs
    _n_hid   = n_hidden
    _n_out   = n_outputs

    if not _HAS_CT:
        print("[Brain] coremltools not available — numpy fallback")
        return False

    if MODEL_PATH.exists() and META_PATH.exists():
        try:
            meta = json.loads(META_PATH.read_text())
            if (meta.get("max_pop") == max_pop and meta.get("n_in")  == n_inputs
                    and meta.get("n_hid") == n_hidden and meta.get("n_out") == n_outputs
                    and meta.get("recurrent") == True and meta.get("has_wh") == True
                    and meta.get("has_bias") == True):
                _model = ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL)
                _use_coreml = True
                print(f"[Brain] Loaded cached model ({MODEL_PATH.name})")
                return True
        except Exception as e:
            print(f"[Brain] Cache load failed ({e}), rebuilding...")

    print(f"[Brain] Compiling recurrent brain model "
          f"(pop={max_pop}, {n_inputs}→{n_hidden}→{n_outputs})...", end="", flush=True)
    t0 = time.time()
    try:
        @mb.program(input_specs=[
            mb.TensorSpec(shape=(max_pop, n_inputs)),            # x
            mb.TensorSpec(shape=(max_pop, n_inputs,  n_hidden)), # W1
            mb.TensorSpec(shape=(max_pop, n_hidden,  n_outputs)),# W2
            mb.TensorSpec(shape=(max_pop, n_hidden,  n_hidden)), # Wh
            mb.TensorSpec(shape=(max_pop, n_hidden)),            # h_prev
            mb.TensorSpec(shape=(max_pop, n_hidden)),            # b1
            mb.TensorSpec(shape=(max_pop, n_outputs)),           # b2
        ])
        def brain_prog(x, W1, W2, Wh, h_prev, b1, b2):
            # x: (B, I) → (B, 1, I) for batched matmul
            x_e    = mb.expand_dims(x=x, axes=[-2])                                      # (B, 1, I)
            h_raw  = mb.squeeze(x=mb.matmul(x=x_e, y=W1), axes=[-2])                    # (B, H)
            hp_e   = mb.expand_dims(x=h_prev, axes=[-2])                                 # (B, 1, H)
            wh_raw = mb.squeeze(x=mb.matmul(x=hp_e, y=Wh), axes=[-2])                   # (B, H)
            h_new  = mb.tanh(x=mb.add(x=mb.add(x=h_raw, y=wh_raw), y=b1), name='h_new') # (B, H)
            h_e    = mb.expand_dims(x=h_new, axes=[-2])                                  # (B, 1, H)
            out_raw = mb.squeeze(x=mb.matmul(x=h_e, y=W2), axes=[-2])                   # (B, O)
            out    = mb.tanh(x=mb.add(x=out_raw, y=b2), name='out')
            return h_new, out

        model = ct.convert(brain_prog,
                           compute_units=ct.ComputeUnit.ALL,
                           minimum_deployment_target=ct.target.macOS13)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        model.save(str(MODEL_PATH))
        META_PATH.write_text(json.dumps(
            {"max_pop": max_pop, "n_in": n_inputs, "n_hid": n_hidden,
             "n_out": n_outputs, "recurrent": True, "has_wh": True, "has_bias": True}))
        _model = ct.models.MLModel(str(MODEL_PATH), compute_units=ct.ComputeUnit.ALL)
        _use_coreml = True
        print(f" done ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:
        print(f" FAILED: {e} — numpy fallback")
        return False


def run_brain(x: np.ndarray, W1: np.ndarray, W2: np.ndarray, Wh: np.ndarray,
              b1: np.ndarray, b2: np.ndarray,
              h_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Recurrent forward pass for all organisms.
    x:      (N, N_INPUTS)
    W1:     (N, N_INPUTS, N_HIDDEN)
    W2:     (N, N_HIDDEN, N_OUTPUTS)
    Wh:     (N, N_HIDDEN, N_HIDDEN)
    b1:     (N, N_HIDDEN)
    b2:     (N, N_OUTPUTS)
    h_prev: (N, N_HIDDEN)
    →  h_new (N, N_HIDDEN),  out (N, N_OUTPUTS)
    """
    if x.shape[0] == 0:
        return np.empty((0, _n_hid), dtype=np.float32), np.empty((0, _n_out), dtype=np.float32)

    if _use_coreml and _model is not None:
        n = x.shape[0]
        h_chunks, o_chunks = [], []
        for s in range(0, n, _max_pop):
            e       = min(s + _max_pop, n)
            chunk   = e - s
            xi      = x[s:e].astype(np.float32)
            W1i     = W1[s:e].astype(np.float32)
            W2i     = W2[s:e].astype(np.float32)
            Whi     = Wh[s:e].astype(np.float32)
            b1i     = b1[s:e].astype(np.float32)
            b2i     = b2[s:e].astype(np.float32)
            hi      = h_prev[s:e].astype(np.float32)
            if chunk < _max_pop:
                pad    = _max_pop - chunk
                xi     = np.vstack([xi,  np.zeros((pad, _n_in),            dtype=np.float32)])
                W1i    = np.concatenate([W1i, np.zeros((pad, _n_in, _n_hid),   dtype=np.float32)])
                W2i    = np.concatenate([W2i, np.zeros((pad, _n_hid, _n_out),  dtype=np.float32)])
                Whi    = np.concatenate([Whi, np.zeros((pad, _n_hid, _n_hid),  dtype=np.float32)])
                b1i    = np.vstack([b1i, np.zeros((pad, _n_hid),            dtype=np.float32)])
                b2i    = np.vstack([b2i, np.zeros((pad, _n_out),            dtype=np.float32)])
                hi     = np.vstack([hi,  np.zeros((pad, _n_hid),            dtype=np.float32)])
            r = _model.predict({"x": xi, "W1": W1i, "W2": W2i, "Wh": Whi,
                                 "h_prev": hi, "b1": b1i, "b2": b2i})
            h_chunks.append(r['h_new'][:chunk])
            o_chunks.append(r['out'][:chunk])
        return np.vstack(h_chunks), np.vstack(o_chunks)

    # numpy fallback
    h_new = np.tanh(np.einsum('bi,bih->bh', x, W1) + np.einsum('bh,bhk->bk', h_prev, Wh) + b1)
    out   = np.tanh(np.einsum('bh,bho->bo', h_new, W2) + b2)
    return h_new, out
