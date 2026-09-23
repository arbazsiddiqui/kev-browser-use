"""Adapter over Kev-9B (jaredpalmer/kev), Qwen3.5-9B-Base LoRA + pointer head, Gated
DeltaNet layers (needs flash-linear-attention on CUDA).

Kev has no pip package -- it's installed from source with `uv sync --extra serve`
inside its own repo checkout, which creates its own `.venv` there (Python 3.12+,
pinned deps that may not match the shared harness venv). Rather than fight that, this
adapter spawns `python -m kev.serve` as a subprocess using *Kev's own* venv interpreter
and talks to it over HTTP -- the same `/v1/systemone` wire contract as jev.py, so most
of this file is jev.py's HTTP client pointed at localhost instead of the TypeSafe API,
with no auth header. Needs a CUDA GPU for the server.

    KEV_REPO_DIR=vendor/kev KEV_RUN=jaredpalmer/kev-9b \
        python3 eval/harness.py run --adapter kev --data <path> --dataset-name <name>
"""
import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

from eval.render import render_candidate, render_state

_OPERATIONS = ("CLICK", "TYPE", "SELECT")
_MAX_RETRIES = 5
_TIMEOUT_S = int(os.environ.get("KEV_TIMEOUT_S", "600"))  # queued requests on a busy server exceeded 60 s
_READY_TIMEOUT_S = 600

_PROC = None
_PORT = None


def _server():
    global _PROC, _PORT
    if _PROC is None:
        repo_dir = os.environ.get("KEV_REPO_DIR")
        if not repo_dir:
            raise RuntimeError("KEV_REPO_DIR not set -- clone jaredpalmer/kev and `uv sync --extra serve` first")
        venv_python = os.path.join(repo_dir, ".venv", "bin", "python")
        if not os.path.exists(venv_python):
            raise RuntimeError(f"no venv python at {venv_python} -- run `uv sync --extra serve` in {repo_dir}")
        _PORT = int(os.environ.get("KEV_PORT", "8009"))
        run_target = os.environ.get("KEV_RUN", "jaredpalmer/kev-9b")
        # a server left over from an earlier run would answer our health check and every request
        # afterwards, silently scoring the wrong model (this happened on 2026-09-22)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/v1/models", timeout=3) as resp:
                raise RuntimeError(
                    f"port {_PORT} already serves a kev model ({resp.read(300)!r}); kill it or set KEV_PORT"
                )
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        env = dict(os.environ)
        env.setdefault("KEV_DTYPE", "bf16")
        # the server logs every request; an undrained PIPE fills after a few hundred rows and
        # blocks the server mid-write, which looked like a request hang (2026-09-22)
        _LOG = open(f"/tmp/kev_serve_{_PORT}.log", "ab")
        _PROC = subprocess.Popen(
            [venv_python, "-m", "kev.serve", "--run", run_target, "--port", str(_PORT)],
            cwd=repo_dir,
            env=env,
            stdout=_LOG,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + _READY_TIMEOUT_S
        last_err = None
        while time.time() < deadline:
            if _PROC.poll() is not None:
                out = open(f"/tmp/kev_serve_{_PORT}.log", "rb").read().decode("utf-8", "replace")
                raise RuntimeError(f"kev.serve exited early (code {_PROC.returncode}): {out[-2000:]}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/v1/models", timeout=5) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError) as e:
                last_err = e
            time.sleep(2.0)
        else:
            raise RuntimeError(f"kev.serve did not become ready within {_READY_TIMEOUT_S}s: {last_err}")
        atexit.register(_shutdown)
        _warm_up(_PORT)
    return _PORT


def _shutdown():
    if _PROC is not None and _PROC.poll() is None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _PROC.kill()


def _warm_up(port):
    """The first request after start can take minutes (kernel compilation, lazy weight load);
    absorb it here with a long timeout so the harness's per-request timeout only sees warm calls."""
    body = {"state": "warm up", "questions": {"q": {"type": "choice", "instructions": "pick one",
            "criteria": {"a": None, "b": None}}}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        urllib.request.urlopen(req, timeout=900).read()
    except urllib.error.HTTPError:
        pass  # a 4xx on the toy body still proves the model answered


def _request(port, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/systemone",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    delay = 1.0
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            body_text = e.read().decode("utf-8", "replace") if hasattr(e, "read") else ""
            if e.code in (429, 503) and attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise RuntimeError(f"kev.serve HTTP {e.code}: {body_text[:500]}") from e
        except urllib.error.URLError as e:
            last_err = e
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise last_err


def _body(row, order):
    """order: the candidate indices in the sequence the model sees. The criteria dict is keyed by the
    real idx either way, so probabilities map back without bookkeeping."""
    element_criteria = {str(row["candidates"][i]["idx"]): render_candidate(row["candidates"][i]) for i in order}
    return {
        "state": render_state(row),
        "model": "kev-latest",
        "questions": {
            "element": {
                "type": "choice",
                "instructions": "Given the goal and previous actions, which candidate element should be acted on next?",
                "criteria": element_criteria,
            },
            "operation": {
                "type": "choice",
                "instructions": "What operation should be performed on the target element?",
                "criteria": {
                    "CLICK": "Click the element",
                    "TYPE": "Type text into the element",
                    "SELECT": "Select an option from the element",
                },
            },
        },
    }


def _element_probs(result, row):
    probs_by_key = result["answers"]["element"]["probabilities"]
    probs = [float(probs_by_key.get(str(c["idx"]), 0.0)) for c in row["candidates"]]
    total = sum(probs)
    return [p / total for p in probs] if total > 0 else probs


def _train_format(row):
    """The exact request text our Kev fine-tunes trained on (train/data/to_kev_format.py): compact
    state, short option labels, no descriptions. Returns (body, labels in candidate order)."""
    from train.data.to_kev_format import candidate_label, to_kev_record
    rec = to_kev_record({**row, "gold_idx": 0, "operation": "CLICK"})
    for q in rec["questions"].values():
        q.pop("label", None)
    rec.pop("_meta", None)
    rec["model"] = "kev-latest"
    return rec, [candidate_label(c) for c in row["candidates"]]


def _predict_train_format(row, port):
    n = len(row["candidates"])
    variants = [(row, list(range(n)))]
    if os.environ.get("KEV_TWO_ORDER") == "1":
        rev = dict(row)
        rev["candidates"] = [dict(c, idx=j) for j, c in enumerate(reversed(row["candidates"]))]
        variants.append((rev, list(reversed(range(n)))))
    t0 = time.perf_counter()
    elem = [0.0] * n
    op_acc = {op: 0.0 for op in _OPERATIONS}
    for r, back in variants:
        body, labels = _train_format(r)
        res = _request(port, body)
        pe = res["answers"]["element"]["probabilities"]
        probs = [float(pe.get(lab, 0.0)) for lab in labels]
        total = sum(probs) or 1.0
        for j, pr in enumerate(probs):
            elem[back[j]] += pr / total / len(variants)
        po = res["answers"]["operation"]["probabilities"]
        tot = sum(float(po.get(op, 0.0)) for op in _OPERATIONS) or 1.0
        for op in _OPERATIONS:
            op_acc[op] += float(po.get(op, 0.0)) / tot / len(variants)
    return {"element_probs": elem, "operation_probs": op_acc,
            "latency_ms": (time.perf_counter() - t0) * 1000}


def predict(row):
    port = _server()
    if os.environ.get("KEV_FORMAT") == "train":
        return _predict_train_format(row, port)
    n = len(row["candidates"])
    orders = [list(range(n))]
    if os.environ.get("KEV_TWO_ORDER") == "1":
        # the same candidates presented in reverse: averaging the two scorings cancels the position
        # bias a single ordering leaves behind
        orders.append(list(reversed(range(n))))

    t0 = time.perf_counter()
    results = [_request(port, _body(row, o)) for o in orders]
    latency_ms = (time.perf_counter() - t0) * 1000
    result = results[0]

    per_order = [_element_probs(r, row) for r in results]
    element_probs = [sum(p[i] for p in per_order) / len(per_order) for i in range(n)]

    op_probs_raw = result["answers"]["operation"]["probabilities"]
    operation_probs = {op: float(op_probs_raw.get(op, 0.0)) for op in _OPERATIONS}
    s2 = sum(operation_probs.values())
    if s2 > 0:
        operation_probs = {k: v / s2 for k, v in operation_probs.items()}

    return {
        "element_probs": element_probs,
        "operation_probs": operation_probs,
        "latency_ms": latency_ms,
        "usage": result.get("usage") or {},
    }
