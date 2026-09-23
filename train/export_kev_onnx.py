#!/usr/bin/env python3
"""Exports a Kev run (LoRA adapter + head.pt on Qwen/Qwen3-0.6B-Base) to the exact ONNX
layout of onnx-community/kev-0.6b-ONNX, so Transformers.js and open-jev load it in a browser.

kev.model.DecisionModel.forward (vendor/kev/kev/model.py) needs marker_positions/segments
computed Python-side by kev.model.encode(). The published ONNX graph instead derives
everything from the five delimiter token ids baked into input_ids, so the graph's only
inputs are input_ids and attention_mask. This file's KevPointerONNX module reproduces that
in tensor ops, verified node-for-node against onnx-community/kev-0.6b-ONNX's own graph
(fetched and inspected with onnx):

  - segment id: Equal(input_ids, question_id) marks each branch's first token; CumSum along
    the sequence turns that into seg = 0 for the state, k for the k-th question branch
    (exactly matches the reference's single CumSum node).
  - block-causal mask: causal(i,j)=j<=i, combined with (seg[j]==seg[i] or seg[j]==0), plus
    the query's own diagonal (so a padded row is never fully masked) -- the reference's
    Equal/Or/And chain around its one CumSum, LessOrEqual and ReduceSum node.
  - RoPE positions: state tokens keep their raw index; every branch's tokens restart at
    position len(state), continuing for the branch's own length (branches reuse position
    ids -- they never attend to each other, so this doesn't collide). The reference computes
    this as len(state) + index - branch_start, where branch_start is gathered via a
    same-segment x is-branch-start-token selector matmul (its /MatMul node); this file does
    the same gather.
  - pointer readout: for every token, gather its own branch's <decide> hidden state via a
    same-segment x is-decide-token selector matmul (the reference's /MatMul_1, named
    identically here because this module's Linear submodules are also named head_q/head_k),
    project it with head_q, project every token with head_k, dot the two and scale by
    1/sqrt(head_dim) (config's "scale": 0.0625). One scalar per token; the caller reads the
    value at each option's </opt> position and softmaxes within its question.

Usage:
    python3 -m train.export_kev_onnx --run jaredpalmer/kev-0.6b \\
        --out-dir train/export/kev-0.6b-base --kev-repo vendor/kev

    python3 -m train.export_kev_onnx --run train/runs/final-kev06-mix \\
        --out-dir train/export/kev-0.6b-browser-use --kev-repo vendor/kev \\
        --repo-name arbazsiddiqui/kev-0.6b-browser-use

Produces (matching the reference's file layout; the fp32 export is an intermediate, not
published, exactly like the reference -- pass --keep-fp32 to keep it on disk for gate testing):
    config.json, tokenizer.json/tokenizer_config.json/vocab.json/merges.txt/
    special_tokens_map.json/added_tokens.json, README.md,
    onnx/model_q4.onnx(+_data), onnx/model_q4f16.onnx(+_data)
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DELIM_NAMES = ("state", "question", "option_start", "option_end", "decide")


def add_kev_repo_to_path(kev_repo):
    kev_repo = str(Path(kev_repo).resolve())
    if kev_repo not in sys.path:
        sys.path.insert(0, kev_repo)
    return kev_repo


class KevPointerONNX(nn.Module):
    """(input_ids, attention_mask) -> per-token pointer logits [batch, seq]. See module
    docstring above for the derivation; this mirrors kev.model.DecisionModel.forward for
    option_isolation=False, non-hybrid backbones (Qwen3, not Gated DeltaNet)."""

    def __init__(self, lm, head, question_id, decide_id):
        super().__init__()
        self.lm = lm
        self.head_q = head.q
        self.head_k = head.k
        self.scale = float(head.scale)
        self.temperature = float(head.temperature)
        self.register_buffer("question_id", torch.tensor(int(question_id), dtype=torch.long), persistent=False)
        self.register_buffer("decide_id", torch.tensor(int(decide_id), dtype=torch.long), persistent=False)

    def forward(self, input_ids, attention_mask):
        valid = attention_mask != 0
        B, L = input_ids.shape
        device = input_ids.device
        idx = torch.arange(L, device=device)

        is_q = (input_ids == self.question_id) & valid
        seg = torch.cumsum(is_q.long(), dim=1)                      # 0 = state, k = branch k
        is_state_key = seg == 0

        same_seg = seg.unsqueeze(2) == seg.unsqueeze(1)             # [B,L,L] (query i, key j)
        allow_seg = same_seg | is_state_key.unsqueeze(1)            # key is state, or same branch as query

        causal = idx.unsqueeze(0) <= idx.unsqueeze(1)               # [L,L] key<=query
        diag = idx.unsqueeze(0) == idx.unsqueeze(1)
        valid_key = valid.unsqueeze(1)

        allow = (causal.unsqueeze(0) & allow_seg & valid_key) | diag.unsqueeze(0)

        dtype = next(self.lm.parameters()).dtype
        neg = torch.finfo(dtype).min
        mask = torch.zeros(B, L, L, dtype=dtype, device=device).masked_fill(~allow, neg).unsqueeze(1)

        is_state = seg == 0
        arange_bl = idx.unsqueeze(0).expand(B, L)
        state_len = (valid & is_state).sum(dim=1, keepdim=True)     # [B,1]
        branch_start_sel = same_seg & is_q.unsqueeze(1)             # key j starts query i's own branch
        branch_start = (branch_start_sel.long() * idx.view(1, 1, L)).sum(dim=2)
        pos = torch.where(is_state, arange_bl, state_len + arange_bl - branch_start)

        hidden = self.lm(input_ids=input_ids, position_ids=pos, attention_mask=mask).last_hidden_state

        is_decide = (input_ids == self.decide_id) & valid
        decide_sel = (same_seg & is_decide.unsqueeze(1)).to(hidden.dtype)   # [B,L,L]
        q_h = self.head_q(hidden)
        k_h = self.head_k(hidden)
        q_gathered = torch.matmul(decide_sel, q_h)                  # each token's own branch's <decide> query vector
        logits = (k_h * q_gathered).sum(-1) * (self.scale / self.temperature)
        return logits


def load_kev_run(run, kev_repo, device="cpu"):
    """Loads run (a local run dir or a Hub id like jaredpalmer/kev-0.6b), merges its LoRA
    into the base in fp32 (kev.evaluate.load's default -- exact, what every reported kev
    number uses), and returns (tokenizer, DecisionModel, delimiter_ids dict)."""
    add_kev_repo_to_path(kev_repo)
    from kev.evaluate import load as kev_load
    from kev.model import SPECIAL

    tok, model = kev_load(run, device, dtype=torch.float32, merge=True)
    model.eval()
    delim_ids = {name: tok.convert_tokens_to_ids(t) for name, t in zip(DELIM_NAMES, SPECIAL)}
    return tok, model, delim_ids


def build_dummy_batch(tok, model, kev_repo):
    """Two short, differently-shaped encoded records (via kev.model.encode, so real delimiter
    layout) padded into one batch -- used only to trace the export; --run's real weights and
    real requests are what the gates check numerically."""
    add_kev_repo_to_path(kev_repo)
    recs = [
        {"state": "Goal: open the menu.", "questions": [
            {"instr": "Which element?", "options": ["[0] <button> \"Menu\"", "[1] <a> \"Home\""], "label": "[0] <button> \"Menu\""},
            {"instr": "Which operation?", "options": ["CLICK", "TYPE", "SELECT"], "label": "CLICK"},
        ]},
        {"state": "Goal: search for shoes and open the first result on a longer page with more context.", "questions": [
            {"instr": "Which element?", "options": ["[0] <input> \"Search\"", "[1] <button> \"Go\"", "[2] <a> \"Result 1\""], "label": "[2] <a> \"Result 1\""},
            {"instr": "Which operation?", "options": ["CLICK", "TYPE", "SELECT"], "label": "CLICK"},
        ]},
    ]
    encs = [model.encode(tok, r) for r in recs]
    L = max(len(e["ids"]) for e in encs)
    pad_id = model.pad_id
    input_ids = torch.full((len(encs), L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encs), L), dtype=torch.long)
    for b, e in enumerate(encs):
        n = len(e["ids"])
        input_ids[b, :n] = torch.tensor(e["ids"], dtype=torch.long)
        attention_mask[b, :n] = 1
    return input_ids, attention_mask


def export_fp32(model, delim_ids, tok, out_path, opset=21, kev_repo=None):
    wrapped = KevPointerONNX(model.lm, model.head, delim_ids["question"], delim_ids["decide"]).eval()
    input_ids, attention_mask = build_dummy_batch(tok, model, kev_repo)

    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (input_ids, attention_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
            opset_version=opset,
            dynamo=False,        # legacy tracer: reliable for this cumsum/gather/mask graph
            external_data=True,  # fp32 weights (~2.4 GB) exceed the 2 GiB inline proto limit
        )
    return wrapped


def optimize_fp32(fp32_path, out_path):
    """torch's traced graph carries a redundant fp32->fp32 Cast around every RMSNorm (HF's
    Qwen3RMSNorm casts to fp32 and back to preserve dtype under mixed precision; here the
    backbone is already fp32, so it's a no-op) -- ~60 of them. onnxconverter_common's float16
    pass mishandles a pre-existing Cast node that feeds a fp32-protected node (it flips the
    Cast's target dtype in place instead of bridging it), so those redundant casts must be
    gone before quantize_q4f16 runs, not just the few genuinely-needed ones. onnxruntime's
    basic graph optimizations (constant folding, identity/no-op elimination, common
    subexpression elimination -- no operator fusion) remove them: 3474 nodes down from far
    more Casts, node names otherwise preserved."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(out_path)
    ort.InferenceSession(str(fp32_path), so, providers=["CPUExecutionProvider"])
    return out_path


def _consumers_map(graph):
    consumers = {}
    for n in graph.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    return consumers


def _nodes_downstream(graph, seed_node_names):
    """Every node reachable forward (by tensor consumption) from the outputs of the named
    seed nodes, seeds included -- used to find the pointer head's full compute chain
    (head_q/head_k's projections, the decide-gather matmul, the dot product, the final
    scale) without hardcoding onnx's auto-generated node names past the seeds."""
    consumers = _consumers_map(graph)
    by_name = {n.name: n for n in graph.node}
    seen = set(seed_node_names)
    frontier = [by_name[n] for n in seed_node_names]
    while frontier:
        nxt = []
        for n in frontier:
            for out in n.output:
                for c in consumers.get(out, []):
                    if c.name not in seen:
                        seen.add(c.name)
                        nxt.append(c)
        frontier = nxt
    return seen


def _extend_with_upstream_casts(graph, protect):
    """convert_float_to_float16 flips an existing Cast node's target dtype in place instead
    of bridging it, so a Cast that already sits right before a protected node (this graph's
    decide-selector cast into the decide-gather matmul, /Cast_5 -> /MatMul) ends up emitting
    fp16 into a node kept fp32 -- a type error onnxruntime rejects at load. Ordinary
    (non-Cast) producers get an automatic bridging cast and don't need this."""
    producer = {}
    for n in graph.node:
        for o in n.output:
            producer[o] = n
    by_name = {n.name: n for n in graph.node}
    protect = set(protect)
    changed = True
    while changed:
        changed = False
        for name in list(protect):
            for inp in by_name[name].input:
                p = producer.get(inp)
                if p is not None and p.op_type == "Cast" and p.name not in protect:
                    protect.add(p.name)
                    changed = True
    return protect


def fuse_rmsnorm(fp32_path, out_path):
    """Fuses each decomposed RMSNorm (Pow -> ReduceMean -> Add -> Sqrt -> Div -> Mul) into one
    SimplifiedLayerNormalization, the op the reference graph uses. Decomposed, the Pow runs in fp16
    after the q4f16 conversion, squares Qwen3's outlier activations past 65504, and the layer turns
    to zeros on WebGPU (every option then scores 1/K). The fused op accumulates in fp32."""
    import onnx
    from onnxruntime.transformers.onnx_model import OnnxModel
    from onnxruntime.transformers.fusion_simplified_layernorm import FusionSimplifiedLayerNormalization
    m = OnnxModel(onnx.load(str(fp32_path)))
    FusionSimplifiedLayerNormalization(m).apply()
    m.prune_graph()
    left = sum(1 for n in m.model.graph.node if n.op_type == "Pow")
    fused = sum(1 for n in m.model.graph.node if n.op_type == "SimplifiedLayerNormalization")
    if left:
        raise RuntimeError(f"{left} Pow nodes left after RMSNorm fusion ({fused} fused)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save_model_to_file(str(out_path), use_external_data_format=True, all_tensors_to_one_file=True)
    return fused


def quantize_q4(fp32_path, out_path, block_size=32, bits=4, algo_config=None, is_symmetric=True):
    """N-bit weight-only quantization (MatMulNBits, GatherBlockQuantized for the embedding),
    excluding the pointer head (kept fp32, matching the reference). Default algorithm is RTN
    (round-to-nearest, MatMulNBitsQuantizer's own default); pass algo_config
    (onnxruntime.quantization.matmul_nbits_quantizer.HQQWeightOnlyQuantConfig or
    GPTQWeightOnlyQuantConfig) for the more accurate but slower alternatives -- bits/block_size
    then come from algo_config itself, not this function's arguments."""
    import onnx
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    # structure-only load (no tensor data) just to find node names by name/shape; the
    # quantizer itself gets the path below, not this object
    graph_only = onnx.load(str(fp32_path), load_external_data=False)
    head_seeds = [n.name for n in graph_only.graph.node if n.name in ("/head_q/MatMul", "/head_q/Add", "/head_k/MatMul", "/head_k/Add")]
    if len(head_seeds) != 4:
        raise RuntimeError(f"expected 4 head_q/head_k seed nodes, found {head_seeds} -- module naming changed?")
    protect = _nodes_downstream(graph_only.graph, head_seeds)
    # RoPE's cos/sin table (computed once, reused by every layer -- 14 cheap nodes) mixes an
    # unconverted fp32 constant with a converted-to-fp16 Cast in its MatMul when left to
    # onnxconverter_common's default per-node conversion; kept fp32 like the reference does.
    protect |= {n.name for n in graph_only.graph.node if "rotary_emb" in n.name}
    protect = _extend_with_upstream_casts(graph_only.graph, protect)

    kwargs = dict(nodes_to_exclude=sorted(protect), quant_format=QuantFormat.QOperator)
    if algo_config is not None:
        kwargs["algo_config"] = algo_config
    else:
        kwargs.update(bits=bits, block_size=block_size, is_symmetric=is_symmetric, op_types_to_quantize=("MatMul", "Gather"))
    # a path (not a preloaded ModelProto) so MatMulNBitsQuantizer sets self.model_path --
    # GPTQ's neural_compressor backend needs it for models over 2 GB (it writes its own
    # "<model_path>_augment.onnx" alongside for calibration bookkeeping)
    quantizer = MatMulNBitsQuantizer(str(fp32_path), **kwargs)
    quantizer.process()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(quantizer.model.model, str(out_path), save_as_external_data=True,
                     all_tensors_to_one_file=True, location=out_path.name + "_data")
    return sorted(protect)


class KevCalibrationReader:
    """CalibrationDataReader over pre-encoded Kev requests, for GPTQWeightOnlyQuantConfig."""

    def __init__(self, encodings):
        self._it = iter(encodings)

    def get_next(self):
        enc = next(self._it, None)
        if enc is None:
            return None
        import numpy as np
        ids = np.array([enc["ids"]], dtype=np.int64)
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}

    def __iter__(self):
        return self

    def __next__(self):
        result = self.get_next()
        if result is None:
            raise StopIteration
        return result


def build_calibration_encodings(tok, model, data_path, n, kev_repo):
    """n requests built the same way as training data (train/data/to_kev_format.py) from
    data_path (rows in the eval schema, e.g. data/train_m2w.jsonl -- training rows, never eval rows),
    encoded with this model's own tokenizer/encode -- GPTQ calibration input."""
    add_kev_repo_to_path(kev_repo)
    from train.data.to_kev_format import to_kev_record
    from eval.harness import load_dataset

    rows = load_dataset(data_path)[:n]
    encodings = []
    for row in rows:
        tk = to_kev_record(row)
        qs = [{"instr": tk["questions"][q]["instructions"], "options": list(tk["questions"][q]["criteria"].keys()),
               "label": tk["questions"][q]["label"]} for q in ("element", "operation")]
        encodings.append(model.encode(tok, {"state": tk["state"], "questions": qs}))
    return encodings


def quantize_q4f16(q4_path, out_path, protect_node_names):
    """Casts the rest of the (already 4-bit) graph to fp16 for WebGPU, keeping the pointer
    head's chain (protect_node_names, from quantize_q4) fp32 -- matches the reference's
    model_q4f16.onnx exactly (verified: its head_q/head_k/gather-matmul/dot-product/scale
    nodes all run wrapped in fp32 casts; only the final logits output is fp16)."""
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(q4_path))
    # onnx's own shape inference cannot type contrib ops (MatMulNBits, GatherBlockQuantized,
    # SimplifiedLayerNormalization), so the converter would skip casts at the fp32 head's boundary.
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    model = SymbolicShapeInference.infer_shapes(model, auto_merge=True, guess_output_rank=True)
    model16 = float16.convert_float_to_float16(model, node_block_list=list(protect_node_names),
                                               keep_io_types=False, disable_shape_infer=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model16, str(out_path), save_as_external_data=True,
                     all_tensors_to_one_file=True, location=out_path.name + "_data")


def build_config(base_config_dict, delim_ids, head, trained_max_state, trained_max_branch, lora_r, lora_alpha):
    cfg = dict(base_config_dict)
    cfg["architectures"] = ["Qwen3Model"]
    cfg["use_cache"] = False
    cfg["kev"] = {
        "description": "Kev typed-decision model: one state plus typed questions packed into one sequence; the logit of an option is the pointer score of its </opt> token against its question's <decide> token; softmax within each question.",
        "delimiters": {
            "state": "<|fim_prefix|>", "question": "<|fim_middle|>",
            "option_start": "<|box_start|>", "option_end": "<|box_end|>", "decide": "<|fim_suffix|>",
        },
        "delimiter_ids": delim_ids,
        "user_text_escape": "<|name|> -> <¦name¦> before tokenizing caller text, so delimiters are unforgeable",
        "max_state_tokens": 8192,
        "max_branch_tokens": 8192,
        "trained_max_state_tokens": trained_max_state,
        "trained_max_branch_tokens": trained_max_branch,
        "max_options": 255,
        "noul_options": ["no", "yes"],
        "head_dim": int(head.q.out_features),
        "scale": float(head.scale),
        "option_isolation": False,
        "lora": {"r": lora_r, "alpha": lora_alpha, "merged": True},
        "inputs": "input_ids, attention_mask (batch, seq), right padded; segments, positions and the block-causal mask are derived in-graph from the delimiter ids",
        "output": "logits (batch, seq): read the value at each option's </opt> position and softmax within the question",
    }
    cfg["transformers.js_config"] = {
        "use_external_data_format": {"model_q4f16.onnx": 1},
        "dtype": "q4f16",
    }
    return cfg


README_TEMPLATE = """---
license: apache-2.0
base_model: {repo_id_base}
base_model_relation: quantized
library_name: transformers.js
pipeline_tag: text-classification
tags: [browser-use, web-agents, decision-model, kev, onnx, transformers.js, webgpu]
---

# {repo_name}

The WebGPU build of [{repo_id_base_name}]({base_url}): merged and 4-bit, {q4f16_mb:.0f} MB, loads with open-jev and Transformers.js.
{score_line}
An independent fine-tune of [jaredpalmer/kev-0.6b](https://huggingface.co/jaredpalmer/kev-0.6b)
(Kev-0.6B, a LoRA adapter plus pointer head on `Qwen/Qwen3-0.6B-Base`). Converted the same
way as [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX),
the Transformers.js build of the base Kev-0.6B checkpoint; this repo carries the same graph
shape and delimiter contract with this fine-tune's weights.

## Usage

### open-jev

```js
import {{ OpenJev }} from "open-jev";

const jev = await OpenJev.load({{ model: "{repo_id}", dtype: "q4f16" }});
```

### Transformers.js

Kev is not a text generator and not a plain sequence classifier. One document (the
*state*) and any number of typed questions are packed into a single sequence with five
delimiter tokens; a block-causal mask lets every question see the state and itself only,
and a pointer head scores each option's `</opt>` token against its question's `<decide>`
token. The graph takes only `input_ids` and `attention_mask` (right padded) and derives
segments, positions and the mask from the delimiter ids in-graph. Its `logits` output has
one value per token: read the value at every option's `</opt>` position and softmax
within the question. `config.json` carries the delimiters, ids and limits under `kev`.

Transformers.js resolves `qwen3` to its text-generation class, which expects a KV-cache
graph, so load this graph through the base `PreTrainedModel` class: unknown or unmapped
types take the single-session encoder-only path, which feeds every graph input by name.
It logs one warning ("assuming encoder-only architecture"), which is expected.

```js
import {{ AutoTokenizer, PreTrainedModel, Tensor }} from "@huggingface/transformers";

const repo = "{repo_id}";
const tokenizer = await AutoTokenizer.from_pretrained(repo);
const model = await PreTrainedModel.from_pretrained(repo, {{ dtype: "q4f16", device: "webgpu" }});

// Caller text can never produce a delimiter: <|name|> -> <¦name¦> (kev.model.user_tokens)
const enc = (t) => Array.from(tokenizer(t.replace(/<\\|([A-Za-z0-9_]+)\\|>/g, "<¦$1¦>"), {{ add_special_tokens: false }}).input_ids.data, Number);
// delimiter ids: resolved with a direct tokenize (never through enc's escaping, which is only
// for caller text -- escaping the delimiter strings themselves tokenizes the escaped text
// instead of the real special token and silently produces a garbage encoding)
const rawId = (t) => Array.from(tokenizer(t, {{ add_special_tokens: false }}).input_ids.data, Number)[0];
const [STATE, Q, OPT, END, DECIDE] = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"].map(rawId);

const state = "Find and purchase two aisle seats for the Adele concert in Las Vegas on June 16th.";
const questions = [
  {{ instr: "Which element should be acted on next?", options: ["[0] <button> \\"Select\\"", "[1] <button> \\"Select\\"", "[8] <label> \\"Aisle seat\\""] }},
  {{ instr: "What operation should be performed?", options: ["CLICK", "TYPE", "SELECT"] }},
];

const tokens = [STATE, ...enc(state)];
const groups = questions.map((q) => {{
  const spans = q.options.map((o) => [OPT, ...enc(o), END]);
  const branch = [Q, ...enc(q.instr), ...spans.flat(), DECIDE];
  const base = tokens.length;
  let cursor = 1 + enc(q.instr).length;
  const ends = spans.map((s) => {{ cursor += s.length; return base + cursor - 1; }}); // index of each </opt>
  tokens.push(...branch);
  return ends;
}});

const {{ logits }} = await model({{
  input_ids: new Tensor("int64", BigInt64Array.from(tokens, BigInt), [1, tokens.length]),
  attention_mask: new Tensor("int64", new BigInt64Array(tokens.length).fill(1n), [1, tokens.length]),
}});
const scores = Array.from(logits.to("float32").data);
const softmax = (xs) => {{ const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b); return e.map((x) => x / s); }};
const answers = groups.map((ends) => softmax(ends.map((i) => scores[i])));
// element: argmax over the candidate options; operation: argmax over CLICK/TYPE/SELECT
```

Limits: the state is cut to 8,192 tokens and each question branch (instruction, options,
delimiters) plus the state must fit in 8,192 tokens (this fine-tune trained on states up to
{trained_max_state} tokens and branches up to {trained_max_branch}; longer inputs run but
are untested).

## Files

| file | size |
|---|---|
| `onnx/model_q4f16.onnx` + `onnx/model_q4f16.onnx_data` | {q4f16_mb:.0f} MB |

## Provenance

- Base: `Qwen/Qwen3-0.6B-Base`, LoRA r={lora_r} merged, pointer head dim {head_dim}.
- Fine-tune: [{repo_id_base}]({base_url}), a Kev-0.6B fine-tune on Mind2Web plus
  WebChain browser-action data, trained with jaredpalmer/kev's own trainer
  (`github.com/jaredpalmer/kev`).
- This repo: ONNX export only, built with `train/export_kev_onnx.py` in
  [github.com/arbazsiddiqui/kev-browser-use](https://github.com/arbazsiddiqui/kev-browser-use), following the graph layout of
  onnx-community/kev-0.6b-ONNX.

## License

Apache-2.0 for the adapter and head. The base model is Apache-2.0 (Qwen3).
"""


def _ensure_tokenizer_sidecars(out_dir, source_repo="jaredpalmer/kev-0.6b"):
    """AutoTokenizer.from_pretrained(Qwen/Qwen3-0.6B-Base).save_pretrained() only emits
    tokenizer.json/tokenizer_config.json/chat_template.jinja; every Kev-0.6B checkpoint's own
    Hub repo (whichever run this export came from, or the base) also ships
    vocab.json/merges.txt/added_tokens.json/special_tokens_map.json -- content-identical
    across every Kev-0.6B derivative since none of them touch the vocabulary. Fetched from
    the upstream base checkpoint so the file set matches the reference layout exactly."""
    from huggingface_hub import hf_hub_download
    for fname in ("vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json"):
        dest = out_dir / fname
        if dest.exists():
            continue
        try:
            src = hf_hub_download(source_repo, fname)
        except Exception as e:
            print(f"tokenizer sidecar {fname} not found on {source_repo}, skipping: {e}")
            continue
        shutil.copy(src, dest)


def write_readme(out_dir, repo_id, repo_id_base, trained_max_state, trained_max_branch, lora_r, head_dim, q4f16_mb,
                  step_success=None, fp32_step_success=None):
    score_line = ""
    if step_success is not None and fp32_step_success is not None:
        score_line = (f"\nScores {step_success:.1f}% step success on the same 1,000-step Mind2Web eval "
                      f"(the full-precision model: {fp32_step_success:.1f}).\n")
    text = README_TEMPLATE.format(
        repo_id=repo_id, repo_id_base=repo_id_base, base_url=f"https://huggingface.co/{repo_id_base}",
        repo_name=repo_id.rpartition("/")[2], repo_id_base_name=repo_id_base.rpartition("/")[2],
        trained_max_state=trained_max_state, trained_max_branch=trained_max_branch,
        lora_r=lora_r, head_dim=head_dim, q4f16_mb=q4f16_mb, score_line=score_line,
    )
    (out_dir / "README.md").write_text(text)


def export(run, out_dir, kev_repo, opset=21, repo_id=None, repo_id_base=None,
           trained_max_state=384, trained_max_branch=1024, keep_fp32=False,
           quant_algo="rtn", gptq_calib_data=None, gptq_calib_n=256, block_size=32, is_symmetric=True,
           step_success=None, fp32_step_success=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    tok, model, delim_ids = load_kev_run(run, kev_repo, device="cpu")

    # torch.onnx's external-data path scatters one file per large initializer (not a single
    # blob), so this traces into its own scratch dir -- never out_dir -- and only the
    # consolidated q4/q4f16 files (single _data blob each, via onnx.save_model below) land
    # in the published layout.
    fp32_dir = Path(tempfile.mkdtemp(prefix="kev_fp32_"))
    fp32_path = fp32_dir / "model_fp32.onnx"
    export_fp32(model, delim_ids, tok, fp32_path, opset=opset, kev_repo=kev_repo)
    fp32_mb = sum(f.stat().st_size for f in fp32_dir.rglob("*") if f.is_file()) / 1e6
    print(f"fp32 export: {fp32_mb:.1f} MB in {fp32_dir}")

    # same directory: the optimized graph's external-data references are relative, and only
    # resolve alongside the scattered per-tensor files torch.onnx.export wrote here
    fp32_opt_path = fp32_dir / "model_fp32_opt.onnx"
    optimize_fp32(fp32_path, fp32_opt_path)

    algo_config = None
    if quant_algo == "hqq":
        from onnxruntime.quantization.matmul_nbits_quantizer import HQQWeightOnlyQuantConfig
        algo_config = HQQWeightOnlyQuantConfig(block_size=32, bits=4, op_types_to_quantize=("MatMul", "Gather"))
    elif quant_algo == "gptq":
        from onnxruntime.quantization.matmul_nbits_quantizer import GPTQWeightOnlyQuantConfig
        encodings = build_calibration_encodings(tok, model, gptq_calib_data, gptq_calib_n, kev_repo)
        algo_config = GPTQWeightOnlyQuantConfig(calibration_data_reader=KevCalibrationReader(encodings),
                                                 block_size=32, op_types_to_quantize=("MatMul", "Gather"))
    elif quant_algo != "rtn":
        raise ValueError(f"unknown quant_algo {quant_algo!r}, expected rtn/hqq/gptq")

    q4_path = onnx_dir / "model_q4.onnx"
    fused_path = fp32_opt_path.with_name("model_fp32_fused.onnx")
    n_fused = fuse_rmsnorm(fp32_opt_path, fused_path)
    print(f"fused {n_fused} RMSNorms into SimplifiedLayerNormalization")
    protect = quantize_q4(fused_path, q4_path, algo_config=algo_config, block_size=block_size, is_symmetric=is_symmetric)
    q4_mb = sum(f.stat().st_size for f in onnx_dir.glob("model_q4.onnx*")) / 1e6
    print(f"model_q4.onnx: {q4_mb:.1f} MB ({quant_algo}, block {block_size}, {'symmetric' if is_symmetric else 'asymmetric'}), pointer head protected nodes: {protect}")

    q4f16_path = onnx_dir / "model_q4f16.onnx"
    quantize_q4f16(q4_path, q4f16_path, protect)
    q4f16_mb = sum(f.stat().st_size for f in onnx_dir.glob("model_q4f16.onnx*")) / 1e6
    print(f"model_q4f16.onnx: {q4f16_mb:.1f} MB")

    if keep_fp32:
        dest = out_dir / "_fp32"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(fp32_dir), str(dest))
        print(f"kept fp32 export at {dest}")
    else:
        shutil.rmtree(fp32_dir)

    tok.save_pretrained(out_dir)
    _ensure_tokenizer_sidecars(out_dir)

    add_kev_repo_to_path(kev_repo)
    from kev.evaluate import resolve_run
    resolved = resolve_run(run)
    adapter_cfg = json.loads((Path(resolved) / "adapter_config.json").read_text())
    lora_r, lora_alpha = adapter_cfg["r"], adapter_cfg["lora_alpha"]

    base_cfg = model.lm.config.to_dict()  # model.lm is the merged backbone (plain Qwen3Model)
    cfg = build_config(base_cfg, delim_ids, model.head, trained_max_state, trained_max_branch,
                        lora_r=lora_r, lora_alpha=lora_alpha)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    repo_id = repo_id or "arbazsiddiqui/kev-0.6b-browser-use-ONNX"
    repo_id_base = repo_id_base or "arbazsiddiqui/kev-0.6b-browser-use"
    write_readme(out_dir, repo_id, repo_id_base, trained_max_state, trained_max_branch,
                 lora_r=lora_r, head_dim=int(model.head.q.out_features), q4f16_mb=q4f16_mb,
                 step_success=step_success, fp32_step_success=fp32_step_success)

    print(f"wrote {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="local Kev run dir or a Hub id (e.g. jaredpalmer/kev-0.6b)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kev-repo", required=True, help="path to a jaredpalmer/kev checkout (for kev.model/kev.evaluate)")
    ap.add_argument("--opset", type=int, default=21,
                     help="MatMulNBitsQuantizer rewrites opset_import to 21 regardless (GatherBlockQuantized "
                          "needs it); exporting below 18 leaves ReduceMean's axes as an attribute, which onnxruntime "
                          "then rejects under the bumped opset (axes moved to an input in ReduceMean-18). 21 avoids the mismatch.")
    ap.add_argument("--repo-id", default=None, help="HF repo id for the ONNX build (README)")
    ap.add_argument("--repo-id-base", default=None, help="HF repo id of the fine-tune this is built from (README)")
    ap.add_argument("--trained-max-state", type=int, default=384)
    ap.add_argument("--trained-max-branch", type=int, default=1024)
    ap.add_argument("--keep-fp32", action="store_true", help="keep the intermediate fp32 ONNX (for gate testing)")
    ap.add_argument("--quant-algo", choices=["rtn", "hqq", "gptq"], default="rtn",
                     help="4-bit weight quantization algorithm for model_q4.onnx/model_q4f16.onnx (default: "
                          "round-to-nearest). hqq needs no calibration data; gptq needs --gptq-calib-data.")
    ap.add_argument("--gptq-calib-data", default=None, help="JSONL of training rows (e.g. data/train_m2w.jsonl) for --quant-algo gptq")
    ap.add_argument("--gptq-calib-n", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=32, help="block size for --quant-algo rtn (ignored for hqq/gptq, hardcoded 32)")
    ap.add_argument("--asymmetric", action="store_true",
                     help="RTN with a zero-point (asymmetric) instead of symmetric quantization -- higher 1k eval "
                          "step success on our checkpoint (30.4 vs 29.4 at block 32) for +10 MB; this is what "
                          "arbazsiddiqui/kev-0.6b-browser-use-ONNX ships")
    ap.add_argument("--step-success", type=float, default=None, help="this build's 1k-eval step success %%, for the README's one-line score")
    ap.add_argument("--fp32-step-success", type=float, default=None, help="the fp32 export's 1k-eval step success %%, for the same line")
    args = ap.parse_args()
    export(args.run, args.out_dir, args.kev_repo, opset=args.opset, repo_id=args.repo_id,
           repo_id_base=args.repo_id_base, trained_max_state=args.trained_max_state,
           trained_max_branch=args.trained_max_branch, keep_fp32=args.keep_fp32,
           quant_algo=args.quant_algo, gptq_calib_data=args.gptq_calib_data, gptq_calib_n=args.gptq_calib_n,
           block_size=args.block_size, is_symmetric=not args.asymmetric,
           step_success=args.step_success, fp32_step_success=args.fp32_step_success)


if __name__ == "__main__":
    main()
