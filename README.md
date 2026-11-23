# LittleBitChat


## Get the actual pretrained Gemma 3 teacher — one shard, not the whole model

The previous example `D:\Models\gemma-3-27b-it` was a placeholder and is not automatically installed. To train our own compressed model, we need the **original pretrained dense Gemma** weights—not a preexisting LittleBit checkpoint.

Gemma 3 27B is hosted by Google on Hugging Face, with license-gated access. First, sign in at https://huggingface.co/google/gemma-3-27b-it and personally **accept Google's license**, then run:

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main

$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements-teacher.txt
& "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\hf.exe" auth login

# Preview the exact original shard and its size; downloads only the JSON index:
& $py littlebitchat.py train acquire --layer 0 --projection self_attn.q_proj --out models\gemma-3-27b-it

# Explicitly authorize a single-shard download (potentially several GB):
& $py littlebitchat.py train acquire --layer 0 --projection self_attn.q_proj --out models\gemma-3-27b-it --download

# Scan then train one real projection from the local original:
& $py littlebitchat.py train inventory --checkpoint models\gemma-3-27b-it --layer-start 0 --layer-end 1 --projections self_attn.q_proj
& $py littlebitchat.py train batch --checkpoint models\gemma-3-27b-it --layer-start 0 --layer-end 1 --projections self_attn.q_proj --rank 32 --steps 50 --device cpu --max-jobs 1 --out training_runs\gemma-first
~~~

The script pins the official model revision and refuses an unknown-size or over-budget weight shard. It **does not** download the other 11 shards automatically or bypass the model license. [Full original teacher setup and limitations](docs/ORIGINAL_GEMMA_TEACHER.md).

## Training quality guard and rank preview

Real Gemma 3 layer-0 q_proj at rank 32 produced **0.04134 bits/weight and normalized holdout RMSE 0.89285**, with holdout error increasing from 0.63038 to 0.66471. The packed representation was correct, but its teacher approximation was poor.

LittleBitChat now selects the **best validation-scoring snapshot** rather than automatically exporting the last optimizer step. It keeps optimizer and best model states separately for resumable QAT, audits the selection on independent held-out random probes, and rejects numeric quality promotion when error regresses or packed inference parity fails. **This is still not a measurement of language-model quality.**

Preview which ranks produce which *estimated physical compression cost* without reading giant weight tensors or doing training:

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py littlebitchat.py train rank-plan --checkpoint models\gemma-3-27b-it --layer 0 --projection self_attn.q_proj --ranks 32 64 128 256 512 640 --target-bpw 0.55
~~~

Read [docs/TRAINING_QUALITY.md](docs/TRAINING_QUALITY.md) before increasing rank or steps. A real trained compressed 27B chatbot has **not** been built by this improvement.

## Automatic, resumable multi-projection training (new)

You can now scan **original pretrained Gemma 3 27B dense teacher** tensors and automatically train selected LittleBit-2 projections sequentially. Every successful export is checksummed, audited and recorded in an atomic progress manifest. Interrupted sweeps can resume without repeating successful jobs or SVD initialization. One new projection per invocation is the safe default; failed quality gates are retained and reported without falsely certifying model quality.

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main

$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

# Requires original local dense Gemma 3 27B Safetensors, not a LittleBit output:
& $py littlebitchat.py train inventory --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1

# Train first projection only, from the actual teacher:
& $py littlebitchat.py train batch --checkpoint "D:\Models\gemma-3-27b-it" --layer-start 0 --layer-end 1 --projections self_attn.q_proj --rank 32 --steps 50 --device cpu --max-jobs 1 --out training_runs\gemma-first
~~~

To process all seven projections in a layer, use a different fresh output directory and omit --projections; use --resume on subsequent runs. Detailed workflow: [BATCH_TRAINING.md](docs/BATCH_TRAINING.md).

**This does not create a complete language model**. It requires the *original pretrained dense teacher* to transfer knowledge, still uses random-input probes (not true Gemma activations), and has not performed model-wide distillation or assembled a runnable 27B checkpoint.

## Train your own LittleBit-2 factors

**New:** LittleBitChat now contains a **real learnable LittleBit-2 training pipeline**. It can train a single binary low-rank projection, use SVD + Joint-ITQ initialization, run SmoothSign quantization-aware updates, resume from the local optimizer checkpoint, and export a signed packed Safetensors tensor that the inference math can reproduce. See [Training documentation](docs/LITTLEBIT2_TRAINING.md) for the full instructions and limitations.

The **first command works without any pretrained checkpoint** by training against a tiny synthetic teacher. This proves factor training and export, but cannot give the model language knowledge.

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements-training.txt
& $py littlebitchat.py train demo --steps 50 --rank 8 --device cpu --out training_runs\demo
~~~

To train against **real learned Gemma 3 knowledge** instead, obtain the original pretrained dense Gemma 3 weights legally and use the layer command. Unlike the original full-model Samsung trainer, this pipeline reads only one dense projection per job:

~~~powershell
& $py littlebitchat.py train layer --checkpoint "D:\Models\gemma-3-27b-it" --layer 0 --projection self_attn.q_proj --rank 32 --itq-iterations 20 --steps 50 --device cpu --out training_runs\real-gemma-layer
~~~

**Not complete:** This does not yet compress/assemble every one of Gemma 3 27B's 434 projection modules, use true calibration activations, perform full-model distillation or create coherent chat responses. It is a genuine **single-layer** training foundation, not a completed 27B checkpoint. No cloud charges, checkpoint downloads or KalshiEdge changes.

Standalone local inference research for **Samsung LittleBit/LittleBit-2 + Gemma 3 27B**, targeting **Windows 11, RTX 4050 (6 GB VRAM), and 16 GB system RAM**.

**Current status:** The migrated packed CUDA engine and tiny synthetic checkpoints previously passed **160 tests on the RTX 4050** in the KalshiEdge lab. This standalone repository adds an **experimental** 62-layer streaming scheduler, a real tokenizer interface, out-of-core embedding reads, and a chat CLI. **No trained, quality-validated Gemma 3 27B LittleBit-2 checkpoint is included or confirmed available.** Full 27B text inference has NOT been validated. Without trained weights, the chat command refuses to generate fabricated replies.

All model operations are local. No paid cloud, no Qwen substitution, no automated model download. The established engine and its tests live under engine/. KalshiEdge is not modified.

## Install locally (Windows PowerShell)

Reuse the working CUDA venv from the previous LittleBit tests:

~~~powershell
cd "$HOME\Desktop"
git clone https://github.com/RoarStaze/LittleBitChat.git
cd LittleBitChat

$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt

& $py littlebitchat.py status
~~~

This does NOT reinstall CUDA. An optional PowerShell installer is included in install.ps1.

## Commands and verification

~~~powershell
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py status --json
& $py littlebitchat.py preflight
& $py -m unittest discover -s engine\tests -v
& $py -m unittest discover -s tests -v
~~~

The 27B architecture preflight covers **62 blocks and 434 decoder projections** and produces a planning-only memory report. Without a checkpoint, the result is PLAN_ONLY_NO_CHECKPOINT with runnable_model=false.

## Chat with a genuine trained checkpoint, if one becomes available

The original uncompressed Gemma model is **NOT** a LittleBit checkpoint. An actual compatible checkpoint must contain Samsung U_packed/V_packed factor and scaling tensors for every required projection. This repository does not train or convert 27B weights.

With a real trained, licensed checkpoint directory and the matching local tokenizer.json:

~~~powershell
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements-chat.txt

& $py littlebitchat.py preflight --checkpoint "D:\Models\Gemma3-27B-LittleBit2" --context 4096

& $py littlebitchat.py chat --checkpoint "D:\Models\Gemma3-27B-LittleBit2" --tokenizer "D:\Models\Gemma3-27B-LittleBit2\tokenizer.json" --context 4096 --max-new-tokens 32 --allow-unverified-checkpoint
~~~

The explicit allow-unverified-checkpoint flag acknowledges **tensor names and shapes do not prove learned weights or model quality**. It does not make random weights behave like a real model. The console accepts /exit and /clear; the --once option can submit a single message. Every conversation turn currently recomputes the whole prompt.

## What the code provides

- Migrated and tested Samsung-compatible LittleBit packed CUDA matvec kernels, Safetensors reader, tiny decoder, and research tests.
- A separate full-dimension checkpoint preflight for Gemma 3 27B: 62 decoder blocks, 52 sliding attention blocks, 10 global blocks, all six per-block norms, embeddings and 434 projections.
- A full-size experimental model scheduler that loads one packed decoder block from disk at a time using the established CUDA warp backend, then releases its weight allocations before loading the next block.
- FP32 per-layer K/V cache in a separate capped CUDA memory pool and Gemma 3 normalization, rotary embeddings, grouped-query attention and gated MLP operations.
- Bounded input embedding row reads and chunked tied output projection without loading the giant full embedding matrix onto GPU.
- A local tokenizer JSON adapter, Gemma instruction turn formatting, greedy token generation and terminal conversation loop.
- Fail-closed behavior when weights/tokenizer are missing, dense, malformed, over-budget or not explicitly permitted for experimental inference.

**Unverified limitations:** The full 27B path has not been executed with authentic trained weights. Performance, text quality, model-wide numerical parity and measured process peak RAM/VRAM are unknown. FP32 K/V, per-token reloading of every decoder block, repeated KV concatenation, and repeated streaming of tied embeddings may be extremely slow. CuPy private pool caps are not whole-process VRAM limits. Vision encoder and multimodal features are not supported. The default 4096-token budget allocates planned capacity of 5 GiB VRAM and ~8.13 GiB RAM, but **does not prove the model fits**.

## Token-disjoint validation at identical BPW (experimental)

For Gemma layer-0 attention, the same token ID produces the same input
activation even across different prompts. The opt-in
`train capture-layer0 --token-disjoint --min-unique-tokens 64`
excludes all token IDs shared between three prompt groups, emits one
activation per selected unique ID, and records aligned token IDs in a
versioned Safetensors capture.

Use `train activation-audit --require-token-disjoint` to check split
integrity **and replay every token ID against the original local weights**.
The strict `train activation-calibrate --require-token-disjoint` refuses
legacy prompt-only captures or mismatched sources before expensive SVD,
then preserves the unchanged BPW, held-out selection, independent audit
and exact fallback policy. The original eight prompts/group likely need
to be expanded to reach a 64-token-per-split default.

[Token-disjoint capture, audit and strict calibration instructions](docs/TOKEN_DISJOINT_VALIDATION.md).

## Low-RAM capture for Gemma 3 layer 0

For a 15.7 GB RAM laptop, use `train layer0-preflight` to check original
embedding, first RMSNorm and q/k/v projection shard availability without
loading the 27B model. After obtaining the matching original
`tokenizer.json`, `train capture-layer0` reconstructs actual
layer-0 attention input activations using only selected embedding rows.
It does not need the full model or `config.json` when the original
Gemma 3 default RMSNorm parameters apply. Layer 1+ and MLP captures
remain unsupported by this lightweight method.

[Step-by-step low-RAM capture instructions](docs/LAYER0_LIGHTWEIGHT_CAPTURE.md).

## Activation-aware projection calibration (experimental)

LittleBitChat now supports optional **real original-model activation capture**
(`train capture-activations`) and **same-BPW calibration with holdout
fallback** (`train activation-calibrate`). Three disjoint capture splits
are required: fit / selection / independent audit. Missing activations are a
hard error, not substituted with Gaussian inputs. Only existing output-row
magnitude scales are modified and the fixed packed tensor format is retained.

Capturing genuine Gemma 3 27B activations requires loading the complete
original model on adequate RAM, potentially on another workstation. This is
**not** whole-model training and does not establish Gemma chat quality.

See [Activation-aware calibration and capture instructions](docs/ACTIVATION_CALIBRATION.md).

## Update later

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
~~~

## Tests

CPU continuous integration checks the migrated LittleBit tests and the new tokenizer, scheduler and CLI contracts. Hardware CUDA tests still need to be run in your own RTX 4050 environment. A synthetic test pass does not establish a runnable 27B chatbot.

See [Architecture and acceptance checklist](docs/ARCHITECTURE.md).
