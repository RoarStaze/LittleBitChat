# LittleBitChat


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

## Update later

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
~~~

## Tests

CPU continuous integration checks the migrated LittleBit tests and the new tokenizer, scheduler and CLI contracts. Hardware CUDA tests still need to be run in your own RTX 4050 environment. A synthetic test pass does not establish a runnable 27B chatbot.

See [Architecture and acceptance checklist](docs/ARCHITECTURE.md).
