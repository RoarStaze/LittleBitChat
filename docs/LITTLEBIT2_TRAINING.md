# LittleBit-2 layer training: real QAT, local checkpoints

## What is implemented

This project can **train one Samsung-compatible LittleBit-2 packed projection** against an original dense Gemma 3 teacher matrix from a user-supplied local Safetensors checkpoint. The new trainer implements PyTorch randomized SVD, Joint-ITQ orthogonal alignment, dual factor magnitude scales, the hard-sign forward / SmoothSign surrogate backward, optional residual compensation, AdamW layerwise training, held-out random-probe RMSE, resumable optimizer/factor state, and standard packed Safetensors export.

The packed export is independently read and evaluated against the trained signed QAT layer. Export metadata records the actual estimated BPW, teacher hash, normalized RMSE and whether the source teacher was synthetic.

**Important:** This is an independently authored **layerwise training research adaptation** of Samsung's method, not a guaranteed paper-equivalent implementation of complete-model QAT. Training input probes are currently randomly generated, NOT real Gemma activations. It cannot, by itself, preserve full language quality or produce a working 27B chatbot. No full model checkpoint is assembled.

## Local install

Training requires optional PyTorch; the existing LittleBitChat status and inference tools do not.

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main

$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py -m pip install -r requirements-training.txt
~~~

PyTorch can be a large dependency. The demonstrated training path runs with --device cpu. CUDA training only works if PyTorch itself reports torch.cuda.is_available() = True, separately from CuPy.

## Start training now: synthetic teacher sanity check

~~~powershell
& $py littlebitchat.py train demo --steps 50 --rank 8 --device cpu --out training_runs\demo
~~~

This trains *learned binary low-rank factors*, saves optimizer progress, and exports an actual Samsung-compatible Safetensors matrix. The source teacher is synthetic, so its output is NOT a language model. The output reports LAYER_QAT_EXPORTED and packed_inference_verified after testing the exported matrix.

The default 0.55 target BPW cannot be met by many tiny matrices because scaling-vector overhead dominates. Always inspect actual_effective_bits.

## Train one real pretrained Gemma projection

If you possess a legitimate local original Gemma 3 27B dense checkpoint, such as a directory with model.safetensors.index.json, run:

~~~powershell
& $py littlebitchat.py train layer --checkpoint "D:\Models\gemma-3-27b-it" --layer 0 --projection self_attn.q_proj --rank 32 --itq-iterations 20 --steps 50 --batch-size 16 --device cpu --out training_runs\real-gemma-layer
~~~

The training script reads ONLY the named teacher projection, checks its shape/dtype before loading, and refuses unknown/missing/dense-shard paths or selected tensor sizes over 512 MiB by default. It does not download the original 50+ GB weights automatically, load the whole model or require someone else's LittleBit checkpoint.

**Do not attempt to SVD a large 27B projection at a huge rank on 6 GiB VRAM.** Start on CPU with a small rank and measure error. A low error on synthetic input probes is not a measure of chatbot quality.

Resume the exact same projection and hyperparameters, but change steps from 50 to 100 and append --resume:

~~~powershell
& $py littlebitchat.py train layer --checkpoint "D:\Models\gemma-3-27b-it" --layer 0 --projection self_attn.q_proj --rank 32 --itq-iterations 20 --steps 100 --batch-size 16 --device cpu --out training_runs\real-gemma-layer --resume
~~~

Resume validates the teacher SHA and hyperparameters, restores the trained factors and AdamW state with PyTorch weights-only loading, and **does not redo SVD or Joint-ITQ**. Packed output filenames include the step number; previous exports are preserved.

## Tests

~~~powershell
& $py -m unittest discover -s training\tests -v
& $py -m unittest discover -s engine\tests -v
& $py -m unittest discover -s tests -v
~~~

The new GitHub Actions PyTorch CPU job tests initialization, gradients, training on random probes, actual BF16 teacher tensor loading, resume, and packed export. The established 160 engine tests and 12 chat tests remain separate.

## Explicit unsolved work

- Training **all 434** Gemma 3 27B decoder projections automatically and using **real layer activation calibration**.
- Joint model-wide teacher/student distillation, evaluation on real language benchmarks, and a full tokenizer/embedding/norm/checkpoint assembler.
- Actual 27B QAT on this RTX 4050, proven full-model memory fit, or usable conversational quality.
- Original dense Gemma 3 weights must be lawfully obtained and stored separately; this code cannot recreate learned weights from nothing.

## Primary sources and license

Research design follows [SamsungLabs/LittleBit](https://github.com/SamsungLabs/LittleBit), especially the LittleBitLinear / SmoothSign code and its Joint-ITQ option, and the [LittleBit-2 ICML 2026 paper](https://proceedings.mlr.press/v306/lee26y.html). This repository implementation is independently authored and is NOT a guarantee of exact scientific reproduction. The upstream Samsung software carries a CC BY-NC 4.0 noncommercial license, and the original Gemma model has separate license terms.

No KalshiEdge code, trading data or credentials are modified.
