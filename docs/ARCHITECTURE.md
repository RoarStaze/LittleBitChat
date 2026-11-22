# LittleBitChat architecture and verification

## Scope

Separate repository, no KalshiEdge imports. Original LittleBit research engine copied into engine/, retaining the prior 160-test GPU suite. New files are littlebitchat.py, chat_runtime.py, chat_tokenizer.py, tests/, install.ps1 and the CI workflow.

**Target:** Gemma 3 27B text-only, Samsung LittleBit-2 packed binary low-rank factors, local RTX 4050 6GB / host 16GB. No Qwen replacement, remote API, paid GPU or pretrained weights included.

## Status and threat model

The working tiny model is synthetic, and the full-size 27B code has no available trained checkpoint. Chat refuses to start without a local compatible checkpoint and matching tokenizer; it requires a deliberate --allow-unverified-checkpoint flag because neither tensor keys nor file checksums can establish model training or response quality.

A supplied packed checkpoint may be malicious or malformed. Preflight rejects dense projection tensors, missing factor keys, inconsistent tensor dimensions, unknown dtypes, oversized tensor payloads and unsafe shard paths, using the existing Safetensors reader. It is not a cryptographic authenticity verifier. Keep checkpoints from trusted sources only.

## Exact data flow

1. The Gemma 3 27B config is checked against the published 62-layer, 32 query-head, 16 KV-head, 5-local/1-global architecture.
2. The checkpoint preflight checks all 434 packed decoder projections, normalization metadata, embedding dimensions and selected per-block byte budget WITHOUT loading all model weights.
3. A local Hugging Face tokenizer.json is loaded with the optional Rust-backed tokenizers Python package. Gemma start/end turn IDs are inserted explicitly and user content is not allowed to contain literal special delimiter tokens. Conversation is rebuilt each turn.
4. For each token, read only the corresponding embedding row directly from its Safetensors shard offset.
5. For all 62 layers, in order: read and allocate only the current packed weight factors into one private CUDA pool; calculate RMSNorm, Q/K norm, RoPE, grouped-query causal attention, gated MLP and residuals using a separate CUDA state pool; evict packed layer weights before moving to the next layer.
6. The FP32 local/global KV cache persists across processed tokens. The current cache uses concatenation and slicing (correctness-first, NOT an optimized ring buffer).
7. Final RMSNorm, then scan tied embedding rows in bounded chunks to select the greedy output token. All logits are model-derived; no canned/fake response text is used.
8. Stop on Gemma EOS or end-of-turn token, or max-new-tokens; decode token IDs using the real tokenizer.

This is an *experimental implementation*. It has not been checked against an actual trained Gemma 3 27B LittleBit-2 checkpoint. Numerically accurate large-model performance has not been established.

## Memory placement and known limits

- BF16 embedding tensor is read from disk for input lookup/output selection; it is not simultaneously resident as full FP32 GPU weights.
- CUDA weight pool is capped at 1024 MiB by default per block; persistent state pool at 2048 MiB.
- FP32 K/V at 4096 tokens has estimated raw capacity of 1.4375 GiB (52 local blocks × 1024 positions plus 10 global blocks × 4096, 16 heads × 128 dimensions × K/V × 4 bytes).
- Memory reservations include an additional 512 MiB GPU workspace + 1.5 GiB GPU context/driver/other reserve. Estimated total GPU reservation 5 GiB. System reservation ~8.13 GiB with BF16 embedding allowance even though chunked streaming can have lower physical RSS.
- These are planning allocations, not observed usage or process-wide hard bounds. The current CuPy/private allocator, vision encoder, kernel compile, caching and operating system can exceed them. There is no measured full-model peak VRAM/RAM.
- Every token reloads every layer's packed factors and scans 2.6+ GiB of embedding payload for tied logits; inference may be **extremely slow** on a laptop.
- This implementation does not support vision, multimodal prompts, batching, high-throughput production sampling, speculative decoding, persistent conversation KV reuse, >4096 context or quantization training.
- Without trained checkpoint weights there is no way to validate 27B inference, quality or speed. The original dense .weight Gemma3 27B checkpoint is **incompatible**.

## Verification gates

- Tests migrated from KalshiEdge: 160/160 passed on the actual RTX 4050 before the move. This historical result is not a fresh verification of all new files.
- New CPU tests: tokenizer control delimiters/injection, missing weights, 62-block single-residency walk and failure cleanup, bounded disk embedding row access, and CLI fail-closed behavior.
- CI runs both the migrated suite and new tests. CUDA-specific tests skip on CPU runner; target RTX 4050 test suite must be run locally.
- Before declaring a real 27B model works: independently verified trained/compatible checkpoint and license; full-model golden-answer numerical parity against a reference; total process peak RAM+VRAM measurement; real tokens/second and coherent response benchmarks; long-context stability and eviction correctness.

## Commands

Run from the LittleBitChat repository directory with your existing CUDA Python:

~~~powershell
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"
& $py littlebitchat.py status --json
& $py littlebitchat.py preflight
& $py -m unittest discover -s engine\tests -v
& $py -m unittest discover -s tests -v
~~~

A checkpoint scan is NOT generation:
~~~powershell
& $py littlebitchat.py preflight --checkpoint "D:\Models\REAL-LITTLEBIT2"
~~~

Attempting to chat with a locally obtained checkpoint requires optional tokenizers and explicit acknowledgement:
~~~powershell
& $py -m pip install -r requirements-chat.txt
& $py littlebitchat.py chat --checkpoint "D:\Models\REAL-LITTLEBIT2" --tokenizer "D:\Models\REAL-LITTLEBIT2\tokenizer.json" --allow-unverified-checkpoint
~~~

## Provenance

Source implementation and upstream equations: SamsungLabs/LittleBit and Hugging Face Transformers Gemma 3. The previously validated LittleBitLab tools and tests were moved from RoarStaze/KalshiEdge/experiments/littlebit70b at commit c16a635. Development of this standalone chat project continues in RoarStaze/LittleBitChat, not the trading repository.
