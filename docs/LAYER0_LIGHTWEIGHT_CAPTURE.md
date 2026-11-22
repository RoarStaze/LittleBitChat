# Low-RAM Gemma 3 layer-0 activation capture

This is a **restricted** alternative to `train capture-activations`, which
requires loading the full 27B BF16 model (~54 GB raw weights). On a machine
with 15.7 GB RAM, do NOT start the full-model capture command.

Official Gemma 3 architecture: text input token IDs enter
`language_model.model.embed_tokens`, whose output is multiplied by the
square root of the hidden dimension **in the original embedding dtype**.
Block 0 applies `language_model.model.layers.0.input_layernorm` (Gemma's
RMSNorm uses FP32 accumulation, a learned (1+weight), then casts back to
the embedding dtype). Its `self_attn.q_proj`, `k_proj` and `v_proj`
are called on that normalized value. For these **three projections in
block 0 only**, the input can be reconstructed without evaluating any
attention operation or loading blocks 1–61.

This does **not** support layer 1+, MLP, `o_proj`, image tokens, or
cross-model tokenizer substitution. It uses the actual original embedding
rows and norm weights, *not* random Gaussian placeholders. The capture
metadata explicitly says `gemma3_layer0_embedding_rmsnorm`, not
`forward_pre_hook`. Source authenticity is not independently verified.

## 1. Inspect what you already have (quick, no heavy weights loaded)

```powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py littlebitchat.py train layer0-preflight `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj
```

This checks the Safetensors shard index and tensor headers without
reading the entire embedding matrix. It reports the shard filenames
holding these three required original tensors:

- `language_model.model.embed_tokens.weight`
- `language_model.model.layers.0.input_layernorm.weight`
- `language_model.model.layers.0.self_attn.q_proj.weight`

`needed_weight_shards_present` must be `true`. The existing model
index alone does NOT mean the embedding or norm shard was downloaded.
The output lists missing shard names without fetching or scanning your
disk. `config_json_present` can be false; the layer-0 lightweight
capture uses the documented Gemma 3 default RMSNorm epsilon (1e-6)
when absent, and indicates this choice in provenance. A local
`config.json`, if present, is validated for hidden size and epsilon.

## 2. Obtain the official matching tokenizer, if missing

The lightweight path still needs a real **Gemma 3 IT tokenizer.json**.
It never fabricates token IDs or downloads anything automatically.
Only after accepting the model license and authenticating with
Hugging Face if required, you can explicitly fetch just that one
small model asset (not 27B weights):

```powershell
& $py -m pip install "tokenizers>=0.19,<1" huggingface_hub

& $py -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='google/gemma-3-27b-it', filename='tokenizer.json', local_dir='models/gemma-3-27b-it'))"
```

Official model page: https://huggingface.co/google/gemma-3-27b-it

If you already own a matching local `tokenizer.json`, use
`--tokenizer PATH_TO_TOKENIZER_JSON` instead. Do not assume a tokenizer
from a different model is equivalent. This model's provided instruction
template is handled by the project's `GemmaChatTokenizer` adapter.

## 3. Capture layer-0 q-projection inputs with ~16 GB RAM

After `layer0-preflight` confirms the required shards and tokenizer,
and your existing three text files are present:

```powershell
& $py littlebitchat.py train capture-layer0 `
  --checkpoint models\gemma-3-27b-it `
  --layer 0 --projection self_attn.q_proj `
  --calibration-texts calibration.txt `
  --selection-texts selection.txt `
  --audit-texts audit.txt `
  --max-rows 512 --max-tokens 128 `
  --out training_runs\capture_layer0_q.safetensors
```

Unlike `capture-activations`, this command does **not** need
`--confirm-full-model-memory`. It memory-maps the indexed Safetensors
file and calls `get_slice` for distinct token embedding rows only.
It loads the small norm vector and one bounded teacher projection.
The complete original embedding tensor (~2.8 GB BF16) and 27B model
are never loaded into memory. Keep available RAM and disk space for the
original *shard files*, which still must exist locally.

This writes the same three-split Safetensors interface consumed by
`train activation-calibrate` (calibration / selection / audit), with
original teacher SHA-256 binding, source prompt hashes and explicit
low-RAM capture-method metadata. It rejects fully reused activation
splits and overlapping prompts. Hidden-state files can reveal private
information; do not publish them.

## 4. Reuse the prior activation-aware same-BPW calibration

```powershell
& $py littlebitchat.py train activation-calibrate `
  --checkpoint models\gemma-3-27b-it `
  --activation-data training_runs\capture_layer0_q.safetensors `
  --layer 0 --projection self_attn.q_proj `
  --rank 622 --max-bpw 0.55 `
  --itq-iterations 20 `
  --svd-oversample 32 --svd-power-iters 5 `
  --refine-output-scales `
  --binary-refine-channels 64 --binary-refine-passes 3 `
  --min-relative-improvement 0.001 `
  --out training_runs\layer0_q_activation
```

The same-BPW check, held-out selection gate, independent audit and
exact factor rollback remain unchanged. This is still **single-layer
projection calibration**, not trained or demonstrated full-model
Gemma 3 conversation quality.

References: official HF Gemma 3 `Gemma3TextModel` and
`Gemma3TextScaledWordEmbedding` implementation, HF Gemma 2
`Gemma2DecoderLayer` and `Gemma2RMSNorm`, and Safetensors
`safe_open.get_slice` documentation.
