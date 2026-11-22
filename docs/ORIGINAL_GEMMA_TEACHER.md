# Acquiring one original Google Gemma 3 27B pretrained weight shard

**Why this exists:** The training command failed because `D:\Models\gemma-3-27b-it` was a placeholder, not a file that exists on your Windows machine. LittleBitChat trains our own compressed LittleBit-2 factors **from Google's original pretrained dense weights**; it does not require another person's finished LittleBit model.

Google's original Gemma 3 model repository is **license-gated**. You must personally read/accept Google's model terms while signed in at:

https://huggingface.co/google/gemma-3-27b-it

The GitHub project, assistant, and scripts cannot accept those legal terms for you. You must also authenticate with Hugging Face using your own account. Never paste your API token into chat.

## One-time setup on the laptop

~~~powershell
cd "$HOME\Desktop\LittleBitChat"
git pull --ff-only origin main
$py = "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\python.exe"

& $py -m pip install -r requirements-teacher.txt
& "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\hf.exe" auth login
& "$HOME\Desktop\LittleBit-CUDA-venv\Scripts\hf.exe" auth whoami
~~~

When prompted, authenticate with your own Hugging Face account. If the account has not accepted Gemma's terms, the download will fail with an actionable error. **No token is passed to LittleBitChat as a command-line argument.**

## First: metadata-only preview

~~~powershell
& $py littlebitchat.py train acquire --layer 0 --projection self_attn.q_proj --out models\gemma-3-27b-it
~~~

Expected `TEACHER_SHARD_PLAN_ONLY` plus the **actual pinned official model revision**, the exact shard mapped to layer 0 q_proj by Google's Safetensors index and its expected size in bytes. This downloads ONLY the index JSON (small metadata); it does **not** download a multi-GB shard.

Only proceed if you have enough disk space. For an original 27B BF16 checkpoint, a single shard can still occupy **several gigabytes**. Size/capacity are checked before any weight download.

## Download just that weight shard

~~~powershell
& $py littlebitchat.py train acquire --layer 0 --projection self_attn.q_proj --out models\gemma-3-27b-it --download --max-shard-gib 8
~~~

The tool pins a single model commit from the official repo, and downloads ONLY the actual shard containing the requested projection. It does not download all 12 shards, the vision encoder, or the entire 50+GB original model. It rejects missing/unsafe index paths, unknown weight size, over-budget files, insufficient free disk space and absent license/access.

## Inspect the acquired weight and start training

~~~powershell
& $py littlebitchat.py train inventory --checkpoint models\gemma-3-27b-it --layer-start 0 --layer-end 1 --projections self_attn.q_proj

& $py littlebitchat.py train batch --checkpoint models\gemma-3-27b-it --layer-start 0 --layer-end 1 --projections self_attn.q_proj --rank 32 --steps 50 --device cpu --max-jobs 1 --out training_runs\gemma-first
~~~

The existing local batch trainer reads only one original dense teacher tensor from its shard and exports a real trained LittleBit-2 projection, with audit and resumption. These outputs are **individual layers**, not a complete 27B compressed model or coherent chatbot. Training the 27B model with language-preserving quality still requires real activation calibration, cross-layer distillation, all necessary shards and model assembly.

For subsequent projections, the same acquisition command can request the shard containing that projection, with **explicit --download** for each new shard. The previously downloaded index will be reused or validated by the official library at a pinned revision.

## Hardware and security cautions

- A 6GB GPU cannot train a full dense 27B model; default layer training uses CPU and a small rank. The model repository size is not the available VRAM.
- **Only a shard is downloaded**; `LittleBitChat chat` will still report a missing complete compressed checkpoint.
- The original source must be licensed and obtained through your own account, without bypassing access controls.
- Model files under the repo's ignored `models/` directory and local `training_runs/` are never committed to GitHub.
- The Hugging Face client maintains its own token cache. Never share tokens or upload downloaded licensed weights publicly.
- Metadata-only `train acquire` still requires network access and account authorization.
