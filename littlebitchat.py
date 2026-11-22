"""LittleBitChat: local-only Gemma 3 LittleBit checkpoint inspector and chat CLI.

NO synthetic replies, internet models, cloud inference or pretrained checkpoint
downloads. To attempt real chat, the user must supply their own trained,
Samsung-compatible packed Gemma 3 27B weights and tokenizer.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from chat_runtime import (ChatUnavailable, FullGemmaChatRuntime, inspect_readiness,
                          ENGINE_DIR)
from chat_tokenizer import GemmaChatTokenizer, TokenizerError


def create_parser():
    p = argparse.ArgumentParser(prog="LittleBitChat",
        description="Local Samsung LittleBit-2 Gemma 3 27B lab. NO pretrained weights included.")
    commands = p.add_subparsers(dest="command", required=True)

    status = commands.add_parser("status", help="Check local checkpoint readiness without downloading weights")
    status.add_argument("--checkpoint", type=Path)
    status.add_argument("--tokenizer", type=Path)
    status.add_argument("--config", type=Path)
    status.add_argument("--context", type=int, default=4096)
    status.add_argument("--json", action="store_true", help="Machine-readable readiness report")

    preflight = commands.add_parser("preflight", help="Plan RAM/VRAM use and validate packed tensor structure")
    preflight.add_argument("--checkpoint", type=Path)
    preflight.add_argument("--index", type=Path, help="Inspect only a local manifest JSON")
    preflight.add_argument("--config", type=Path)
    preflight.add_argument("--context", type=int, default=4096)
    preflight.add_argument("--gpu-gib", type=float, default=6.0)
    preflight.add_argument("--ram-gib", type=float, default=16.0)

    chat = commands.add_parser("chat", help="Try real local chat using YOUR trained packed weights")
    chat.add_argument("--checkpoint", type=Path)
    chat.add_argument("--tokenizer", type=Path)
    chat.add_argument("--config", type=Path)
    chat.add_argument("--context", type=int, default=4096)
    chat.add_argument("--max-new-tokens", type=int, default=32)
    chat.add_argument("--once", help="One message, then exit instead of opening a console")
    chat.add_argument("--allow-unverified-checkpoint", action="store_true",
                      help="Acknowledges that tensor shapes do not establish model quality")
    return p


def _chat(args):
    if args.checkpoint is None:
        raise ChatUnavailable("A local --checkpoint is required for chat. No trained Gemma 3 LittleBit-2 weights are included.")
    if not args.allow_unverified_checkpoint:
        raise ChatUnavailable("A model's training/quality cannot be verified from tensor names. "
                              "Use --allow-unverified-checkpoint only if you trust your own local weights.")
    readiness = inspect_readiness(checkpoint=args.checkpoint, tokenizer=args.tokenizer,
                                   config=args.config, context=args.context)
    if readiness["status"] == "MISSING_TOKENIZER":
        raise ChatUnavailable("Local tokenizer.json is missing; specify --tokenizer PATH")
    tok = GemmaChatTokenizer(readiness["tokenizer_file"])
    model = FullGemmaChatRuntime(args.checkpoint, config=args.config, context=args.context,
                                 allow_unverified=True)
    if tok.vocab_size > model.arch.vocab:
        raise ChatUnavailable("Tokenizer vocabulary exceeds checkpoint embedding size")

    print("LittleBitChat | experimental 27B checkpoint execution")
    print("Training quality is UNVERIFIED; performance and full-size CUDA correctness are UNTESTED.")
    print("Type /exit to quit, /clear to clear conversation.")
    transcript: list[tuple[str, str]] = []
    while True:
        if args.once is not None:
            user_text = args.once
        else:
            try:
                user_text = input("\nYou > ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if user_text.strip().lower() in ("/exit", "/quit"):
            break
        if user_text.strip().lower() == "/clear":
            transcript.clear()
            if args.once is not None:
                break
            print("Conversation cleared.")
            continue
        if not user_text.strip():
            if args.once is not None:
                raise ChatUnavailable("Cannot send an empty message")
            continue
        current = transcript + [("user", user_text)]
        prompt = tok.encode_conversation(current)
        # Recompute prompt from history (no cross-turn KV reuse); drop oldest
        # complete user/model pair if required, never drop the current message.
        while len(prompt) + args.max_new_tokens > args.context and len(current) > 1:
            current = current[2:]
            prompt = tok.encode_conversation(current)
        if len(prompt) + args.max_new_tokens > args.context:
            raise ChatUnavailable("Message exceeds the configured context limit")
        result = model.generate(prompt, max_new_tokens=args.max_new_tokens, stop_ids=tok.stop_ids)
        response = tok.decode(result["generated_token_ids"])
        print("\nLittleBit > " + (response if response else "[No response tokens generated]"))
        print(f"  ({result['generated_tokens']} tokens, {result['elapsed_seconds']}s; quality unverified)")
        transcript = current + [("model", response)] if response.strip() else current
        if args.once is not None:
            break
    return 0


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = inspect_readiness(checkpoint=args.checkpoint, tokenizer=args.tokenizer,
                                       config=args.config, context=args.context)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"LittleBitChat: {result['status']}")
                print(result["message"])
                print("No synthetic demonstrations are presented as trained conversations.")
            return 0
        if args.command == "preflight":
            from gemma3_bridge import read_gemma_config, test_fixture_path
            from gemma3_27b_ready import build_report, MemoryLimits
            if args.checkpoint is not None and args.index is not None:
                raise ChatUnavailable("Specify --checkpoint OR --index, not both")
            cfg = read_gemma_config(args.config or test_fixture_path())
            limits = MemoryLimits(gpu_total_gib=args.gpu_gib, system_total_gib=args.ram_gib)
            result = build_report(cfg, context_tokens=args.context,
                                  checkpoint=args.checkpoint, manifest=args.index, limits=limits)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "chat":
            return _chat(args)
        parser.error("Unknown command")
    except (ChatUnavailable, TokenizerError, ValueError, OSError) as exc:
        print("LittleBitChat error: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
