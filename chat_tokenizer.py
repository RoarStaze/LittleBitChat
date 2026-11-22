"""Strict local Gemma 3 chat templating for a real tokenizer.json file.

No tokenizers package is imported until a local file is supplied. The
runtime never downloads anything or substitutes a fake learned tokenizer.
"""
from __future__ import annotations

from pathlib import Path


class TokenizerError(ValueError):
    """Local tokenizer missing, malformed, or incompatible."""


_SPECIALS = ("<bos>", "<start_of_turn>", "<end_of_turn>", "<eos>")
_RESERVED = _SPECIALS + ("<pad>", "<start_of_image>", "<image>", "<unused0>")


class GemmaChatTokenizer:
    def __init__(self, tokenizer_path=None, *, backend=None):
        if backend is None:
            file = Path(tokenizer_path) if tokenizer_path is not None else None
            if file is None or not file.is_file() or file.name != "tokenizer.json":
                raise TokenizerError("A local Gemma tokenizer.json is required; no tokenizer will be downloaded")
            if file.stat().st_size > 200 * 1024 * 1024:
                raise TokenizerError("Tokenizer JSON exceeds local safety limit")
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise TokenizerError("Install the optional tokenizers package: python -m pip install tokenizers") from exc
            try:
                backend = Tokenizer.from_file(str(file))
            except Exception as exc:
                raise TokenizerError("Unable to parse local tokenizer.json") from exc
        self.backend = backend
        self.ids = {symbol: backend.token_to_id(symbol) for symbol in _SPECIALS}
        if any(type(token_id) is not int or token_id < 0 for token_id in self.ids.values()):
            raise TokenizerError("Tokenizer is missing required Gemma 3 control tokens")
        if len(set(self.ids.values())) != len(_SPECIALS):
            raise TokenizerError("Gemma control token IDs are not distinct")
        self.stop_ids = {self.ids["<eos>"], self.ids["<end_of_turn>"]}
        self.vocab_size = int(backend.get_vocab_size(with_added_tokens=True))

    def _encode(self, content: str) -> list[int]:
        if not isinstance(content, str):
            raise TokenizerError("Conversation content must be text")
        try:
            return list(self.backend.encode(content, add_special_tokens=False).ids)
        except Exception as exc:
            raise TokenizerError("Tokenizer could not encode the conversation") from exc

    def encode_conversation(self, turns) -> list[int]:
        """Gemma IT conversation: BOS + complete turns + open model turn.

        Role/content are kept separate: control IDs are inserted directly,
        and untrusted user strings may not contain reserved delimiter tokens.
        Gemma 3 does not accept a separate system role in this simple template.
        """
        if not isinstance(turns, (list, tuple)) or not turns:
            raise TokenizerError("At least one conversation turn is required")
        prompt = [self.ids["<bos>"]]
        for role, content in turns:
            if role not in ("user", "model"):
                raise TokenizerError("Only user/model turns are supported")
            if not isinstance(content, str) or not content.strip() or len(content) > 100_000:
                raise TokenizerError("Conversation turn text is empty or oversized")
            if any(marker in content for marker in _RESERVED):
                raise TokenizerError("Message contains Gemma control delimiters; rephrase it")
            prompt.append(self.ids["<start_of_turn>"])
            prompt.extend(self._encode(role + "\n"))
            prompt.extend(self._encode(content))
            prompt.append(self.ids["<end_of_turn>"])
        prompt.append(self.ids["<start_of_turn>"])
        prompt.extend(self._encode("model\n"))
        return prompt

    def decode(self, token_ids) -> str:
        if not isinstance(token_ids, (list, tuple)) or any(
                type(x) is not int or x < 0 for x in token_ids):
            raise TokenizerError("Invalid generated token ID sequence")
        try:
            return str(self.backend.decode(list(token_ids), skip_special_tokens=True))
        except Exception as exc:
            raise TokenizerError("Tokenizer could not decode generated tokens") from exc
