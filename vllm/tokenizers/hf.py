# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import copy
import queue
import re
import threading
from array import array
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias, TypeVar

import tokenizers
from transformers import AutoTokenizer, PythonBackend, TokenizersBackend
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from vllm.transformers_utils.config import get_sentence_transformer_tokenizer_config

from .protocol import TokenizerLike

HfTokenizer: TypeAlias = PythonBackend | TokenizersBackend
_T = TypeVar("_T", bound=TokenizerLike)


class ThreadSafeHFTokenizerMixin:
    """Mixin class for thread-safe HF fast tokenizers."""

    pass


class SegmentedPromptEncoder:
    """Tokenize long prompts piecewise, with a cache of piece token ids.

    A fast tokenizer first splits the raw text at its added tokens
    (leftmost-longest matching), then normalizes, pre-tokenizes and encodes
    each piece between two added tokens independently of the rest. Splitting
    the text the same way, encoding the pieces separately and interleaving
    the added token ids therefore gives exactly the ids of the whole-string
    ``tokenizer(text)`` call.

    Chat templates delimit every message with added tokens, so a multi-turn
    conversation re-sends mostly the same pieces. Misses are encoded in
    parallel through the Rust batch path, and a bounded LRU maps piece text
    to ids so each turn only encodes what it added.
    """

    # Shorter prompts take the regular tokenizer call.
    MIN_CHARS = 32 * 1024
    # Approximate bytes of piece text and ids kept in the cache.
    CACHE_BYTES = 256 * 1024 * 1024

    def __init__(
        self,
        backend: tokenizers.Tokenizer,
        added_ids: dict[str, int],
        copies: int,
    ) -> None:
        self._backend = backend
        self._added_ids = added_ids
        # Longest first: `re` alternation then matches leftmost-longest like
        # the tokenizer's Aho-Corasick split. The group keeps the matches.
        self._pattern = re.compile(
            "("
            + "|".join(re.escape(c) for c in sorted(added_ids, key=len, reverse=True))
            + ")"
        )
        # Copying a tokenizer takes most of a second; do it at startup.
        self._backends: queue.SimpleQueue[tokenizers.Tokenizer] = queue.SimpleQueue()
        self._backends.put(backend)
        for _ in range(copies - 1):
            self._backends.put(copy.deepcopy(backend))
        self._cache: OrderedDict[str, array] = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.Lock()

    @classmethod
    def maybe_create(
        cls, tokenizer: TokenizerLike, copies: int = 1
    ) -> "SegmentedPromptEncoder | None":
        """Return an encoder when piecewise encoding is exact for `tokenizer`.

        `copies` backend tokenizers are made for concurrent callers."""
        if not isinstance(tokenizer, TokenizersBackend):
            return None
        # Skip the thread pool wrapper; the encode path must be the stock one.
        tok_cls = next(
            c
            for c in type(tokenizer).__mro__
            if not issubclass(c, ThreadSafeHFTokenizerMixin)
        )
        if (
            tok_cls.__call__ is not PreTrainedTokenizerBase.__call__
            or tok_cls._encode_plus is not TokenizersBackend._encode_plus
            or hasattr(tok_cls, "_switch_to_input_mode")
            or tokenizer.split_special_tokens
        ):
            return None
        backend = getattr(tokenizer, "_tokenizer", None)
        if type(backend) is not tokenizers.Tokenizer:
            return None

        added_ids = dict[str, int]()
        for token_id, added in backend.get_added_tokens_decoder().items():
            if added.normalized:
                # Matched after normalization, within a piece.
                continue
            if added.lstrip or added.rstrip or added.single_word:
                return None
            if added.content:
                added_ids[added.content] = token_id
        if not added_ids:
            return None

        backend = copy.deepcopy(backend)
        backend.no_truncation()
        backend.no_padding()
        backend.encode_special_tokens = False
        return cls(backend, added_ids, copies)

    @contextlib.contextmanager
    def _borrow_backend(self):
        try:
            backend = self._backends.get_nowait()
        except queue.Empty:
            backend = copy.deepcopy(self._backend)
        try:
            yield backend
        finally:
            self._backends.put(backend)

    def encode(self, text: str, **kwargs) -> Sequence[int] | None:
        """Return ``tokenizer(text, **kwargs)["input_ids"]``, or None when the
        regular tokenizer call should handle the prompt."""
        if (
            len(text) < self.MIN_CHARS
            or kwargs.keys() - {"add_special_tokens", "truncation", "max_length"}
            or kwargs.get("add_special_tokens", True)
        ):
            return None
        max_length = kwargs.get("max_length") if kwargs.get("truncation") else None

        # Even indices: text between added tokens; odd indices: added tokens.
        parts = self._pattern.split(text)
        pieces = dict.fromkeys(p for p in parts[::2] if p)

        cache = self._cache
        with self._lock:
            found = {p: cache[p] for p in pieces if p in cache}
            for p in found:
                cache.move_to_end(p)
        missing = [p for p in pieces if p not in found]

        if missing:
            with self._borrow_backend() as backend:
                encode_batch = getattr(
                    backend, "encode_batch_fast", backend.encode_batch
                )
                encodings = encode_batch(missing, add_special_tokens=False)
            new = {p: array("I", e.ids) for p, e in zip(missing, encodings)}
            found.update(new)
            with self._lock:
                for p, ids in new.items():
                    size = len(p) + 4 * len(ids)
                    if p in cache or size > self.CACHE_BYTES:
                        continue
                    cache[p] = ids
                    self._cache_bytes += size
                while self._cache_bytes > self.CACHE_BYTES:
                    p, ids = cache.popitem(last=False)
                    self._cache_bytes -= len(p) + 4 * len(ids)

        added_ids = self._added_ids
        token_ids = array("I")
        for i, part in enumerate(parts):
            if i % 2:
                token_ids.append(added_ids[part])
            elif part:
                token_ids.extend(found[part])
        if max_length is not None and len(token_ids) > max_length:
            # Let the tokenizer apply its truncation.
            return None
        return token_ids


def maybe_make_thread_pool(tokenizer: _T, copies: int = 1):
    """If `tokenizer` is a `TokenizersBackend`, modify the tokenizer
    in-place to make the public interface thread-safe by routing calls
    through a deep-copied tokenizer pool.

    Note that:
    - Only ``TokenizerLike``'s public interface is thread-safe.
      This doesn't include ``_tokenizer`` property nor any mutation
      methods like ``add_special_tokens`` or ``add_tokens``.
    - Adjacent method calls could happen on different deep copies.
    """
    if not isinstance(tokenizer, TokenizersBackend) or isinstance(
        tokenizer, ThreadSafeHFTokenizerMixin
    ):
        return tokenizer

    og_tokenizer = copy.copy(tokenizer)

    tokenizer_pool: queue.Queue[TokenizersBackend] = queue.Queue()
    for _ in range(copies):
        tokenizer_pool.put(copy.deepcopy(og_tokenizer))

    @contextlib.contextmanager
    def _borrow_from_pool():
        try:
            tok = tokenizer_pool.get_nowait()
            yield tok
        except queue.Empty:
            tok = copy.deepcopy(og_tokenizer)
            yield tok
        finally:
            tokenizer_pool.put(tok)

    class TokenizerPool(tokenizer.__class__, ThreadSafeHFTokenizerMixin):  # type: ignore
        def apply_chat_template(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.apply_chat_template(*args, **kwargs)

        def batch_decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_decode(*args, **kwargs)

        def batch_encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.batch_encode(*args, **kwargs)

        def convert_tokens_to_ids(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_ids(*args, **kwargs)

        def convert_ids_to_tokens(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_ids_to_tokens(*args, **kwargs)

        def convert_tokens_to_string(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.convert_tokens_to_string(*args, **kwargs)

        def decode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.decode(*args, **kwargs)

        def encode(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok.encode(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with _borrow_from_pool() as tok:
                return tok(*args, **kwargs)

        def __reduce__(self):
            return maybe_make_thread_pool, (og_tokenizer, copies)

    TokenizerPool.__name__ = f"TokenizerPool{og_tokenizer.__class__.__name__}"

    tokenizer.__class__ = TokenizerPool
    # Return the tokenizer: TokenizerPool.__reduce__ reconstructs through this
    # function, so falling off the end would unpickle to None (issue #45433).
    return tokenizer


def get_cached_tokenizer(tokenizer: HfTokenizer) -> HfTokenizer:
    """By default, transformers will recompute multiple tokenizer properties
    each time they are called, leading to a significant slowdown.
    This proxy caches these properties for faster access.
    """
    cached_tokenizer = copy.copy(tokenizer)

    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_len = len(tokenizer)
    # The underlying tokenizer class could be a specific backend,
    # which does not always implement is_fast in Transformers
    tokenizer_is_fast = getattr(tokenizer, "is_fast", True)

    max_token_id = max(tokenizer_vocab.values())
    max_chars_per_token = max(len(tok) for tok in tokenizer_vocab)

    # Some tokenizers (e.g., QwenTokenizer) have special tokens that
    # are added and included in the implementation of the vocab_size
    # property, but not in get_vocab(); if there is an implementation
    # of vocab size, we should take the greater value.
    if hasattr(tokenizer, "vocab_size"):
        with contextlib.suppress(NotImplementedError):
            max_token_id = max(max_token_id, tokenizer.vocab_size)

    class CachedTokenizer(tokenizer.__class__):  # type: ignore
        @property
        def all_special_ids(self) -> list[int]:
            return tokenizer_all_special_ids

        @property
        def all_special_tokens(self) -> list[str]:
            return tokenizer_all_special_tokens

        @property
        def max_token_id(self) -> int:
            return max_token_id

        @property
        def max_chars_per_token(self) -> int:
            return max_chars_per_token

        @property
        def is_fast(self) -> bool:
            return tokenizer_is_fast

        def get_vocab(self) -> dict[str, int]:
            return tokenizer_vocab

        def __len__(self) -> int:
            return tokenizer_len

        def __reduce__(self):
            return get_cached_tokenizer, (tokenizer,)

    CachedTokenizer.__name__ = f"Cached{tokenizer.__class__.__name__}"

    cached_tokenizer.__class__ = CachedTokenizer
    return cached_tokenizer


class CachedHfTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        **kwargs,
    ) -> HfTokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_repo_id,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                cache_dir=download_dir,
                **kwargs,
            )
        except ValueError as e:
            # If the error pertains to the tokenizer class not existing or not
            # currently being imported,
            # suggest using the --trust-remote-code flag.
            if not trust_remote_code and (
                "does not exist or is not currently imported." in str(e)
                or "requires you to execute the tokenizer file" in str(e)
            ):
                err_msg = (
                    "Failed to load the tokenizer. If the tokenizer "
                    "is a custom tokenizer not yet available in the "
                    "HuggingFace transformers library, consider "
                    "setting `trust_remote_code=True` in LLM or using "
                    "the `--trust-remote-code` flag in the CLI. If the "
                    "model was created with a newer version of "
                    "transformers, consider upgrading: "
                    "`uv pip install --upgrade transformers`"
                )
                raise RuntimeError(err_msg) from e
            else:
                raise e

        # The special_tokens in tokenizer should also be
        # controlled by do_lower_case in encoder_config
        encoder_config = get_sentence_transformer_tokenizer_config(
            path_or_repo_id, revision
        )
        if isinstance(encoder_config, dict) and encoder_config.get(
            "do_lower_case", False
        ):
            special_tokens_map = {
                k: v.lower() for k, v in tokenizer.special_tokens_map.items()
            }
            tokenizer.add_special_tokens(special_tokens_map)

        return get_cached_tokenizer(tokenizer)
