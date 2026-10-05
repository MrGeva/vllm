# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-split planning (on by default) for the MiniMax-M3 MSA prefill index score.

The stock builder plans the prefill ``fmha_sm100`` OnlyScore pass with
``num_kv_splits=1``. With one index head per rank (TP4), a prefill chunk then
runs one work item per 256-row query tile, each scanning the request's whole
cached index-K serially: a 512-token chunk keeps 2 of 148 SMs busy however long
the context is. fmha_sm100 honours split-KV only for 128-row query tiles
(``max_qo_len <= 128``), so this slices every prefill request into
pseudo-requests of at most 128 query tokens (each with its own bottom-right
causal prefix and page list) and plans them with ``num_kv_splits > 1``. Each
work item writes the max logit of every 128-key block it visits at the block's
absolute index; a block belongs to exactly one split and the max needs no
cross-split merge, so the scores (and hence the top-k) are unchanged.

On by default with ``splits="auto"`` and no self-test or verification. Disable
with ``--additional-config '{"aria_msa_prefill_kv_split": false}'``; ``true``
selects the defaults; a dict overrides any of:

- ``splits``: ``"auto"`` (default) sizes the splits so that the slices times
  splits give ``ctas_per_sm`` x SMs work items; an int > 1 fixes them; -1 uses
  fmha_sm100's own cost-model auto split (one extra host sync per step).
- ``ctas_per_sm``: work items per SM for ``"auto"`` (default 2).
- ``max_prefill_tokens``: use the stock plan above this many prefill tokens
  (default 16384).
- ``selftest``: at startup, plan random requests both ways on this GPU and
  compare scores and top-k bit for bit; any mismatch or error keeps the stock
  plan. This also JIT-compiles the split variants before serving; without it
  they compile on first use (default false).
- ``verify``: number of live prefill steps whose first indexer layer is
  re-scored with the stock plan and compared on TP rank 0 (default 0).

Results are printed as ``ARIA_MSA_KVSPLIT_SELFTEST`` / ``ARIA_MSA_KVSPLIT_VERIFY``
JSON lines.
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

CONFIG_KEY = "aria_msa_prefill_kv_split"
# fmha_sm100 runs split-KV only with 128-row query tiles.
SLICE_LEN = 128
MAX_SPLITS = 64
# The plan caches a bf16 split-KV O workspace of q * splits * heads * 128
# elements even in OnlyScore mode; bound it.
WORKSPACE_BYTES = 64 << 20
# Verification allocates a [prefill_tokens, heads, max_k_tiles] scratch.
VERIFY_MAX_TOKENS = 4096
# Self-test verdicts per process, keyed by everything the test depends on
# (builders are constructed more than once, e.g. for the profiling KV cache).
_SELFTEST_CACHE: dict[tuple, bool] = {}


@dataclass
class KVSplitConfig:
    splits: int  # 0: SM-sized auto, -1: library auto, > 1: fixed
    ctas_per_sm: float = 2.0
    max_prefill_tokens: int = 16384
    selftest: bool = False
    verify: int = 0


def parse_config(vllm_config: VllmConfig) -> KVSplitConfig | None:
    additional_config = vllm_config.additional_config
    value = None
    if isinstance(additional_config, dict):
        value = additional_config.get(CONFIG_KEY)
    if value is False or value == 0:
        return None
    # Unset (or null) means on with the defaults, like true.
    if value is None or value is True:
        return KVSplitConfig(splits=0)
    if isinstance(value, int):
        if value < 2:
            raise ValueError(f"{CONFIG_KEY}={value}: an int must be >= 2")
        return KVSplitConfig(splits=value)
    if not isinstance(value, dict):
        raise ValueError(f"{CONFIG_KEY} must be a bool, an int or a dict")
    unknown = set(value) - {
        "splits",
        "ctas_per_sm",
        "max_prefill_tokens",
        "selftest",
        "verify",
    }
    if unknown:
        raise ValueError(f"{CONFIG_KEY}: unknown keys {sorted(unknown)}")
    splits = value.get("splits", "auto")
    if splits == "auto":
        splits = 0
    elif not isinstance(splits, int) or not (splits == -1 or splits >= 2):
        raise ValueError(f"{CONFIG_KEY}.splits must be 'auto', -1 or an int >= 2")
    return KVSplitConfig(
        splits=splits,
        ctas_per_sm=float(value.get("ctas_per_sm", 2.0)),
        max_prefill_tokens=int(value.get("max_prefill_tokens", 16384)),
        selftest=bool(value.get("selftest", False)),
        verify=int(value.get("verify", 0)),
    )


def slice_prefill(
    qo_lens: torch.Tensor, kv_lens: torch.Tensor, slice_len: int = SLICE_LEN
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cut each request's queries into consecutive slices of <= slice_len.

    ``qo_lens``/``kv_lens`` are CPU int tensors per request (kv includes the
    request's own queries). Returns per-slice ``(request, qo_len, kv_len)``
    where a slice's kv_len stops at its last query, so its bottom-right causal
    offset ``kv_len - qo_len`` is the absolute position of its first query.
    Slices keep request order, so their query rows are the request's rows.
    """
    qo = qo_lens.to(torch.int64)
    kv = kv_lens.to(torch.int64)
    num_slices = (qo + slice_len - 1) // slice_len
    req = torch.repeat_interleave(torch.arange(qo.numel()), num_slices)
    first = torch.cumsum(num_slices, 0) - num_slices
    idx = torch.arange(req.numel()) - torch.repeat_interleave(first, num_slices)
    start = idx * slice_len
    slice_qo = torch.clamp(qo[req] - start, max=slice_len)
    slice_kv = kv[req] - qo[req] + start + slice_qo
    return req, slice_qo.to(torch.int32), slice_kv.to(torch.int32)


def page_gather_index(
    slice_req: torch.Tensor,
    slice_kv: torch.Tensor,
    page_size: int,
    row_stride: int,
    row_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flat block-table index of every slice's pages, concatenated in slice
    order: slice i lists its request's first cdiv(kv_len_i, page_size) pages
    (the plan's kv_page_indptr is built the same way). With ``row_offsets``
    (per-request start of its pages in a gathered page list), the index into
    that list instead."""
    num_pages = (slice_kv.to(torch.int64) + page_size - 1) // page_size
    rows = torch.repeat_interleave(slice_req.to(torch.int64), num_pages)
    first = torch.cumsum(num_pages, 0) - num_pages
    cols = torch.arange(rows.numel()) - torch.repeat_interleave(first, num_pages)
    if row_offsets is not None:
        return row_offsets[rows] + cols
    return rows * row_stride + cols


def choose_num_splits(
    cfg: KVSplitConfig,
    num_slices: int,
    total_q: int,
    num_heads: int,
    num_sms: int,
) -> int:
    workspace_cap = WORKSPACE_BYTES // max(1, total_q * num_heads * 128 * 2)
    if cfg.splits == -1 and workspace_cap >= MAX_SPLITS:
        # The library's auto split uses at most MAX_SPLITS.
        return -1
    if cfg.splits > 1:
        splits = cfg.splits
    else:
        splits = math.ceil(cfg.ctas_per_sm * num_sms / max(1, num_slices))
    return max(1, min(splits, MAX_SPLITS, workspace_cap))


class PrefillKVSplitPlanner:
    """Builds the split prefill score plan; owns startup self-test state."""

    def __init__(
        self,
        cfg: KVSplitConfig,
        num_heads: int,
        page_size: int,
        max_k_tiles: int,
        device: torch.device,
    ) -> None:
        self.cfg = cfg
        self.num_heads = num_heads
        self.page_size = page_size
        self.max_k_tiles = max_k_tiles
        self.device = device
        self.num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        # Cleared when the startup self-test fails; the stock plan is used then.
        self.enabled = True
        logger.info_once(
            "MSA prefill KV split on: splits=%s ctas_per_sm=%s "
            "max_prefill_tokens=%d selftest=%s verify=%d",
            "auto" if cfg.splits == 0 else cfg.splits,
            cfg.ctas_per_sm,
            cfg.max_prefill_tokens,
            cfg.selftest,
            cfg.verify,
        )

    def plan(
        self,
        side_qo: torch.Tensor,
        side_kv: torch.Tensor,
        block_table: torch.Tensor,
        check_enabled: bool = True,
        row_offsets: torch.Tensor | None = None,
    ) -> tuple[dict, torch.Tensor, dict[str, Any]] | None:
        """Split plan + page table for the prefill side, or None for stock.

        ``side_qo``/``side_kv`` are CPU per-request lengths; ``block_table`` is
        the prefill side's [num_prefills, max_pages] device block table. The
        self-test passes ``check_enabled=False`` to skip the serving gates.
        With ``row_offsets`` (CPU, per request), the page table indexes the
        requests' gathered pages instead of the block table.
        """
        total_q = int(side_qo.sum())
        if check_enabled and (
            not self.enabled or total_q > self.cfg.max_prefill_tokens
        ):
            return None
        # A slice with an odd page count also writes -inf at block index
        # cdiv(kv_len, page_size); keep that inside the score row.
        if int(side_kv.max()) > (self.max_k_tiles - 1) * self.page_size:
            return None
        from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

        slice_req, slice_qo, slice_kv = slice_prefill(side_qo, side_kv)
        splits = choose_num_splits(
            self.cfg, slice_req.numel(), total_q, self.num_heads, self.num_sms
        )
        if splits == 1:
            return None
        plan = _fmha_sm100_plan(
            slice_qo,
            slice_kv,
            self.num_heads,
            num_kv_heads=1,
            qo_offset=slice_kv - slice_qo,  # absolute position of first query
            page_size=self.page_size,
            output_maxscore=True,
            causal=True,
            num_kv_splits=splits,
        )
        plan["max_k_tiles"] = self.max_k_tiles
        table = block_table if block_table.is_contiguous() else block_table.contiguous()
        flat = page_gather_index(
            slice_req, slice_kv, self.page_size, table.shape[1], row_offsets
        )
        if row_offsets is not None:
            page_table = flat.to(torch.int32).to(table.device, non_blocking=True)
        else:
            page_table = (
                table.view(-1)
                .index_select(0, flat.to(table.device, non_blocking=True))
                .to(torch.int32)
            )
        info = {
            "kvs_slices": slice_req.numel(),
            "kvs_splits": int(plan["num_kv_splits"]),
            "kvs_pages": flat.numel(),
        }
        return plan, page_table, info

    def selftest(
        self,
        head_dim: int,
        dtype: torch.dtype,
        force_begin_blocks: int,
        force_end_blocks: int,
        emit: bool,
    ) -> bool:
        """Score random paged requests with the stock and the split plan on
        this GPU and require identical max-scores and top-k. Also JIT-compiles
        the split variants before serving."""
        key = (
            self.device,
            self.num_heads,
            head_dim,
            dtype,
            self.cfg.splits,
            self.cfg.ctas_per_sm,
            force_begin_blocks,
            force_end_blocks,
        )
        if key in _SELFTEST_CACHE:
            self.enabled = _SELFTEST_CACHE[key]
            return self.enabled
        cases = (
            ("chunk512_ctx60k", [512], [60_000]),
            ("chunk48_ctx30k", [48], [30_000]),
            ("mixed3", [300, 1000, 129], [100_000, 5_000, 0]),
        )
        results: list[dict[str, Any]] = []
        ok = True
        try:
            gen = torch.Generator(device="cpu").manual_seed(1234)
            for name, qo_lens, prefix_lens in cases:
                result = self._selftest_case(
                    qo_lens,
                    prefix_lens,
                    head_dim,
                    dtype,
                    force_begin_blocks,
                    force_end_blocks,
                    gen,
                )
                result["case"] = name
                results.append(result)
                ok = ok and result["ok"]
            torch.cuda.synchronize(self.device)
        except Exception as e:  # noqa: BLE001 - any failure keeps stock
            logger.warning("MSA prefill KV-split self-test raised: %r", e)
            results.append({"error": repr(e)[:300]})
            ok = False
        self.enabled = ok
        _SELFTEST_CACHE[key] = ok
        if emit:
            write_record(
                "ARIA_MSA_KVSPLIT_SELFTEST",
                {"ok": ok, "enabled": ok, "num_sms": self.num_sms, "cases": results},
            )
        if ok:
            logger.info("MSA prefill KV-split self-test passed; split plan on")
        else:
            logger.warning("MSA prefill KV-split self-test failed; stock plan kept")
        return ok

    def _selftest_case(
        self,
        qo_lens: list[int],
        prefix_lens: list[int],
        head_dim: int,
        dtype: torch.dtype,
        force_begin_blocks: int,
        force_end_blocks: int,
        gen: torch.Generator,
    ) -> dict[str, Any]:
        page = self.page_size
        side_qo = torch.tensor(qo_lens, dtype=torch.int32)
        side_kv = side_qo + torch.tensor(prefix_lens, dtype=torch.int32)
        pages = [(int(k) + page - 1) // page for k in side_kv]
        total_pages = sum(pages)
        # Shuffled physical pages so the page tables are exercised.
        perm = torch.randperm(total_pages, generator=gen).to(torch.int32)
        block_table = torch.zeros(len(pages), max(pages), dtype=torch.int32)
        offset = 0
        for row, count in enumerate(pages):
            block_table[row, :count] = perm[offset : offset + count]
            offset += count
        block_table = block_table.to(self.device)
        k_cache = torch.randn(total_pages, page, head_dim, generator=gen).to(
            device=self.device, dtype=dtype
        )
        total_q = int(side_qo.sum())
        q = torch.randn(total_q, self.num_heads, head_dim, generator=gen).to(
            device=self.device, dtype=dtype
        )
        positions = torch.cat(
            [
                torch.arange(int(p), int(p) + int(n), dtype=torch.int32)
                for n, p in zip(side_qo, prefix_lens)
            ]
        )
        num_valid_pages = (positions // page + 1).to(self.device)

        stock_plan, stock_table = stock_prefill_plan(
            side_qo, side_kv, self.num_heads, page, self.max_k_tiles, block_table
        )
        split = self.plan(side_qo, side_kv, block_table, check_enabled=False)
        if split is None:
            return {"ok": False, "reason": "split plan not built"}
        split_plan, split_table, info = split
        k_pages = k_cache.view(total_pages, 1, page, head_dim)
        scale = head_dim**-0.5
        scores = []
        topks = []
        for plan, table in ((stock_plan, stock_table), (split_plan, split_table)):
            score = score_prefill(q, k_pages, plan, table, scale, self.max_k_tiles)
            scores.append(score)
            topks.append(
                topk_select(
                    score, num_valid_pages, force_begin_blocks, force_end_blocks
                )
            )
        valid = (
            torch.arange(self.max_k_tiles, device=self.device)[None, None, :]
            < num_valid_pages[:, None, None]
        )
        stock_finite = torch.isfinite(scores[0]) & valid
        result = {
            "q": total_q,
            "ctx_max": max(prefix_lens),
            **info,
            "score_mismatch_valid": int(((scores[0] != scores[1]) & valid).sum()),
            "score_mismatch_all": int((scores[0] != scores[1]).sum()),
            "finite_valid": int(stock_finite.sum()),
            "valid": int(valid.sum()),
            "topk_mismatch_rows": int(
                (topks[0] != topks[1]).any(dim=-1).any(dim=-1).sum()
            ),
        }
        result["ok"] = (
            # Library auto (-1) may legitimately pick no split.
            (info["kvs_splits"] > 1 or self.cfg.splits == -1)
            and result["score_mismatch_all"] == 0
            and result["topk_mismatch_rows"] == 0
            # Every causally valid block was scored.
            and result["finite_valid"] == result["valid"]
        )
        return result


def stock_prefill_plan(
    side_qo: torch.Tensor,
    side_kv: torch.Tensor,
    num_heads: int,
    page_size: int,
    max_k_tiles: int,
    block_table: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    """The stock builder's prefill plan and page table (num_kv_splits=1)."""
    from vllm.third_party.fmha_sm100.api import _fmha_sm100_plan

    plan = _fmha_sm100_plan(
        side_qo,
        side_kv,
        num_heads,
        num_kv_heads=1,
        qo_offset=side_kv - side_qo,
        page_size=page_size,
        output_maxscore=True,
        causal=True,
        num_kv_splits=1,
    )
    plan["max_k_tiles"] = max_k_tiles
    nvp = (side_kv + page_size - 1) // page_size
    cols = torch.arange(block_table.shape[1], device=block_table.device)
    valid = cols[None, :] < nvp.to(block_table.device)[:, None]
    return plan, block_table[valid].to(torch.int32)


def score_prefill(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    plan: dict,
    page_table: torch.Tensor,
    scale: float,
    max_k_tiles: int,
) -> torch.Tensor:
    from vllm.third_party.fmha_sm100.api import _fmha_sm100

    score = torch.full(
        (q.shape[0], q.shape[1], max_k_tiles),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    _fmha_sm100(
        q,
        k_pages,
        k_pages,
        plan,
        kv_indices=page_table,
        output_o=False,
        output_maxscore=True,
        sm_scale=scale,
        max_score=score,
    )
    return score


def topk_select(
    score: torch.Tensor,
    num_valid_pages: torch.Tensor,
    force_begin_blocks: int,
    force_end_blocks: int,
    topk: int = 16,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    from vllm.third_party.fmha_sm100.api import sparse_topk_select

    if output is None:
        output = torch.empty(
            score.shape[0], score.shape[1], topk, dtype=torch.int32, device=score.device
        )
    sparse_topk_select(
        score,
        topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=force_begin_blocks,
        force_end_blocks=force_end_blocks,
        output=output,
        max_score_layout="THK",
    )
    return output


def write_record(marker: str, payload: dict[str, Any]) -> None:
    record = (
        marker
        + " "
        + json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    sys.stdout.flush()
    os.write(sys.stdout.fileno(), record)
