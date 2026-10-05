# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuTe DSL consumer of the o_proj push mailbox (oproj_ar_push.py).

The o_proj GEMM in push mode (``_small_m_kernel.SmallMMxfp8GemmPush``)
multicasts every rank's bf16 partial into a Lamport mailbox
``[3, tp, capacity_m, hidden]`` of bf16 (-0.0 = empty). This kernel, one
cluster per token, polls its rank's copy of the current generation until
every element of its fragments has arrived, then computes what FlashInfer's
MNNVL one-shot all-reduce + residual + RMSNorm (``oneshotAllreduceFusionKernel``)
computes:

    s        = bf16(sum over ranks in rank order, fp32 from 0)
    residual = bf16(s + residual_in)
    normed   = bf16(residual * rsqrt(mean(bf16(residual^2)) + eps) * (1 + w))

``residual`` is bitwise equal to the stock kernel's; ``normed`` differs only
through the fp32 order of the sum of squares (cluster shape).

Every wait is bounded: a fragment that has not arrived ``timeout_ns`` after
the first poll (or at once, if an earlier call timed out) is read as zero,
counted in the device poison word and in a host-mapped error counter, and the
call completes. The GEMM read the generation before launching this kernel, so
each CTA reads it once at entry and the last CTA to do so advances it, off the
critical path; each thread re-arms the elements it consumed.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.model_executor.kernels.linear.mxfp8._small_m_kernel import (
    PUSH_GENERATIONS,
    PUSH_SENTINEL,
    PUSH_STAGE_WORD,
)

# int32 words of the stage state after the generation (PUSH_STAGE_WORD).
ARRIVE_WORD = 1
POISON_WORD = 2
STAGE_WORDS = 4
# bf16 elements per thread fragment: 16 bytes of mailbox per rank and of
# output.
VEC = 8
SENTINEL_X2 = PUSH_SENTINEL | (PUSH_SENTINEL << 16)
WARP = 32


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        args,
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v4(opcode: str, addr: Int64, loc, ip):
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l,~{memory}",
        loc,
        ip,
        side_effects=True,
    )
    return [
        Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ld_volatile_v4(addr: Int64, *, loc=None, ip=None):
    return _v4("ld.volatile.global.v4.u32", addr, loc, ip)


@dsl_user_op
def _ld_v4(addr: Int64, *, loc=None, ip=None):
    return _v4("ld.global.v4.u32", addr, loc, ip)


@dsl_user_op
def _st_v4(addr: Int64, w0, w1, w2, w3, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r,~{memory}",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ld_volatile_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l,~{memory}",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _st_relaxed_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.relaxed.gpu.global.u32 [$0], $1;",
        "l,r,~{memory}",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _atom_add_acq_rel(addr: Int64, value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Uint32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _red_add_sys(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.sys.global.add.u32 [$0], $1;",
        "l,r,~{memory}",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _sentinel_halves(word: Uint32, *, loc=None, ip=None) -> Uint32:
    """Nonzero if either bf16 half of ``word`` is still the empty sentinel."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 lo, hi;
                .reg .pred p, q;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                setp.eq.b32 p, lo, {PUSH_SENTINEL};
                setp.eq.b32 q, hi, {PUSH_SENTINEL};
                or.pred p, p, q;
                selp.b32 $0, 1, 0, p;
            }}
            """,
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _unset_to_zero(word: Uint32, *, loc=None, ip=None) -> Uint32:
    """``word`` with sentinel halves (never sent; only left by a timeout) as 0."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            f"""
            {{
                .reg .b32 lo, hi;
                .reg .pred p, q;
                and.b32 lo, $1, 0xFFFF;
                and.b32 hi, $1, 0xFFFF0000;
                setp.eq.b32 p, lo, {PUSH_SENTINEL};
                setp.eq.b32 q, hi, {PUSH_SENTINEL << 16};
                selp.b32 lo, 0, lo, p;
                selp.b32 hi, 0, hi, q;
                or.b32 $0, lo, hi;
            }}
            """,
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_lo(word: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 w; shl.b32 w, $1, 16; mov.b32 $0, w; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_hi(word: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 w; and.b32 w, $1, 0xFFFF0000; mov.b32 $0, w; }",
            "=f,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def _round_bf16(value: Float32, *, loc=None, ip=None) -> Float32:
    """``value`` rounded to bf16 (nearest even), returned as fp32."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h, z;
                .reg .b32 w;
                cvt.rn.bf16.f32 h, $1;
                mov.b16 z, 0;
                mov.b32 w, {z, h};
                mov.b32 $0, w;
            }
            """,
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [
                Float32(hi).ir_value(loc=loc, ip=ip),
                Float32(lo).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _rsqrt(value: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrtf()'s instruction."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(value).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.f32 $0, $1;",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_cluster_f32(
    local_addr: Int32, peer: Int32, value: Float32, *, loc=None, ip=None
):
    """Store ``value`` at ``local_addr``'s offset in cluster CTA ``peer``."""
    _asm(
        None,
        [
            Int32(local_addr).ir_value(loc=loc, ip=ip),
            Int32(peer).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b32 ra;
            mapa.shared::cluster.u32 ra, $0, $1;
            st.shared::cluster.f32 [ra], $2;
        }
        """,
        "r,r,f,~{memory}",
        loc,
        ip,
        side_effects=True,
    )


class OprojArPushConsumer:
    """All-reduce + residual add + GemmaRMSNorm of one pushed o_proj output.

    Grid (m, cluster_size), cluster (1, cluster_size): cluster = token; CTA c
    covers fragments (trip * cluster_size + c) * threads + tid of hidden / 8.
    ``wait_for_gemm`` makes the kernel wait for the GEMM grid to complete
    (griddepcontrol.wait) before polling, instead of only on the mailbox.
    """

    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        eps: float,
        weight_bias: float,
        wait_for_gemm: bool,
    ):
        if hidden % VEC:
            raise ValueError("hidden must be a multiple of 8")
        if threads % WARP or threads > 1024:
            raise ValueError("threads must be a warp multiple <= 1024")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.eps = eps
        self.weight_bias = weight_bias
        self.wait_for_gemm = wait_for_gemm
        self.fragments = hidden // VEC
        self.trips = -(-self.fragments // (cluster_size * threads))
        self.warps = threads // WARP

    def smem_bytes(self) -> int:
        return (self.warps + self.cluster_size + 3) * 4

    @cute.jit
    def __call__(
        self,
        mailbox: Int64,
        residual_in: Int64,
        gamma: Int64,
        residual_out: Int64,
        norm_out: Int64,
        stage: Int64,
        error_counter: Int64,
        timeout_ns: Int64,
        m: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            mailbox,
            residual_in,
            gamma,
            residual_out,
            norm_out,
            stage,
            error_counter,
            timeout_ns,
        ).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            smem=self.smem_bytes(),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        mailbox: Int64,
        residual_in: Int64,
        gamma: Int64,
        residual_out: Int64,
        norm_out: Int64,
        stage: Int64,
        error_counter: Int64,
        timeout_ns: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        grid_x, grid_y, _ = cute.arch.grid_dim()
        crank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        smem = cutlass.utils.SmemAllocator()
        warp_sums = smem.allocate_array(Float32, self.warps)
        cluster_sums = smem.allocate_array(Float32, self.cluster_size)
        # [generation, poisoned, timed out]
        shared_state = smem.allocate_array(Int32, 3)
        if tidx == 0:
            cute.arch.store(
                shared_state.llvm_ptr,
                Int32(_ld_volatile_u32(stage + 4 * PUSH_STAGE_WORD)),
            )
            cute.arch.store(
                (shared_state + 1).llvm_ptr,
                Int32(_ld_volatile_u32(stage + 4 * POISON_WORD)),
            )
            cute.arch.store((shared_state + 2).llvm_ptr, Int32(0))

        row = Int64(token) * self.hidden
        trips = self.trips
        gamma_f = cute.make_rmem_tensor(cute.make_layout((trips, VEC)), Float32)
        resid_f = cute.make_rmem_tensor(cute.make_layout((trips, VEC)), Float32)
        gamma_f.fill(0.0)
        resid_f.fill(0.0)
        # Gamma and the residual are inputs of earlier kernels: this kernel
        # launches only after every GEMM CTA passed its PDL wait.
        for trip in cutlass.range_constexpr(trips):
            frag = (trip * self.cluster_size + crank) * self.threads + tidx
            if frag < self.fragments:
                g = _ld_v4(gamma + Int64(frag) * VEC * 2)
                r = _ld_v4(residual_in + (row + Int64(frag) * VEC) * 2)
                for w in cutlass.range_constexpr(4):
                    gamma_f[trip, 2 * w] = _bf16_lo(g[w])
                    gamma_f[trip, 2 * w + 1] = _bf16_hi(g[w])
                    resid_f[trip, 2 * w] = _bf16_lo(r[w])
                    resid_f[trip, 2 * w + 1] = _bf16_hi(r[w])

        if cutlass.const_expr(self.wait_for_gemm):
            cute.arch.griddepcontrol_wait()

        cute.arch.barrier()
        generation = Uint32(cute.arch.load(shared_state.llvm_ptr, Int32))
        poisoned = cute.arch.load((shared_state + 1).llvm_ptr, Int32) != Int32(0)
        slots = Int64(generation) * self.tp * self.capacity_m
        # Every CTA has read the generation once it arrives; the last arrival
        # advances it for the next call (the GEMM read it before launching
        # this grid). Its latency overlaps the polling below.
        if tidx == 0:
            seen = _atom_add_acq_rel(stage + 4 * ARRIVE_WORD, Uint32(1))
            if seen == Uint32(grid_x * grid_y) - Uint32(1):
                _st_relaxed_u32(stage + 4 * ARRIVE_WORD, Uint32(0))
                _st_relaxed_u32(
                    stage + 4 * PUSH_STAGE_WORD,
                    (generation + Uint32(1)) % Uint32(PUSH_GENERATIONS),
                )

        words = cute.make_rmem_tensor(cute.make_layout((trips, self.tp, 4)), Uint32)
        words.fill(Uint32(0))
        start = _globaltimer()
        timed_out = cutlass.Boolean(False)
        for trip in cutlass.range_constexpr(trips):
            frag = (trip * self.cluster_size + crank) * self.threads + tidx
            pending = cutlass.Boolean(frag < self.fragments)
            while pending:
                pending = cutlass.Boolean(False)
                for src in cutlass.range_constexpr(self.tp):
                    got = _ld_volatile_v4(
                        mailbox
                        + (
                            (slots + src * self.capacity_m + Int64(token)) * self.hidden
                            + Int64(frag) * VEC
                        )
                        * 2
                    )
                    for w in cutlass.range_constexpr(4):
                        words[trip, src, w] = got[w]
                        pending = pending | (_sentinel_halves(got[w]) != Uint32(0))
                if pending:
                    expired = poisoned | (_globaltimer() - start > timeout_ns)
                    timed_out = timed_out | expired
                    pending = pending & ~expired
        if timed_out:
            cute.arch.store((shared_state + 2).llvm_ptr, Int32(1))
        cute.arch.barrier()
        cute.arch.griddepcontrol_launch_dependents()
        if tidx == 0:  # noqa: SIM102
            if cute.arch.load((shared_state + 2).llvm_ptr, Int32) != Int32(0):
                _atom_add_acq_rel(stage + 4 * POISON_WORD, Uint32(1))
                _red_add_sys(error_counter, Uint32(1))

        thread_sum = Float32(0.0)
        for trip in cutlass.range_constexpr(trips):
            frag = (trip * self.cluster_size + crank) * self.threads + tidx
            if frag < self.fragments:
                for src in cutlass.range_constexpr(self.tp):
                    _st_v4(
                        mailbox
                        + (
                            (slots + src * self.capacity_m + Int64(token)) * self.hidden
                            + Int64(frag) * VEC
                        )
                        * 2,
                        Uint32(SENTINEL_X2),
                        Uint32(SENTINEL_X2),
                        Uint32(SENTINEL_X2),
                        Uint32(SENTINEL_X2),
                    )
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for e in cutlass.range_constexpr(VEC):
                    acc = Float32(0.0)
                    for src in cutlass.range_constexpr(self.tp):
                        word = _unset_to_zero(words[trip, src, e // 2])
                        if cutlass.const_expr(e % 2 == 0):
                            acc = acc + _bf16_lo(word)
                        else:
                            acc = acc + _bf16_hi(word)
                    value = _round_bf16(_round_bf16(acc) + resid_f[trip, e])
                    # Keep the pre-norm value; the gamma slot is not reused.
                    resid_f[trip, e] = value
                    thread_sum = thread_sum + _round_bf16(value * value)
                for w in cutlass.range_constexpr(4):
                    packed[w] = _pack_bf16x2(
                        resid_f[trip, 2 * w], resid_f[trip, 2 * w + 1]
                    )
                _st_v4(
                    residual_out + (row + Int64(frag) * VEC) * 2,
                    packed[0],
                    packed[1],
                    packed[2],
                    packed[3],
                )

        # Sum of squares: warp butterfly, warps in order, CTAs in order.
        for offset in cutlass.range_constexpr(5):
            thread_sum = thread_sum + cute.arch.shuffle_sync_bfly(
                thread_sum, offset=1 << (4 - offset)
            )
        if lane == 0:
            cute.arch.store((warp_sums + warp).llvm_ptr, thread_sum)
        cute.arch.barrier()
        cta_sum = Float32(0.0)
        for w in cutlass.range_constexpr(self.warps):
            cta_sum = cta_sum + cute.arch.load((warp_sums + w).llvm_ptr, Float32)
        if tidx < self.cluster_size:
            _st_cluster_f32(Int32((cluster_sums + crank).toint()), Int32(tidx), cta_sum)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        total_sum = Float32(0.0)
        for c in cutlass.range_constexpr(self.cluster_size):
            total_sum = total_sum + cute.arch.load((cluster_sums + c).llvm_ptr, Float32)
        inv_rms = _rsqrt(total_sum / Float32(self.hidden) + Float32(self.eps))

        for trip in cutlass.range_constexpr(trips):
            frag = (trip * self.cluster_size + crank) * self.threads + tidx
            if frag < self.fragments:
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for w in cutlass.range_constexpr(4):
                    lo = (resid_f[trip, 2 * w] * inv_rms) * (
                        Float32(self.weight_bias) + gamma_f[trip, 2 * w]
                    )
                    hi = (resid_f[trip, 2 * w + 1] * inv_rms) * (
                        Float32(self.weight_bias) + gamma_f[trip, 2 * w + 1]
                    )
                    packed[w] = _pack_bf16x2(lo, hi)
                _st_v4(
                    norm_out + (row + Int64(frag) * VEC) * 2,
                    packed[0],
                    packed[1],
                    packed[2],
                    packed[3],
                )
