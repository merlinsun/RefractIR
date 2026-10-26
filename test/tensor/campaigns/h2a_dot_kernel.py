"""Minimal standalone dot kernel for the H2a compile-only capability check.

All operands are 16x16 f32 tiles. `input_precision="ieee"` fixes exact f32
semantics so the V100 lowering path can be inspected without TF32.
"""

import triton
import triton.language as tl


@triton.jit
def kernel(a_ptr, b_ptr, c_ptr):
  rows = tl.arange(0, 16)
  columns = tl.arange(0, 16)
  offsets = rows[:, None] * 16 + columns[None, :]
  a = tl.load(a_ptr + offsets)
  b = tl.load(b_ptr + offsets)
  result = tl.dot(a, b, input_precision="ieee")
  tl.store(c_ptr + offsets, result)
