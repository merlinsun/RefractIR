"""Minimal upstream shape and loop pattern for Triton issue #11186."""

import torch
import triton
import triton.language as tl


@triton.jit
def kernel(x_ptr, y_ptr):
  rows = tl.arange(0, 256)
  result = tl.zeros((256,), dtype=tl.float32)
  for _ in range(2):
    values = tl.load(x_ptr + rows[:, None] + 256 * tl.arange(0, 2)[None, :])
    result += tl.sum(values, axis=1)
  tl.store(y_ptr + rows, result)


def main():
  x = torch.arange(256, device="cuda", dtype=torch.float32).repeat(2)
  y = torch.empty((256,), device="cuda", dtype=torch.float32)
  kernel[(1,)](x, y, num_warps=1)
  torch.cuda.synchronize()
  expected = 4 * x[:256]
  mismatch = y != expected
  print(
    {
      "mismatches": int(mismatch.sum().item()),
      "actual_prefix": y[:8].tolist(),
      "expected_prefix": expected[:8].tolist(),
    }
  )
  if bool(mismatch.any()):
    raise SystemExit(1)


if __name__ == "__main__":
  main()
