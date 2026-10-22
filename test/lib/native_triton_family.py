"""Eight frozen Triton kernels that mirror one SIR body, under four launches.

The native exploration arm keeps the control flow and addresses hand-written:
each kernel computes the same two-state add loop as `SOURCE` but varies a
`tl.range` option or an address hint, and each is launched under four warp
settings. The SIR source supplies only the signature, inputs, and layouts; the
emission record states `sir_role: signature-only`, so a reader never mistakes
the SIR text for the kernel that ran.
"""

import hashlib
from dataclasses import dataclass

SCHEMA = "refractir.native-triton-family.v1"

SOURCE = """\
fun @kernel(%x: tensor [4, 4] f32, %y: tensor [4, 4] f32) : tensor [4, 4] f32 {
  let mut %i: i32 = undef;
  let mut %acc_x: tensor [4, 4] f32 = undef;
  let mut %acc_y: tensor [4, 4] f32 = undef;
  let mut %out: tensor [4, 4] f32 = undef;
^entry:
  %acc_x = %x;
  %acc_y = %y;
  %i = 0;
  br ^header;
^header:
  br %i < 2, ^body, ^exit;
^body:
  %acc_x = tensor add %acc_x, %acc_y;
  %i = %i + 1;
  br ^header;
^exit:
  %out = tensor add %acc_x, %acc_y;
  ret %out;
}
"""

_TEMPLATE = """\
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, y_ptr, out_ptr):
    offsets = tl.arange(0, 16)
    acc_x = tl.load(x_ptr + offsets)
    acc_y = tl.load(y_ptr + offsets)
    {body}
    tl.store(out_ptr + offsets, acc_x + acc_y)
"""

VARIANTS = {
  "plain": "for _ in tl.range(0, 2):\n        acc_x = acc_x + acc_y",
  "range-stages-2": "for _ in tl.range(0, 2, num_stages=2):\n        acc_x = acc_x + acc_y",
  "range-stages-3": "for _ in tl.range(0, 2, num_stages=3):\n        acc_x = acc_x + acc_y",
  "range-unroll-2": "for _ in tl.range(0, 2, loop_unroll_factor=2):\n        acc_x = acc_x + acc_y",
  "range-no-acc-buffer": (
    "for _ in tl.range(0, 2, disallow_acc_multi_buffer=True):\n        acc_x = acc_x + acc_y"
  ),
  "hint-multiple-of": (
    "offsets = tl.multiple_of(offsets, 16)\n"
    "    for _ in tl.range(0, 2):\n        acc_x = acc_x + acc_y"
  ),
  "hint-max-contiguous": (
    "offsets = tl.max_contiguous(offsets, 16)\n"
    "    for _ in tl.range(0, 2):\n        acc_x = acc_x + acc_y"
  ),
  "range-flatten": "for _ in tl.range(0, 2, flatten=True):\n        acc_x = acc_x + acc_y",
}


@dataclass(frozen=True)
class NativeKernel:
  name: str
  source: str

  @property
  def sha256(self):
    return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


KERNELS = tuple(
    NativeKernel(name=name, source=_TEMPLATE.format(body=body))
    for name, body in VARIANTS.items()
)
CONFIGURATIONS = tuple(
    {"num_warps": warps, "num_stages": 1, "enable_fp_fusion": False}
    for warps in (1, 2, 4, 8)
)


def invocations():
  """Every kernel under every launch configuration."""
  for kernel in KERNELS:
    for configuration in CONFIGURATIONS:
      yield kernel, configuration


def triton_identity(environment, *, cache_mode="default"):
  """The Triton identity block a native record must state."""
  gpus = environment.get("gpus") or [{}]
  return {
    "version": str(environment.get("triton_version") or "unavailable"),
    "revision": str(environment.get("triton_source_revision") or "unavailable"),
    "torch": str(environment.get("torch_version") or "unavailable"),
    "device": str(gpus[0].get("name") or "unavailable"),
    "cache_mode": cache_mode,
  }


def manifest():
  """The frozen family, digesting the source and every kernel."""
  return {
    "schema": SCHEMA,
    "source_sha256": hashlib.sha256(SOURCE.encode("utf-8")).hexdigest(),
    "kernels": [
      {"name": kernel.name, "sha256": kernel.sha256} for kernel in KERNELS
    ],
    "configurations": [dict(configuration) for configuration in CONFIGURATIONS],
    "executions": len(KERNELS) * len(CONFIGURATIONS),
  }
