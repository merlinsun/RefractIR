import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    _sir_x_d0 = tl.arange(0, 2)[:, None]
    _sir_x_d1 = tl.arange(0, 2)[None, :]
    _sir_x_offsets = _sir_x_d0 * 2 + _sir_x_d1
    _sir_x = tl.load(x_ptr + _sir_x_offsets)
    _sir_out = tl.softmax(_sir_x, dim=1)
    _sir_out_d0 = tl.arange(0, 2)[:, None]
    _sir_out_d1 = tl.arange(0, 2)[None, :]
    _sir_out_offsets = _sir_out_d0 * 2 + _sir_out_d1
    tl.store(out_ptr + _sir_out_offsets, _sir_out)
