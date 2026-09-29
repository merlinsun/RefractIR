import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, out_ptr):
    _sir_x_d0 = tl.arange(0, 256)[:, None]
    _sir_x_d1 = tl.arange(0, 2)[None, :]
    _sir_x_offsets = _sir_x_d0 + _sir_x_d1 * 256
    _sir_input_load_tmp0 = tl.load(x_ptr + _sir_x_offsets)
    _sir_rows = tl.sum(_sir_input_load_tmp0, axis=1, keep_dims=False)
    _sir_acc = (_sir_rows * 0.0)
    _sir_i = 0
    _sir_i = tl.full((), 0, tl.int32)
    for _sir_i in tl.range(0, 2):
        _sir_input_load_tmp1 = tl.load(x_ptr + _sir_x_offsets)
        _sir_rows = tl.sum(_sir_input_load_tmp1, axis=1, keep_dims=False)
        _sir_acc = (_sir_acc + _sir_rows)
    _sir_i = tl.full((), 2, tl.int32)
    _sir_out_d0 = tl.arange(0, 256)
    _sir_out_offsets = _sir_out_d0
    tl.store(out_ptr + _sir_out_offsets, _sir_acc)
