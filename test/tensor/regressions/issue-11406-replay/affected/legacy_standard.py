import triton
import triton.language as tl
from triton.language import core, math
jit = triton.jit
max = tl.max
sum = tl.sum

@core._tensor_member_fn
@jit
@math._add_math_1arg_docstr("softmax")
def softmax(x, dim=None, keep_dims=False, ieee_rounding=False):
    if dim is None:
        _dim: core.constexpr = 0
    else:
        _dim: core.constexpr = dim
    z = x - max(x, _dim, keep_dims=keep_dims)
    num = math.exp(z)
    den = sum(num, _dim, keep_dims=keep_dims)
    return math.fdiv(num, den, ieee_rounding)
