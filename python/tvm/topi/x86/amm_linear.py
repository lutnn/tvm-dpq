# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""AMM Linear schedule on x86"""


import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), "../../../../3rdparty/blink-mm-kernels"))  # nopep8

from tvm import te, autotvm
from ..utils import traverse_inline

from blink_mm_kernels.fused_dist_argmin_fp32 import *
from blink_mm_kernels.scan import *


@autotvm.register_topi_compute("amm_linear.cpu")
def compute_amm_linear(
    cfg,
    data,
    bias,
    centroids,
    lut,
    scale,
    subvec_len,
    out_dtype
):
    """Compute amm linear"""
    argmin = compute_fused_dist_argmin_fp32(
        cfg, data, centroids, None, None, None)
    batch, samples, _ = data.shape
    out_features = lut.shape[0]
    return compute_scan(
        cfg, argmin, lut, (batch, samples, out_features), scale, bias)


@autotvm.register_topi_schedule("amm_linear.cpu")
def schedule_amm_linear(cfg, outs):
    """Schedule amm linear strategy"""
    s = te.create_schedule([x.op for x in outs])
    assert len(outs) == 1

    def _callback(op):
        # schedule amm_linear
        if "amm_linear" in op.tag:
            output = op.output(0)
            nonlocal s
            s, argmin = schedule_scan(cfg, s, output, "x86")
            s, _ = schedule_fused_dist_argmin_fp32(cfg, s, argmin)

            if outs[0] != output:
                # erf
                fused = s[outs[0]].fuse(*s[outs[0]].op.axis)
                s[outs[0]].parallel(fused)

    traverse_inline(s, outs[0].op, _callback)

    return s
