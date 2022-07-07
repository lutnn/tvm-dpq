# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""AMM Convolution schedule on ARM"""

import os.path as osp
import sys
sys.path.append(osp.join(osp.dirname(__file__), "../../../../3rdparty/blink-mm-kernels"))  # nopep8

from tvm import te, autotvm
from ..utils import traverse_inline

from blink_mm_kernels.fused_dist_argmin_fp32 import *
from blink_mm_kernels.scan import *


@autotvm.register_topi_compute("amm_conv2d_int8.x86")
def compute_amm_conv2d_int8(
        cfg,
        data,
        bias,
        centroids,
        lut,
        scale,
        subvec_len,
        output_shape,
        kernel_size,
        strides,
        padding,
        data_layout,
        kernel_layout,
        out_layout,
        out_dtype
):
    """Compute amm conv2d int8"""
    argmin = compute_fused_dist_argmin_fp32(
        cfg, data, centroids, kernel_size, strides, padding)
    return compute_scan(
        cfg, argmin, lut, output_shape, scale, bias)


@autotvm.register_topi_schedule("amm_conv2d_int8.x86")
def schedule_amm_conv2d_int8(cfg, outs):
    """Schedule amm conv2d int8 strategy"""
    s = te.create_schedule([x.op for x in outs])
    assert len(outs) == 1

    def _callback(op):
        # schedule amm_conv2d
        if "amm_conv2d_int8" in op.tag:
            output = op.output(0)
            nonlocal s
            s, argmin = schedule_scan(cfg, s, output, "x86")
            s, _ = schedule_fused_dist_argmin_fp32(cfg, s, argmin)

    traverse_inline(s, outs[0].op, _callback)

    return s
