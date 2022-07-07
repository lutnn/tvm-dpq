#include "convolution.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"
#include "amm_convolution_make.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(AmmConv2DAttrs);

// define relationship, we simplify the expression in the current design.
bool AmmConv2DRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
    // data, bias, centeroids, lut, scale
    ICHECK_EQ(types.size(), 6);
    const auto* data = types[0].as<TensorTypeNode>();
    const auto* bias = types[1].as<TensorTypeNode>();
    const auto* centroids = types[2].as<TensorTypeNode>();
    const auto* lut = types[3].as<TensorTypeNode>();
    const auto* scale = types[4].as<TensorTypeNode>();
    
    if (data == nullptr) return false;
    static const Layout kNCHW("NCHW");
    Layout kOIHW("OIHW");

    // set output shape and type
    const auto* param = attrs.as<AmmConv2DAttrs>();
    auto dtype = param->out_dtype;
    auto output_shape = param->output_shape; 
    
    // set output type 
    reporter->Assign(types[5], TensorType(output_shape, dtype));

    return true;
}


// register global 
TVM_REGISTER_GLOBAL("relay.op.nn._make.amm_conv2d")
.set_body_typed([]( Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,
                    Array<IndexExpr> subvec_len, Array<IndexExpr> output_shape, Array<IndexExpr> kernel_size, 
                    Array<IndexExpr> strides, Array<IndexExpr> padding, 
                    String data_layout, String kernel_layout, String out_layout, DataType out_dtype) {
    return MakeAmmConv<AmmConv2DAttrs>(data, bias, centroids, lut, scale,  
                                subvec_len, output_shape, kernel_size, strides, padding, 
                                data_layout, kernel_layout, out_layout, out_dtype, "nn.amm_conv2d"); 
}); 



RELAY_REGISTER_OP("nn.amm_conv2d")
    .describe(R"code(AMM 2D convolution layer (e.g. spatial convolution over images).

This layer creates a amm convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: 
- **bias**: 
- **centroids**:
- **lut**:
- **scale**:

Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AmmConv2DAttrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("bias", "Tensor", "The bias tensor.")
    .add_argument("centroids", "Tensor", "The centroid tensor.")
    .add_argument("lut", "Tensor", "The lut tensor.")
    .add_argument("scale", "Tensor", "The scale tensor.")
    .set_support_level(2)
    .add_type_rel("AmmConv2D", AmmConv2DRel);
    // set attribute to TOpPattern, we fuse amm op by ourselves. 
    // .set_attr<TOpPattern>("TOpPattern", kOpaque);
    // .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<AmmConv2DAttrs>); 
}
}

