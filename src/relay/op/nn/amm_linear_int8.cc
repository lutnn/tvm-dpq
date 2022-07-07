#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <vector>

#include "../../transforms/infer_layout_utils.h"
#include "../op_common.h"
#include "amm_convolution_make.h"

namespace tvm {
namespace relay {

TVM_REGISTER_NODE_TYPE(AmmLinearAttrs);

template <typename T>
inline Expr MakeAmmLinearInt8(Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,
                        Array<IndexExpr> subvec_len, Array<IndexExpr> output_shape, DataType out_dtype, std::string op_name) {
  auto attrs = make_object<T>();
  attrs->output_shape = std::move(output_shape);
  attrs->subvec_len = std::move(subvec_len);
  attrs->out_dtype = std::move(out_dtype);

  const Op& op = Op::Get(op_name);
  return Call(op, {data, bias, centroids, lut, scale}, Attrs(attrs), {});
}

// define relationship, we simplify the expression in the current design.
bool AmmLinearInt8Rel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
               const TypeReporter& reporter) {
    // data, bias, centeroids, lut, scale
    ICHECK_EQ(types.size(), 6);
    const auto* data = types[0].as<TensorTypeNode>();
    const auto* bias = types[1].as<TensorTypeNode>();
    const auto* centroids = types[2].as<TensorTypeNode>();
    const auto* lut = types[3].as<TensorTypeNode>();
    const auto* scale = types[4].as<TensorTypeNode>();
    
    if (data == nullptr) return false;

    // set output shape and type
    const auto* param = attrs.as<AmmLinearAttrs>();
    auto dtype = param->out_dtype;
    auto output_shape = param->output_shape; 
    
    // set output type 
    reporter->Assign(types[5], TensorType(output_shape, dtype));

    return true;
}


// register global 
TVM_REGISTER_GLOBAL("relay.op.nn._make.amm_linear_int8")
.set_body_typed([]( Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,
                    Array<IndexExpr> subvec_len, Array<IndexExpr> output_shape, DataType out_dtype) {
    return MakeAmmLinearInt8<AmmLinearAttrs>(data, bias, centroids, lut, scale,  
                                subvec_len, output_shape, out_dtype, "nn.amm_linear_int8"); 
}); 



RELAY_REGISTER_OP("nn.amm_linear_int8")
    .describe(R"code(AMM Linear layer (e.g. Matrix Multiplication).

This layer creates a amm linear kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: 
- **bias**: 
- **centroids**:
- **lut**:
- **scale**:

Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,

)code" TVM_ADD_FILELINE)
    .set_attrs_type<AmmLinearInt8Attrs>()
    .set_num_inputs(5)
    .add_argument("data", "Tensor", "The input tensor.")
    .add_argument("bias", "Tensor", "The bias tensor.")
    .add_argument("centroids", "Tensor", "The centroid tensor.")
    .add_argument("lut", "Tensor", "The lut tensor.")
    .add_argument("scale", "Tensor", "The scale tensor.")
    .set_support_level(2)
    .add_type_rel("AmmLiear", AmmLinearInt8Rel);
    // set attribute to TOpPattern, we fuse amm op by ourselves. 
    // .set_attr<TOpPattern>("TOpPattern", kOpaque);
    // .set_attr<FInferCorrectLayout>("FInferCorrectLayout", ConvInferCorrectLayout<AmmConv2DAttrs>); 
}
}

