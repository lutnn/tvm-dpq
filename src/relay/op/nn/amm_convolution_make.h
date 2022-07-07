/*!
 * \file src/relay/op/nn/amm_convolution_make.h
 * \brief utilities for creating amm convolution ops
 */
#ifndef TVM_RELAY_OP_NN_AMM_CONVOLUTION_MAKE_H_
#define TVM_RELAY_OP_NN_AMM_CONVOLUTION_MAKE_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/op.h>

#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
/*
    .set_body_typed([]( Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,
                        Array<IndexExpr> output_shape, Array<IndexExpr> kernel_size, Array<IndexExpr> strides,
                        Array<IndexExpr> padding, String data_layout, String kernel_layout, String out_layout, String out_dtype)
*/
template <typename T>
inline Expr MakeAmmConv(Expr data, Expr bias, Expr centroids, Expr lut, Expr scale,
                        Array<IndexExpr> subvec_len, Array<IndexExpr> output_shape, Array<IndexExpr> kernel_size, 
                        Array<IndexExpr> strides, Array<IndexExpr> padding, String data_layout, 
                        String kernel_layout, String out_layout, DataType out_dtype, std::string op_name) {
  auto attrs = make_object<T>();
  attrs->output_shape = std::move(output_shape);
  attrs->kernel_size = std::move(kernel_size);
  attrs->strides = std::move(strides);
  attrs->padding = std::move(padding);
  attrs->subvec_len = std::move(subvec_len);
  attrs->data_layout = std::move(data_layout);
  attrs->kernel_layout = std::move(kernel_layout);
  attrs->out_layout = std::move(out_layout);
  attrs->out_dtype = std::move(out_dtype);

  const Op& op = Op::Get(op_name);
  return Call(op, {data, bias, centroids, lut, scale}, Attrs(attrs), {});
}

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_OP_NN_AMM_CONVOLUTION_MAKE_H_
