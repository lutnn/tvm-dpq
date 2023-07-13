# TVM for LUT-NN

LUT-NN relies on TVM to generate kernels.
We implement TVM computation graphs and schedules specific to LUT-NN in this repository.
This enables the fast compilation and inference of LUT-NN.
To install TVM for LUT-NN, please follow [TVM v0.9.0's document](https://tvm.apache.org/docs/v0.9.0/install/index.html).
You can also take a look at [TVM's original readme](README.tvm.md) for more information. 