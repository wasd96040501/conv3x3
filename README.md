# conv3x3
Vary fast 3x3 convolution on cpu, only with padding = 0, stride = 1, dialation = 1.<br>
This repo is aimed to share my idea with you.

---
# Results
Here are some benchmark results, compared to caffe2 convolution cpu implement.<br>
Platform: ubuntu16.04, haswell(intel i7 4702MQ @ 2.20GHz)

```
test case: 0
out_channel: 8 in_channel: 4 out_height: 32 out_width: 32
error:0
GFLOPS by Eigen is: 7.70078
GFLOPS by direct-conv  is: 15.2591


test case: 3
out_channel: 16 in_channel: 16 out_height: 64 out_width: 64
error:0
GFLOPS by Eigen is: 11.8056
GFLOPS by direct-conv  is: 29.8521


test case: 6
out_channel: 48 in_channel: 64 out_height: 256 out_width: 256
error:0
GFLOPS by Eigen is: 8.32179
GFLOPS by direct-conv  is: 27.3182

```

---
# Idea
My idea is simple.
<br><br>
In caffe2 cpu implement of convolution, 
Im2Col should be called to convert image into right hand side of a matrix multiplication, 
which will expand the memory usage up to 9x larger(3x3 conv). 
And this will lead to 9x more memory load when you execute matrix multiplication.
<br><br>
It is universally acknowledged that the most common performance limit to 
matrix multilication is the rate of operation / (load,store), 
which means some special (m, n, k) will lead to vary poor performance, because (m * n * k) / (m * k + n * k) is vary small,
unfortunately im2col will exactly leads to this condition (vary large n compared to m).
<br><br>
So, I decide to skip im2col, and it is exactly possible to implment when stride and dialation is 1.<br>
And also luckily, this can be done with little modification to gemm kernel.
<br><br>
(For readers not familiar with gemm, have a look at https://github.com/flame/blislab)<br>
In gemm, we divide the whole task into some small tasks to get a better use of cache, 
that is, we pick out some panel from matrix(packing), which is exactly fit the storage of L1, L2 or L3 cache.
Then we compute these tasks in panel without access main memory to get higher cache hits.
<br><br>
Here comes my idea.<br>
Convolution can be converted into matrix multiplication with right hand side be im2coled from image.<br>
Why not just do im2col in panel packing (mentioned above, in code, this is called pack_rhs), and skip explicit im2col?<br>
With that, we no longer need expand memory usage to 9x larger(3x3 conv), 
then increase the rate of operation / (load, store).

---
# About the code
This code is directly based on the gemm implement of Eigen.
So, I think on platform like arm, we can also obtain performance imporvement. <br>
However, this code uses Eigen::internal, which means it is unstable when Eigen upgrades.
In future, I will remove the dependency on Eigen.

--- 
# Problem
- In some special case, like when out_width is not multiple of 4, or out_channel not multiple of 8, performence decrease hugely
---
# Future
- More generic code generation for kernel like 5x5, 7x7... conv.
- Remove the constrain of padding = 0, which can be done with little modification to pack_rhs.
- Improve border case performance
- Some special stride and dialation
- Turn into a mature convolution package. 
