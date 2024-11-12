## cuBLAS_vs_kernel_MatrixMul.cu
```
zeke@zeke:~/cuda_$ nvcc -lcublas -o cuBLAS_vs_kernel_MatrixMul cuBLAS_vs_kernel_MatrixMul.cu
zeke@zeke:~/cuda_$ ./test
自定义核函数运行时间: 121.376 ms
cuBLAS（Tensor Core）运行时间: 13.3429 ms
最大误差: 0.567871
zeke@zeke:~/cuda_$ ./test
自定义核函数运行时间: 117.569 ms
cuBLAS（Tensor Core）运行时间: 4.84483 ms
最大误差: 0.567871
zeke@zeke:~/cuda_$ ./test
自定义核函数运行时间: 117.191 ms
cuBLAS（Tensor Core）运行时间: 4.9529 ms
最大误差: 0.567871
```
## vec_x_matrix_FP16.cu
```
zeke@zeke:~/cuda_$ nvcc -lcublas -o vec_x_matrix_FP16 vec_x_matrix_FP16.cu
zeke@zeke:~/cuda_$ ./test
运行时间: 11.2326 ms
结果正确！
```

## silu_fp16_bf16_vs_fp32.cu
fp16、bf16 进行运算的速度快是因为内存传输少。
```
zeke@zeke:~/cuda_$ nvcc -lcublas -o test test.cu
zeke@zeke:~/cuda_$ ./test
FP32 kernel execution time: 2.3976 ms
FP16 kernel execution time: 0.31232 ms
BF16 kernel execution time: 0.308224 ms
FP16 results are correct within acceptable error margin.
BF16 results are correct within acceptable error margin.
```
