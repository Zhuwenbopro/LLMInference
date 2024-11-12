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
