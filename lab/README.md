```
zeke@zeke:~/cuda_$ nvcc -lcublas -o test test.cu
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
