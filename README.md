# Life3d_by_YexuanLin
Here are LYX <并行程序设计>课程的CUDA编程相关作业

# 测试实验是在实验室的NVIDIA Tesla V100上进行的

life3d.cu 是原始的串行版本。
N = 256 T = 32下：时间为36s左右

life3d_lyx.cu 是初始改良的并行版本 会不断更新

life3d_lyx_1.0.cu 是通过将三维矩阵三层for循环，利用cuda改写成kernel函数实现并行。
N = 256 T = 32下：时间为0.65s左右（相比串行版本加速55.4倍）

life3d_lyx_2.0.cu 是在1.0基础上，通过extern __shared__ char shared_current[]，动态分配共享内存进行实现。
N = 256 T = 32下：时间为0.0095s左右（相比1.0基础版本加速68.4倍）

life3d_lyx_3.0.cu 是在2.0基础上，通过完全规避潜在的bank conflicts & loop unrolling，进一步加速。
N = 256 T = 32下：时间为0.0091s左右（相比2.0共享内存加速1.04倍）

# 总计提升4020倍！
