/*-----------------------------------------------
 * 个人信息填写
 * 学号: SA23234012
 * 姓名: 林业轩
 * 邮箱: lyx0724@mail.ustc.edu.cn
 ------------------------------------------------*/
#include <cuda_runtime.h>  // CUDA runtime functions
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using std::cin;
using std::cout; 
using std::endl;
using std::ifstream;
using std::ofstream;

// 宏定义用于错误检查
#define CUDA_CHECK_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA内核函数：计算下一个时刻的宇宙状态，使用共享内存优化并消除Bank冲突
__global__ void life3d_kernel_shared(int N, const char *current, char *next)
{
    // 定义线程块尺寸
    const int BLOCK_SIZE = 8; // 每个维度的线程数

    // 计算全局线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    int z = blockIdx.z * BLOCK_SIZE + tz;

    // 定义共享内存尺寸，包括1层halo和填充
    // 增加填充字节以消除Bank冲突
    extern __shared__ char shared_current[];
    const int SHARED_DIM = BLOCK_SIZE + 3; // 增加1个填充字节

    // 计算共享内存中的索引
    int shared_x = tx + 1;
    int shared_y = ty + 1;
    int shared_z = tz + 1;

    // 加载主数据到共享内存
    if (x < N && y < N && z < N)
        shared_current[(shared_x * SHARED_DIM + shared_y) * SHARED_DIM + shared_z] = current[x * N * N + y * N + z];
    else
        shared_current[(shared_x * SHARED_DIM + shared_y) * SHARED_DIM + shared_z] = 0;

    // 加载X方向的halo
    if (tx == 0 && x > 0)
        shared_current[(0 * SHARED_DIM + shared_y) * SHARED_DIM + shared_z] = current[((x - 1 + N) % N) * N * N + y * N + z];
    if (tx == BLOCK_SIZE - 1 && x < N - 1)
        shared_current[((BLOCK_SIZE + 1) * SHARED_DIM + shared_y) * SHARED_DIM + shared_z] = current[((x + 1) % N) * N * N + y * N + z];

    // 加载Y方向的halo
    if (ty == 0 && y > 0)
        shared_current[(shared_x * SHARED_DIM + 0) * SHARED_DIM + shared_z] = current[x * N * N + ((y - 1 + N) % N) * N + z];
    if (ty == BLOCK_SIZE - 1 && y < N - 1)
        shared_current[(shared_x * SHARED_DIM + (BLOCK_SIZE + 1)) * SHARED_DIM + shared_z] = current[x * N * N + ((y + 1) % N) * N + z];

    // 加载Z方向的halo
    if (tz == 0 && z > 0)
        shared_current[(shared_x * SHARED_DIM + shared_y) * SHARED_DIM + 0] = current[x * N * N + y * N + ((z - 1 + N) % N)];
    if (tz == BLOCK_SIZE - 1 && z < N - 1)
        shared_current[(shared_x * SHARED_DIM + shared_y) * SHARED_DIM + (BLOCK_SIZE + 1)] = current[x * N * N + y * N + ((z + 1) % N)];

    // 同步线程，确保所有共享内存数据已加载
    __syncthreads();

    // 计算存活邻居数，使用手动循环展开优化
    if (x < N && y < N && z < N)
    {
        int alive = 0;

        // 预计算共享内存索引基址
        const int base = (shared_x * SHARED_DIM + shared_y) * SHARED_DIM + shared_z;

        // 手动展开邻居访问，避免循环开销
        alive += shared_current[base - SHARED_DIM - SHARED_DIM - 1]; // (-1, -1, -1)
        alive += shared_current[base - SHARED_DIM - SHARED_DIM    ]; // (-1, -1,  0)
        alive += shared_current[base - SHARED_DIM - SHARED_DIM + 1]; // (-1, -1, +1)

        alive += shared_current[base - SHARED_DIM              - 1]; // (-1,  0, -1)
        // Skip center (-1,  0,  0) since current cell is skipped
        alive += shared_current[base - SHARED_DIM              + 1]; // (-1,  0, +1)

        alive += shared_current[base - SHARED_DIM + SHARED_DIM - 1]; // (-1, +1, -1)
        alive += shared_current[base - SHARED_DIM + SHARED_DIM    ]; // (-1, +1,  0)
        alive += shared_current[base - SHARED_DIM + SHARED_DIM + 1]; // (-1, +1, +1)

        alive += shared_current[base              - SHARED_DIM - 1]; // ( 0, -1, -1)
        alive += shared_current[base              - SHARED_DIM    ]; // ( 0, -1,  0)
        alive += shared_current[base              - SHARED_DIM + 1]; // ( 0, -1, +1)

        // Skip center (0, 0, 0)

        alive += shared_current[base              + SHARED_DIM - 1]; // ( 0, +1, -1)
        alive += shared_current[base              + SHARED_DIM    ]; // ( 0, +1,  0)
        alive += shared_current[base              + SHARED_DIM + 1]; // ( 0, +1, +1)

        alive += shared_current[base + SHARED_DIM - SHARED_DIM - 1]; // (+1, -1, -1)
        alive += shared_current[base + SHARED_DIM - SHARED_DIM    ]; // (+1, -1,  0)
        alive += shared_current[base + SHARED_DIM - SHARED_DIM + 1]; // (+1, -1, +1)

        alive += shared_current[base + SHARED_DIM              - 1]; // (+1,  0, -1)
        // Skip center (+1,  0,  0)
        alive += shared_current[base + SHARED_DIM              + 1]; // (+1,  0, +1)

        alive += shared_current[base + SHARED_DIM + SHARED_DIM - 1]; // (+1, +1, -1)
        alive += shared_current[base + SHARED_DIM + SHARED_DIM    ]; // (+1, +1,  0)
        alive += shared_current[base + SHARED_DIM + SHARED_DIM + 1]; // (+1, +1, +1)

        // 计算当前细胞状态
        char current_state = shared_current[base];
        if (current_state && (alive < 5 || alive > 7))
            next[x * N * N + y * N + z] = 0;
        else if (!current_state && alive == 6)
            next[x * N * N + y * N + z] = 1;
        else
            next[x * N * N + y * N + z] = current_state;
    }
}

// 读取输入文件
__host__ void read_file(const char *input_file, char *buffer, int N)
{
    ifstream file(input_file, ios::binary | ios::ate);
    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << input_file << endl;
        exit(EXIT_FAILURE);
    }
    streamsize file_size = file.tellg();
    if (file_size != N * N * N * sizeof(char))
    {
        cerr << "Error: File size does not match matrix dimensions." << endl;
        exit(EXIT_FAILURE);
    }
    file.seekg(0, ios::beg);
    if (!file.read(buffer, file_size))
    {
        cerr << "Error: Could not read file " << input_file << endl;
        exit(EXIT_FAILURE);
    }
    file.close();
}

// 写入输出文件
__host__ void write_file(const char *output_file, const char *buffer, int N)
{
    ofstream file(output_file, ios::binary | ios::trunc);
    if (!file)
    {
        cerr << "Error: Could not open file " << output_file << endl;
        exit(EXIT_FAILURE);
    }
    file.write(buffer, N * N * N * sizeof(char));
    file.close();
}

// 存活细胞数
__host__ int population(int N, const char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
__host__ void print_universe(int N, const char *universe, const char *str)
{
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    cout << str;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (universe[x * N * N + y * N + z])
                    cout << "O ";
                else
                    cout << "* ";
            }
            cout << endl;
        }
        cout << endl;
    }
    cout << "population: " << population(N, universe) << endl;
}

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << "Usage: " << argv[0] << " N T input_file output_file" << endl;
        return EXIT_FAILURE;
    }

    int N = stoi(argv[1]);        // 矩阵边长
    int T = stoi(argv[2]);        // 时间步数
    const char *input_file = argv[3];   // 输入文件路径
    const char *output_file = argv[4];  // 输出文件路径

    // 分配主机内存
    char *universe = (char *)malloc(N * N * N * sizeof(char));
    if (universe == nullptr)
    {
        cerr << "Error: Unable to allocate memory for universe." << endl;
        return EXIT_FAILURE;
    }

    // 读取初始宇宙状态
    read_file(input_file, universe, N);

    // 计算初始存活细胞数
    int start_pop = population(N, universe);

    // 分配设备内存
    char *d_current, *d_next;
    size_t size = N * N * N * sizeof(char);
    cudaError_t err = cudaMalloc((void**)&d_current, size);
    CUDA_CHECK_ERROR(err);
    err = cudaMalloc((void**)&d_next, size);
    CUDA_CHECK_ERROR(err);

    // 将初始宇宙状态从主机复制到设备
    err = cudaMemcpy(d_current, universe, size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);

    // 定义线程块和网格的大小
    const int BLOCK_SIZE = 8; // 每个维度的线程数，可以根据需要调整
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // 计算共享内存大小，包括halo层和填充
    size_t shared_mem_size = (BLOCK_SIZE + 3) * (BLOCK_SIZE + 3) * (BLOCK_SIZE + 3) * sizeof(char);

    // 记录开始时间
    auto start_time = chrono::high_resolution_clock::now();

    // 运行CUDA并行化的生命游戏
    for (int t = 0; t < T; t++)
    {
        // 启动CUDA内核
        life3d_kernel_shared<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(N, d_current, d_next);

        // 检查内核是否有错误
        err = cudaGetLastError();
        CUDA_CHECK_ERROR(err);

        // 同步设备，确保内核执行完成
        err = cudaDeviceSynchronize();
        CUDA_CHECK_ERROR(err);

        // 交换当前和下一个宇宙状态指针
        char *temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    // 记录结束时间
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end_time - start_time;

    // 将最终的宇宙状态从设备复制回主机
    err = cudaMemcpy(universe, d_current, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err);

    // 计算最终存活细胞数
    int final_pop = population(N, universe);

    // 写入最终宇宙状态到输出文件
    ofstream outfile(output_file, ios::binary | ios::trunc);
    if (!outfile)
    {
        cerr << "Error: Could not open file " << output_file << endl;
        free(universe);
        cudaFree(d_current);
        cudaFree(d_next);
        return EXIT_FAILURE;
    }
    outfile.write(universe, N * N * N * sizeof(char));
    outfile.close();

    // 打印结果
    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cells per sec: " << ((double)T * N * N * N) / time << endl;

    // 可选：打印矩阵（仅N较小时）
    // print_universe(N, universe, "Final Universe:\n");

    // 释放内存
    free(universe);
    cudaFree(d_current);
    cudaFree(d_next);

    return 0;
}
