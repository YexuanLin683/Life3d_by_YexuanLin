/*-----------------------------------------------
 * 请在此处填写你的个人信息
 * 学号: SA23234012
 * 姓名: 林业轩
 * 邮箱: lyx0724@mail.ustc.edu.cn
 ------------------------------------------------*/
#include <cuda_runtime.h>  // for CUDA runtime functions
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#define AT(x, y, z) universe[(x) * N * N + (y) * N + z]

using namespace std;
using std::cin;
using std::cout; 
using std::endl;
using std::ifstream;
using std::ofstream;

// CUDA错误检查宏
#define CUDA_CHECK_ERROR(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d) at %s:%d\n", \
                cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// 存活细胞数
__host__ int population(int N, char *universe)
{
    int result = 0;
    for (int i = 0; i < N * N * N; i++)
        result += universe[i];
    return result;
}

// 打印世界状态
__host__ void print_universe(int N, char *universe)
{
    // 仅在N较小(<= 32)时用于Debug
    if (N > 32)
        return;
    for (int x = 0; x < N; x++)
    {
        for (int y = 0; y < N; y++)
        {
            for (int z = 0; z < N; z++)
            {
                if (AT(x, y, z))
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

// CUDA内核函数：计算下一个时刻的宇宙状态
__global__ void life3d_kernel(int N, const char *current, char *next)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N * N;
    
    if (idx >= total)
        return;

    int x = idx / (N * N);
    int y = (idx % (N * N)) / N;
    int z = idx % N;

    int alive = 0;
    // 计算邻居的存活数
    for (int dx = -1; dx <= 1; dx++)
        for (int dy = -1; dy <= 1; dy++)
            for (int dz = -1; dz <= 1; dz++)
            {
                if (dx == 0 && dy == 0 && dz == 0)
                    continue;
                int nx = (x + dx + N) % N;
                int ny = (y + dy + N) % N;
                int nz = (z + dz + N) % N;
                alive += current[nx * N * N + ny * N + nz];
            }

    char current_state = current[idx];
    if (current_state && (alive < 5 || alive > 7))
        next[idx] = 0;
    else if (!current_state && alive == 6)
        next[idx] = 1;
    else
        next[idx] = current_state;
}

// 核心计算代码，将世界向前推进T个时刻（CUDA并行版本）
__host__ void life3d_run_cuda(int N, char *universe, int T)
{
    char *d_current, *d_next;
    size_t size = N * N * N * sizeof(char);

    // 分配设备内存
    cudaError_t err = cudaMalloc((void**)&d_current, size);
    CUDA_CHECK_ERROR(err);
    err = cudaMalloc((void**)&d_next, size);
    CUDA_CHECK_ERROR(err);

    // 将初始宇宙状态从主机复制到设备
    err = cudaMemcpy(d_current, universe, size, cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR(err);

    // 定义线程块和网格的大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (N * N * N + threadsPerBlock - 1) / threadsPerBlock;

    for (int t = 0; t < T; t++)
    {
        // 启动CUDA内核
        life3d_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, d_current, d_next);
        
        // 检查内核是否有错误
        err = cudaGetLastError();
        CUDA_CHECK_ERROR(err);

        // 交换当前和下一个宇宙状态指针
        char *temp = d_current;
        d_current = d_next;
        d_next = temp;
    }

    // 将最终的宇宙状态从设备复制回主机
    err = cudaMemcpy(universe, d_current, size, cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR(err);

    // 释放设备内存
    cudaFree(d_current);
    cudaFree(d_next);
}

// 读取输入文件
__host__ void read_file(char *input_file, char *buffer, int N)
{
    ifstream file(input_file, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        cout << "Error: Could not open file " << input_file << endl;
        exit(1);
    }
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer, file_size))
    {
        std::cerr << "Error: Could not read file " << input_file << std::endl;
        exit(1);
    }
    file.close();
}

// 写入输出文件
__host__ void write_file(char *output_file, char *buffer, int N)
{
    ofstream file(output_file, std::ios::binary | std::ios::trunc);
    if (!file)
    {
        cout << "Error: Could not open file " << output_file << endl;
        exit(1);
    }
    file.write(buffer, N * N * N);
    file.close();
}

int main(int argc, char **argv)
{
    // cmd args
    if (argc < 5)
    {
        cout << "usage: ./life3d N T input output" << endl;
        return 1;
    }
    int N = std::stoi(argv[1]);
    int T = std::stoi(argv[2]);
    char *input_file = argv[3];
    char *output_file = argv[4];

    // 分配主机内存
    char *universe = (char *)malloc(N * N * N * sizeof(char));
    if (universe == NULL)
    {
        cout << "Error: Unable to allocate memory for universe." << endl;
        return 1;
    }

    // 读取初始宇宙状态
    read_file(input_file, universe, N);

    // 计算初始存活细胞数
    int start_pop = population(N, universe);

    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    // 运行CUDA并行化的生命游戏
    life3d_run_cuda(N, universe, T);

    // 记录结束时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end_time - start_time;

    // 计算最终存活细胞数
    int final_pop = population(N, universe);

    // 写入最终宇宙状态
    write_file(output_file, universe, N);

    // 打印结果
    cout << "start population: " << start_pop << endl;
    cout << "final population: " << final_pop << endl;
    double time = duration.count();
    cout << "time: " << time << "s" << endl;
    cout << "cells per sec: " << (double)(T) / time * N * N * N << endl;

    // 释放主机内存
    free(universe);

    return 0;
}

