#include "wavefield.cuh"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

using namespace wavefield;

// 主测试函数
int main() {
    // --- 1. 定义波场和模拟参数 ---
    int Nx = 256, Ny = 256;             // FFT 网格点数
    float Lx_grid = 100.0f;           // 波场物理域 X 长度 (m)
    float Ly_grid = 100.0f;           // 波场物理域 Y 长度 (m)
    float Hs = 0.5f;                  // 有义波高 (m)
    float Tp = 3.5f;                 // 峰值周期 (s)
    float gamma = 3.3f;               // 峰度因子
    unsigned long seed = 1024;        // 随机数种子

    // 模拟时间参数
    float t_start = 0.0f;
    float t_end = 100.0f;              // 模拟总时长 (s)
    float dt = 0.1f;                  // 时间步长 (s)
    float t = t_start;

    // 查询点坐标
    float query_x = 0.0f;
    float query_y = 0.0f;

    std::cout << "--- 2D 波场单点持续查询测试程序 ---\n";
    std::cout << "网格: " << Nx << "x" << Ny << ", 域长: " << Lx_grid << "x" << Ly_grid << " m\n";
    std::cout << "波谱: Hs=" << Hs << "m, Tp=" << Tp << "s, Gamma=" << gamma << std::endl;
    std::cout << "查询点: (" << query_x << ", " << query_y << ") m\n";
    std::cout << "------------------------------------------\n";

    try {
        // --- 2. 实例化波场计算器 ---
        // 构造函数会自动完成内存分配和波谱 a(k) 的生成
        std::cout << "初始化 WaveFieldCalculator...\n";
        WaveFieldCalculator wave_calc(
            Nx, Ny, Lx_grid, Ly_grid, Hs, Tp, gamma, seed);
        std::cout << "初始化完成，开始持续查询。\n\n";

        // --- 3. 持续时间查询 ---

        // 设置输出格式
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(8) << "Time (s)" << " | "
                  << std::setw(12) << "Wave Height (m)" << " | "
                  << std::setw(18) << "Vel. Potential (m^2/s)" << "\n";
        std::cout << "--------------------------------------------------------\n";


        while (t <= t_end) {
            // 查询波高
            float eta = wave_calc.query_wave_height_point(query_x, query_y, t);

            // 查询速度势
            float phi = wave_calc.query_velocity_potential_point(query_x, query_y, t);

            // 打印结果
            std::cout << std::setw(8) << t << " | "
                      << std::setw(17) << eta << " | "
                      << std::setw(18) << phi << "\n";

            // 更新时间
            t += dt;
        }

    } catch (const std::exception& e) {
        // 捕获 C++ 异常
        std::cerr << "\n程序发生错误: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        // 捕获未知异常
        std::cerr << "\n程序发生未知错误。\n";
        return 1;
    }

    std::cout << "\n持续查询测试结束。\n";
    return 0;
}
