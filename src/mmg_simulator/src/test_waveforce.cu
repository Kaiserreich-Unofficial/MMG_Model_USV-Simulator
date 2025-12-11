#include "waveforce.cuh"
#include "wavefield.cuh" // 必须包含波场头文件
#include <iostream>
#include <cmath>
#include <tuple>
#include <iomanip>

int main()
{
    std::cout << "--- Froude-Krylov 力计算器初始化 ---\n";

    // 1. 定义波场和网格参数
    int Nx = 64, Ny = 64;   // FFT 网格点数
    float Lx_grid = 100.0f; // 波场物理域 X 长度 (m)
    float Ly_grid = 100.0f; // 波场物理域 Y 长度 (m)

    // 2. 定义 JONSWAP 波谱参数
    float Hs = 0.5f;         // 有义波高 (m)
    float Tp = 3.5f;         // 峰值周期 (s)
    float gamma = 3.3f;      // 峰度因子
    unsigned long seed = 42; // 随机数种子

    // 3. 实例化 WaveFieldCalculator 和 waveforce_generator
    try
    {
        // 3a. 实例化波场计算器 (生成 a_k, 频率, 波数)
        std::cout << "正在初始化 WaveFieldCalculator...\n";
        wavefield::WaveFieldCalculator wave_calc(
            Nx, Ny, Lx_grid, Ly_grid,
            Hs, Tp, gamma, seed);
        std::cout << "WaveFieldCalculator 初始化成功。\n";

        // 3b. 实例化波浪力计算器 (传入波场计算器的指针)
        std::cout << "正在初始化 WaveForceGenerator...\n";
        waveforce::waveforce_generator generator(&wave_calc);
        std::cout << "WaveForceGenerator 初始化成功，船体微元生成完毕。\n";

        std::cout << "\n--- 船舶运动和波浪力仿真循环 ---\n";

        // 4. 仿真参数
        float simulation_time = 20.0f;
        float dt = 0.5f;

        // 5. 模拟船舶运动状态 (Global Frame)
        float x_ship = 0.0f;
        float y_ship = 0.0f;
        float psi = 0.0f; // 初始首摇角 (0 弧度)

        std::cout << std::fixed << std::setprecision(4);
        std::cout << " t (s) | Ship X (m) | Ship Y (m) | Psi (rad) | Fu (N) | Fv (N) | Mr (N.m)\n";
        std::cout << "---------------------------------------------------------------------------------\n";

        for (float t = 0.0f; t <= simulation_time; t += dt)
        {

            // 模拟简单的螺旋运动轨迹
            x_ship = 5.0f * std::cos(t / 10.0f);
            y_ship = 5.0f * std::sin(t / 10.0f);
            psi = 0.1f * t; // 简单的持续首摇

            // 6. 调用查询接口
            // 注意: auto [Fu, Fv, Mr] 需要 C++17 支持
            auto [Fu, Fv, Mr] = generator.query_fk_forces(t, x_ship, y_ship, psi);

            // 7. 输出结果
            std::cout << std::setw(7) << t << " | "
                      << std::setw(10) << x_ship << " | "
                      << std::setw(10) << y_ship << " | "
                      << std::setw(9) << psi << " | "
                      << std::setw(6) << Fu << " | "
                      << std::setw(6) << Fv << " | "
                      << std::setw(8) << Mr << "\n";
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "发生错误: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "发生未知错误。" << std::endl;
        return 1;
    }
    return 0;
}
