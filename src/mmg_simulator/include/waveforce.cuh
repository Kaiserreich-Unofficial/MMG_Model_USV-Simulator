#ifndef WAVEFORCE_CUH
#define WAVEFORCE_CUH

#include <vector>
#include <tuple>
#include <cmath>
#include <stdexcept>

#include "wavefield.cuh" // 引入波场计算器

namespace waveforce
{
    // 常量定义 (使用 wavefield 中的 RHO 和 g)
    constexpr float g = 9.81f; // 重力加速度 (m/s²)
    constexpr float RHO = 1025.0f; // 海水密度 (kg/m³)

    // 船舶表面微元结构体
    struct SurfaceElement
    {
        float x0;       // 随体坐标系 x
        float y0;       // 随体坐标系 y
        float z0;       // 随体坐标系 z
        float nx0;      // 法向量 x 分量
        float ny0;      // 法向量 y 分量
        float nz0;      // 法向量 z 分量
        float area;     // 微元面积 dA
    };

    // 假设的船舶参数常量
    constexpr float SHIP_L = 100.0f;
    constexpr float SHIP_B = 20.0f;
    constexpr float SHIP_T0 = 8.0f;
    constexpr float DELTA_L = 1.0f;
    constexpr float DELTA_S = DELTA_L * DELTA_L;
    constexpr float FREEBOARD = 2.0f; // 干舷高度 (用于生成船体微元边界)

    class waveforce_generator
    {
    public:
        // 构造函数：需要波场计算器的指针或引用
        waveforce_generator(wavefield::WaveFieldCalculator* wave_calc);

        // 核心方法: 计算 FK 波浪力 (Fu, Fv, Mr)
        std::tuple<float, float, float> query_fk_forces(float t, float x, float y, float psi);

    private:
        wavefield::WaveFieldCalculator* wave_calculator_; // 波场计算器指针
        std::vector<SurfaceElement> elements;           // 船体表面微元

        // 1. 生成船体微元 (只在构造函数中调用一次)
        std::vector<SurfaceElement> generate_ship_elements();

        // 2. 核心：计算指定点处的瞬时波面高 $\eta$ (使用波场数据进行高效求和)
        float query_wave_height_point_summation(float x, float y, float t) const;

        // 3. 核心：计算指定点处的动态压力 $P_{\text{dyn}}$ (包含深度衰减 $e^{kz}$)
        float calculate_dynamic_pressure_point(float x, float y, float z, float t) const;
    };
}

#endif // WAVEFORCE_CUH
