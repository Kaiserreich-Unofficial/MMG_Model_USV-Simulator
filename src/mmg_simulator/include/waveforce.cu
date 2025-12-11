#include "waveforce.cuh"
#include <iostream>
#include <cmath>

namespace waveforce
{
    // ------------------------------------------------------------------------
    // C. 构造函数和船体生成
    // ------------------------------------------------------------------------
    waveforce_generator::waveforce_generator(wavefield::WaveFieldCalculator* wave_calc)
        : wave_calculator_(wave_calc)
    {
        if (!wave_calc)
        {
            throw std::invalid_argument("WaveFieldCalculator pointer cannot be null.");
        }
        elements = generate_ship_elements();
    }

    std::vector<SurfaceElement> waveforce_generator::generate_ship_elements()
    {
        std::vector<SurfaceElement> elems;

        // 1. 船底 (Z = -SHIP_T0, nz = -1.0)
        for (float x = -SHIP_L / 2.0f + DELTA_L / 2.0f; x < SHIP_L / 2.0f; x += DELTA_L)
        {
            for (float y = -SHIP_B / 2.0f + DELTA_L / 2.0f; y < SHIP_B / 2.0f; y += DELTA_L)
            {
                elems.push_back({x, y, -SHIP_T0, 0.0f, 0.0f, -1.0f, DELTA_S});
            }
        }

        // 2. 侧壁 (Y = +/- B/2) - 范围扩展到 FREEBOARD
        for (float x = -SHIP_L / 2.0f + DELTA_L / 2.0f; x < SHIP_L / 2.0f; x += DELTA_L)
        {
            for (float z = -SHIP_T0 + DELTA_L / 2.0f; z < FREEBOARD; z += DELTA_L)
            {
                elems.push_back({x, SHIP_B / 2.0f, z, 0.0f, 1.0f, 0.0f, DELTA_S});  // 右舷
                elems.push_back({x, -SHIP_B / 2.0f, z, 0.0f, -1.0f, 0.0f, DELTA_S}); // 左舷
            }
        }

        // 3. 首尾垂直平面 (X = +/- L/2) - 范围扩展到 FREEBOARD
        for (float y = -SHIP_B / 2.0f + DELTA_L / 2.0f; y < SHIP_B / 2.0f; y += DELTA_L)
        {
            for (float z = -SHIP_T0 + DELTA_L / 2.0f; z < FREEBOARD; z += DELTA_L)
            {
                elems.push_back({SHIP_L / 2.0f, y, z, 1.0f, 0.0f, 0.0f, DELTA_S});  // 船头
                elems.push_back({-SHIP_L / 2.0f, y, z, -1.0f, 0.0f, 0.0f, DELTA_S}); // 船尾
            }
        }

        // std::cout << "船体微元生成完毕，总数: " << elems.size() << " 个." << std::endl;
        return elems;
    }


    // ------------------------------------------------------------------------
    // D. 核心求和函数
    // ------------------------------------------------------------------------

    // D.1 计算指定点处的瞬时波面高 $\eta$ (高效求和版)
    float waveforce_generator::query_wave_height_point_summation(float x, float y, float t) const
    {
        float eta_sum = 0.0f;

        // 通过 WaveFieldCalculator 访问 Host 端数据
        const auto& h_omega = wave_calculator_->get_h_omega();
        const auto& h_a_k = wave_calculator_->get_h_a_k();
        int Nx = wave_calculator_->get_Nx();
        int Ny = wave_calculator_->get_Ny();
        float dkx = wave_calculator_->get_dkx();
        float dky = wave_calculator_->get_dky();
        int Ntot_complex = h_omega.size();

        for (int j = 0; j < Ny; j++)
        {
            float ky = (j <= Ny / 2) ? j * dky : (j - Ny) * dky;
            for (int i = 0; i < Nx / 2 + 1; i++)
            {
                int idx = j * (Nx / 2 + 1) + i;
                float kx = i * dkx;
                float omega = h_omega[idx];
                cufftComplex a_k = h_a_k[idx];

                float phase_arg = (kx * x + ky * y) - omega * t;

                // Re[ a_k * e^{-i \omega t} ]
                float temp_real = a_k.x * std::cos(phase_arg) - a_k.y * std::sin(phase_arg);

                // C2R 半谱求和还原因子 (k=0 边界项不加倍)
                float scale = (i == 0 && j == 0) ? 1.0f : 2.0f;

                eta_sum += scale * temp_real;
            }
        }
        return eta_sum;
    }

    // D.2 计算指定点处的动态压力 $P_{\text{dyn}}$ (包含深度衰减 $e^{kz}$)
    float waveforce_generator::calculate_dynamic_pressure_point(float x, float y, float z, float t) const
    {
        float P_dyn_sum = 0.0f;

        // 通过 WaveFieldCalculator 访问 Host 端数据
        const auto& h_omega = wave_calculator_->get_h_omega();
        const auto& h_k_abs = wave_calculator_->get_h_k_abs();
        const auto& h_a_k = wave_calculator_->get_h_a_k();
        int Nx = wave_calculator_->get_Nx();
        int Ny = wave_calculator_->get_Ny();
        float dkx = wave_calculator_->get_dkx();
        float dky = wave_calculator_->get_dky();

        for (int j = 0; j < Ny; j++)
        {
            float ky = (j <= Ny / 2) ? j * dky : (j - Ny) * dky;
            for (int i = 0; i < Nx / 2 + 1; i++)
            {
                int idx = j * (Nx / 2 + 1) + i;
                float kx = i * dkx;

                float omega = h_omega[idx];
                float k = h_k_abs[idx];
                cufftComplex a_k = h_a_k[idx];

                // 1. 深度衰减项 e^{k z} (z < 0)
                float depth_attenuation = std::exp(k * z);

                // 2. 总相位
                float phase_arg = (kx * x + ky * y) - omega * t;

                // 3. 计算 a_k * e^{i\theta} 的实部
                float temp_real = a_k.x * std::cos(phase_arg) - a_k.y * std::sin(phase_arg);

                // 4. C2R 半谱求和还原因子
                float scale = (i == 0 && j == 0) ? 1.0f : 2.0f;

                // P_dyn 贡献 = $\rho g \cdot e^{kz} \cdot \text{scale} \cdot \operatorname{Re}[a_k e^{i\theta}]$
                P_dyn_sum += RHO * g * depth_attenuation * scale * temp_real;
            }
        }

        return P_dyn_sum;
    }


    // ------------------------------------------------------------------------
    // E. 核心方法: query_fk_forces (动态湿表面积分)
    // ------------------------------------------------------------------------
    std::tuple<float, float, float> waveforce_generator::query_fk_forces(float t, float x, float y, float psi)
    {
        float Fu = 0.0f;
        float Fv = 0.0f;
        float Mr = 0.0f;

        float cos_psi = std::cos(psi);
        float sin_psi = std::sin(psi);

        // 遍历所有船体微元 (包括可能出水的)
        for (const auto &element : elements)
        {
            // A. 坐标和法向量变换 (Body -> Global)
            float x_prime0 = element.x0;
            float y_prime0 = element.y0;
            float z_prime0 = element.z0;

            // 1. 变换面元位置到世界坐标系
            float x_global = x + x_prime0 * cos_psi - y_prime0 * sin_psi;
            float y_global = y + x_prime0 * sin_psi + y_prime0 * cos_psi;
            float z_global = z_prime0; // 假设船舶运动仅为 3-DOF (x, y, psi)

            // 2. 计算该点处的瞬时波面高 $\eta$
            float eta_global = query_wave_height_point_summation(x_global, y_global, t);

            // 3. **瞬时湿表面判断**：如果微元位置 z_global 高于瞬时波面 $\eta$，则不浸没。
            if (z_global > eta_global)
            {
                continue;
            }

            // 4. 变换法向量到世界坐标系
            float nx_global = element.nx0 * cos_psi - element.ny0 * sin_psi;
            float ny_global = element.nx0 * sin_psi + element.ny0 * cos_psi;

            // 5. 计算动态压力 P_dynamic
            float p_dynamic = calculate_dynamic_pressure_point(x_global, y_global, z_global, t);

            // C. 世界坐标系下的微小力 dF_Global = p * n_Global * dA
            float dFx_global = p_dynamic * nx_global * element.area;
            float dFy_global = p_dynamic * ny_global * element.area;

            // D. 转换力到随体坐标系 (Global -> Body)
            float dFu = dFx_global * cos_psi + dFy_global * sin_psi;
            float dFv = -dFx_global * sin_psi + dFy_global * cos_psi;

            // E. 累加力和计算首摇力矩 Mr = x' * dFv - y' * dFu
            Fu += dFu;
            Fv += dFv;
            Mr += (x_prime0 * dFv - y_prime0 * dFu);
        }

        return {Fu, Fv, Mr};
    }
}
