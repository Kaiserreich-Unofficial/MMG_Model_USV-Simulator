#include "wavefield.cuh"

// =========================================================================
// CUDA 辅助函数和 Kernel 定义
// =========================================================================

// ------------------- JONSWAP 谱 (Device & Host) -------------------
__host__ __device__ inline float jonswap_spectrum(float omega, float Hs, float Tp, float gamma)
{
    omega = omega < 1e-6f ? 1e-6f : omega;
    float omega_p = 2.0f * M_PI / Tp;
    float alpha = 0.076f * powf(Hs * Hs / powf(Tp, 4.0f), 0.22f);
    float sigma = omega <= omega_p ? 0.07f : 0.09f;
    float r = expf(-powf(omega - omega_p, 2.0f) / (2.0f * sigma * sigma * omega_p * omega_p));
    float S = alpha * g * g * powf(omega, -5.0f) * expf(-1.25f * powf(omega_p / omega, 4.0f)) * powf(gamma, r);
    return S;
}

// ------------------- CUDA 随机相位初始化 Kernel -------------------
__global__ void init_random_phase(curandState *states, int Ntot, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Ntot)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ------------------- 波谱生成 Kernel -------------------
__global__ void compute_ak(cufftComplex *a_k, float *omega, curandState *states, float dkx, float dky,
                           int Nx, int Ny, float Hs, float Tp, float gamma)
{
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int Nx_half = Nx / 2 + 1;
    if (i < Nx_half && j < Ny)
    {
        int idx = j * Nx_half + i;
        float S = jonswap_spectrum(omega[idx], Hs, Tp, gamma);
        float amp = 2.8 * sqrtf(S * dkx * dky);
        float phi = curand_uniform(&states[idx]) * 2.0f * M_PI;
        a_k[idx].x = amp * cosf(phi);
        a_k[idx].y = amp * sinf(phi);
    }
}

// ------------------- 单点波高查询 Kernel -------------------
__global__ void query_wave_point_kernel(
    const cufftComplex *a_k, const float *omega,
    int Nx, int Ny, float Lx, float Ly,
    float x, float y, float t,
    float *out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Ntot = Ny * (Nx / 2 + 1);
    if (idx >= Ntot)
        return;

    int j = idx / (Nx / 2 + 1);
    int i = idx % (Nx / 2 + 1);

    float kx = i * 2.0f * M_PI / Lx;
    float ky = (j <= Ny / 2) ? j * 2.0f * M_PI / Ly : (j - Ny) * 2.0f * M_PI / Ly;

    cufftComplex val = a_k[idx];
    float phase = kx * x + ky * y - omega[idx] * t;
    float contrib = val.x * cosf(phase) - val.y * sinf(phase);

    atomicAdd(out, contrib); // 所有线程累加到一个输出
}

// ------------------- 单点速度势查询 Kernel -------------------
__global__ void query_potential_point_kernel(
    const cufftComplex *a_k, const float *omega,
    int Nx, int Ny, float Lx, float Ly,
    float x, float y, float t,
    float *out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int Ntot = Ny * (Nx / 2 + 1);
    if (idx >= Ntot)
        return;

    int j = idx / (Nx / 2 + 1);
    int i = idx % (Nx / 2 + 1);

    float kx = i * 2.0f * M_PI / Lx;
    float ky = (j <= Ny / 2) ? j * 2.0f * M_PI / Ly : (j - Ny) * 2.0f * M_PI / Ly;

    cufftComplex val = a_k[idx];
    float phase = kx * x + ky * y - omega[idx] * t;
    float contrib = (val.x / omega[idx]) * cosf(phase) - (val.y / omega[idx]) * sinf(phase);

    atomicAdd(out, contrib);
}

namespace wavefield
{
    // 构造函数实现
    WaveFieldCalculator::WaveFieldCalculator(int nx, int ny, float lx, float ly, float hs, float tp, float g_factor, unsigned long seed)
        : Nx(nx), Ny(ny), Lx_grid(lx), Ly_grid(ly), Hs(hs), Tp(tp), gamma(g_factor), seed(seed)
    {
        // 1. 计算常量
        Ntot_complex = Ny * (Nx / 2 + 1);
        Ntot_real = Nx * Ny;
        dkx = 2.0f * M_PI / Lx_grid;
        dky = 2.0f * M_PI / Ly_grid;

        // 2. 分配 Device 内存 (使用更安全的 CUDA_CHECK)
        if (cudaMalloc(&d_a_k, sizeof(cufftComplex) * Ntot_complex) != cudaSuccess)
            throw std::runtime_error("Failed to allocate d_a_k memory.");
        if (cudaMalloc(&d_states, sizeof(curandState) * Ntot_complex) != cudaSuccess)
            throw std::runtime_error("Failed to allocate d_states memory.");
        if (cudaMalloc(&d_omega, sizeof(float) * Ntot_complex) != cudaSuccess)
            throw std::runtime_error("Failed to allocate d_omega memory.");

        // 3. 计算 Host 频率和波数数组
        h_omega.resize(Ntot_complex);
        h_k_abs.resize(Ntot_complex);
        h_a_k.resize(Ntot_complex); // 确保 Host 存储空间
        for (int j = 0; j < Ny; j++)
        {
            float ky = (j <= Ny / 2) ? j * dky : (j - Ny) * dky;
            for (int i = 0; i < Nx / 2 + 1; i++)
            {
                float kx = i * dkx;
                float k = std::sqrt(kx * kx + ky * ky);
                float omega = std::sqrt(g * k);
                int idx = j * (Nx / 2 + 1) + i;
                h_omega[idx] = omega < 1e-6f ? 1e-6f : omega;
                h_k_abs[idx] = k;
            }
        }

        // 4. 传输 Host 频率到 Device
        if (cudaMemcpy(d_omega, h_omega.data(), sizeof(float) * Ntot_complex, cudaMemcpyHostToDevice) != cudaSuccess)
            throw std::runtime_error("Failed to copy d_omega to device.");

        // 5. 启动 CUDA Kernel 生成 a_k
        int blocks = (Ntot_complex + 256 - 1) / 256;
        init_random_phase<<<blocks, 256>>>(d_states, Ntot_complex, seed);
        dim3 blockDim(16, 16);
        dim3 gridDim((Nx / 2 + 1 + 15) / 16, (Ny + 15) / 16);
        compute_ak<<<gridDim, blockDim>>>(
            d_a_k, d_omega, d_states, dkx, dky, Nx, Ny, Hs, Tp, gamma);

        if (cudaGetLastError() != cudaSuccess)
            throw std::runtime_error("CUDA kernel launch failed in constructor.");
        cudaDeviceSynchronize(); // 确保 Kernel 完成

        // 6. 关键修正：将生成的 a_k 从 Device 复制到 Host
        if (cudaMemcpy(h_a_k.data(), d_a_k, sizeof(cufftComplex) * Ntot_complex, cudaMemcpyDeviceToHost) != cudaSuccess)
            throw std::runtime_error("Failed to copy d_a_k to h_a_k in constructor.");
    }

    // 析构函数实现
    WaveFieldCalculator::~WaveFieldCalculator()
    {
        if (d_a_k)
            cudaFree(d_a_k);
        if (d_states)
            cudaFree(d_states);
        if (d_omega)
            cudaFree(d_omega);
    }

    // 全场波高查询接口实现 (与之前版本相同)
    void WaveFieldCalculator::query_wave_height(float t, float *h_out)
    {
        std::vector<cufftComplex> h_a_k(Ntot_complex);
        if (cudaMemcpy(h_a_k.data(), d_a_k, sizeof(cufftComplex) * Ntot_complex, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("CUDA Memcpy (d_a_k to h_a_k) failed in query.");
        }

        std::vector<cufftComplex> a_k_t(Ntot_complex);
        for (int j = 0; j < Ny; j++)
        {
            for (int i = 0; i < Nx / 2 + 1; i++)
            {
                int idx = j * (Nx / 2 + 1) + i;
                float omega = h_omega[idx];
                float coswt = cosf(-omega * t);
                float sinwt = sinf(-omega * t);
                cufftComplex val = h_a_k[idx];

                a_k_t[idx].x = val.x * coswt - val.y * sinwt;
                a_k_t[idx].y = val.x * sinwt + val.y * coswt;
            }
        }

        cufftHandle plan;
        cufftPlan2d(&plan, Ny, Nx, CUFFT_C2R);
        cufftComplex *d_a_k_temp = nullptr;
        float *d_h = nullptr;

        if (cudaMalloc(&d_a_k_temp, sizeof(cufftComplex) * Ntot_complex) != cudaSuccess ||
            cudaMalloc(&d_h, sizeof(float) * Ntot_real) != cudaSuccess)
        {
            if (d_a_k_temp)
                cudaFree(d_a_k_temp);
            if (d_h)
                cudaFree(d_h);
            cufftDestroy(plan);
            throw std::runtime_error("Failed to allocate temporary device memory for FFT.");
        }

        cudaMemcpy(d_a_k_temp, a_k_t.data(), sizeof(cufftComplex) * Ntot_complex, cudaMemcpyHostToDevice);

        cufftExecC2R(plan, d_a_k_temp, d_h);
        cudaMemcpy(h_out, d_h, sizeof(float) * Ntot_real, cudaMemcpyDeviceToHost);

        cudaFree(d_a_k_temp);
        cudaFree(d_h);
        cufftDestroy(plan);
    }

    // 全场速度势查询接口实现 (与之前版本相同)
    void WaveFieldCalculator::query_velocity_potential(float t, float *phi_out)
    {
        std::vector<cufftComplex> h_a_k(Ntot_complex);
        if (cudaMemcpy(h_a_k.data(), d_a_k, sizeof(cufftComplex) * Ntot_complex, cudaMemcpyDeviceToHost) != cudaSuccess)
        {
            throw std::runtime_error("CUDA Memcpy (d_a_k to h_a_k) failed in query.");
        }

        std::vector<cufftComplex> a_k_t(Ntot_complex);
        for (int j = 0; j < Ny; j++)
        {
            for (int i = 0; i < Nx / 2 + 1; i++)
            {
                int idx = j * (Nx / 2 + 1) + i;
                float omega = h_omega[idx];
                float coswt = cosf(-omega * t);
                float sinwt = sinf(-omega * t);
                cufftComplex val = h_a_k[idx];

                // 速度势 φ ~ a_k/ω
                a_k_t[idx].x = (val.x / omega) * coswt - (val.y / omega) * sinwt;
                a_k_t[idx].y = (val.x / omega) * sinwt + (val.y / omega) * coswt;
            }
        }

        cufftHandle plan;
        cufftPlan2d(&plan, Ny, Nx, CUFFT_C2R);
        cufftComplex *d_a_k_temp = nullptr;
        float *d_phi = nullptr;

        if (cudaMalloc(&d_a_k_temp, sizeof(cufftComplex) * Ntot_complex) != cudaSuccess ||
            cudaMalloc(&d_phi, sizeof(float) * Ntot_real) != cudaSuccess)
        {
            if (d_a_k_temp)
                cudaFree(d_a_k_temp);
            if (d_phi)
                cudaFree(d_phi);
            cufftDestroy(plan);
            throw std::runtime_error("Failed to allocate temporary device memory for FFT.");
        }

        cudaMemcpy(d_a_k_temp, a_k_t.data(), sizeof(cufftComplex) * Ntot_complex, cudaMemcpyHostToDevice);

        cufftExecC2R(plan, d_a_k_temp, d_phi);
        cudaMemcpy(phi_out, d_phi, sizeof(float) * Ntot_real, cudaMemcpyDeviceToHost);

        cudaFree(d_a_k_temp);
        cudaFree(d_phi);
        cufftDestroy(plan);
    }

    // ------------------- 辅助函数实现 -------------------

    size_t WaveFieldCalculator::get_grid_index(float x, float y) const
    {
        // 计算每个网格的物理尺寸
        float dx = Lx_grid / Nx;
        float dy = Ly_grid / Ny;

        // 处理循环边界条件 (将 x, y 映射到 [0, L) 范围内)
        x = fmod(x, Lx_grid);
        y = fmod(y, Ly_grid);
        if (x < 0)
            x += Lx_grid;
        if (y < 0)
            y += Ly_grid;

        // 计算索引 (m, n) - 四舍五入到最近的网格点
        int m = static_cast<int>(roundf(x / dx));
        int n = static_cast<int>(roundf(y / dy));

        // 边界保护: 处理 L_x 和 L_y 边界
        m = (m == Nx) ? 0 : m; // L_x 边界映射回 0
        n = (n == Ny) ? 0 : n; // L_y 边界映射回 0

        // 扁平化索引: index = n * Nx + m (Ny 是外层循环)
        return (size_t)n * Nx + (size_t)m;
    }

    // ------------------- 单点波高查询实现 -------------------
    float WaveFieldCalculator::query_wave_height_point(float x, float y, float t)
    {
        float h_point = 0.0f;
        float *d_out = nullptr;
        cudaMalloc(&d_out, sizeof(float));
        cudaMemset(d_out, 0, sizeof(float));

        int threads = 256;
        int blocks = (Ntot_complex + threads - 1) / threads;
        query_wave_point_kernel<<<blocks, threads>>>(
            d_a_k, d_omega, Nx, Ny, Lx_grid, Ly_grid, x, y, t, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_point, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);

        return h_point;
    }
    // ------------------- 单点速度势查询实现 -------------------
    float WaveFieldCalculator::query_velocity_potential_point(float x, float y, float t)
    {
        float phi_point = 0.0f;
        float *d_out = nullptr;
        cudaMalloc(&d_out, sizeof(float));
        cudaMemset(d_out, 0, sizeof(float));

        int threads = 256;
        int blocks = (Ntot_complex + threads - 1) / threads;
        query_potential_point_kernel<<<blocks, threads>>>(
            d_a_k, d_omega, Nx, Ny, Lx_grid, Ly_grid, x, y, t, d_out);
        cudaDeviceSynchronize();

        cudaMemcpy(&phi_point, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out);

        return phi_point;
    }
}
