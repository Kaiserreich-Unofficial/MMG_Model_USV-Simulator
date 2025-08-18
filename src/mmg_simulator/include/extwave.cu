#include "extwave.cuh"
#include <ros/ros.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand_kernel.h>
#include <cmath>
#include <stdexcept>
#include <omp.h>

constexpr float g = 9.81f;    // 重力加速度 m/s^2
constexpr float rho = 997.0f; // 海水密度 kg/m^3
__device__ __host__ float jonswap(float omega, float alpha, float omega_p, float gamma)
{
    if (omega <= 0.f)
        return 0.f;
    float sigma = (omega <= omega_p) ? 0.07f : 0.09f;
    float r = expf(-powf((omega - omega_p), 2.f) / (2.f * sigma * sigma * omega_p * omega_p));
    float S = alpha * g * g / powf(omega, 5.f) * expf(-1.25f * powf(omega_p / omega, 4.f));
    return S * powf(gamma, r);
}

// Generate spectrum Core
__global__ void generateSpectrumKernel(
    thrust::complex<float> *spectrum_fx,
    thrust::complex<float> *spectrum_fy,
    thrust::complex<float> *spectrum_mz,
    int N, float df, float alpha, float omega_p, float gamma, float wave_dir, unsigned long long seed)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (k >= N / 2)
        return;

    curandState state;
    curand_init(seed, k, 0, &state);
    float phi = curand_uniform(&state) * 2.f * M_PI;

    float f = k * df;
    float omega = 2.f * M_PI * f;
    float S = jonswap(omega, alpha, omega_p, gamma);
    float A = sqrtf(2.f * S * df);

    thrust::complex<float> spec = A * thrust::polar(1.f, phi);

    spectrum_fx[k] = spec * cosf(wave_dir);
    spectrum_fy[k] = spec * sinf(wave_dir);
    spectrum_mz[k] = spec * sinf(2.f * wave_dir);

    // 共轭对称赋值
    spectrum_fx[N - k] = thrust::conj(spectrum_fx[k]);
    spectrum_fy[N - k] = thrust::conj(spectrum_fy[k]);
    spectrum_mz[N - k] = thrust::conj(spectrum_mz[k]);
}

CudaWaveForceGenerator::CudaWaveForceGenerator(
    int N_, float dt_, float Hs, float Tp, float waveDirectionRad,
    float L, float B // NEW: 船长、船宽
    ) : N(N_), dt(dt_), wave_dir(waveDirectionRad)
{
    T = N * dt;

    // 计算谱参数
    float omega_p = 2.f * M_PI / Tp;
    float alpha = 5.f / 16.f * (Hs * Hs * powf(omega_p, 4)) / (g * g);
    float df = 1.f / T;
    float gamma = 3.3f;

    // 保存物理参数
    this->Hs = Hs;
    this->L = L;
    this->B = B;

    allocateCudaMemory();

    // 启动CUDA核函数生成频谱
    int threadsPerBlock = 256;
    int blocks = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;

    generateSpectrumKernel<<<blocks, threadsPerBlock>>>(
        d_spectrum_fx, d_spectrum_fy, d_spectrum_mz,
        N, df, alpha, omega_p, gamma, wave_dir, 1234ULL);

    cudaDeviceSynchronize();

    runIFFT(); // 内部会做 RMS 缩放
}

CudaWaveForceGenerator::~CudaWaveForceGenerator()
{
    freeCudaMemory();
}

void CudaWaveForceGenerator::allocateCudaMemory()
{
    cudaMalloc(&d_spectrum_fx, sizeof(thrust::complex<float>) * N);
    cudaMalloc(&d_spectrum_fy, sizeof(thrust::complex<float>) * N);
    cudaMalloc(&d_spectrum_mz, sizeof(thrust::complex<float>) * N);
    cudaMalloc(&d_time_series, sizeof(float) * N * 3);
}

void CudaWaveForceGenerator::freeCudaMemory()
{
    if (d_spectrum_fx)
        cudaFree(d_spectrum_fx);
    if (d_spectrum_fy)
        cudaFree(d_spectrum_fy);
    if (d_spectrum_mz)
        cudaFree(d_spectrum_mz);
    if (d_time_series)
        cudaFree(d_time_series);
}

void CudaWaveForceGenerator::runIFFT()
{
    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, 1);

    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex *>(d_spectrum_fx),
                 reinterpret_cast<cufftComplex *>(d_spectrum_fx),
                 CUFFT_INVERSE);

    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex *>(d_spectrum_fy),
                 reinterpret_cast<cufftComplex *>(d_spectrum_fy),
                 CUFFT_INVERSE);

    cufftExecC2C(plan,
                 reinterpret_cast<cufftComplex *>(d_spectrum_mz),
                 reinterpret_cast<cufftComplex *>(d_spectrum_mz),
                 CUFFT_INVERSE);

    std::vector<float> h_time_series(N * 3);
    std::vector<thrust::complex<float>> h_fx(N), h_fy(N), h_mz(N);

    cudaMemcpy(h_fx.data(), d_spectrum_fx, sizeof(thrust::complex<float>) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_fy.data(), d_spectrum_fy, sizeof(thrust::complex<float>) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mz.data(), d_spectrum_mz, sizeof(thrust::complex<float>) * N, cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        // IFFT 后结果需要除以 N 做归一化（cufft 不自动除）
        h_time_series[3 * i + 0] = h_fx[i].real() / static_cast<float>(N);
        h_time_series[3 * i + 1] = h_fy[i].real() / static_cast<float>(N);
        h_time_series[3 * i + 2] = h_mz[i].real() / static_cast<float>(N);
    }

    // ===== RMS 缩放（更稳健的实现） =====
    // 水线面面积
    float Awp = this->L * this->B; // m^2
    // 目标力 RMS（Fx/Fy）
    float target_rms_force = 0.5f * rho * g * Awp * this->Hs;

    // 经验杠杆臂用于将力转换成力矩目标（可调）
    float lever_arm = this->L / 4.0f; // 例如 L/4，视船型调整
    float target_rms_mz = target_rms_force * lever_arm;

    // 计算当前 RMS：合力（Fx^2 + Fy^2）
    double sum_sq_force = 0.0;
    double sum_sq_mz = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double fx = static_cast<double>(h_time_series[3 * i + 0]);
        double fy = static_cast<double>(h_time_series[3 * i + 1]);
        double mz = static_cast<double>(h_time_series[3 * i + 2]);
        sum_sq_force += (fx * fx + fy * fy); // 合力平方和
        sum_sq_mz += (mz * mz);
    }
    double current_rms_force = sqrt(sum_sq_force / static_cast<double>(N));
    double current_rms_mz = sqrt(sum_sq_mz / static_cast<double>(N));

    // 防止除零与极端放大：设置上下界
    const double EPS = 1e-12;
    const double MAX_SCALE = 1e3;  // 上限（根据需要调）
    const double MIN_SCALE = 1e-3; // 下限（避免被缩为0）

    double scale_force = 1.0;
    if (current_rms_force > EPS)
        scale_force = static_cast<double>(target_rms_force) / current_rms_force;
    // 限制范围并保证为有限值
    if (!isfinite(scale_force) || isnan(scale_force))
        ROS_WARN("scale_force is not finite or nan: %f", scale_force);
        scale_force = 1.0;
    scale_force = fmax(scale_force, MIN_SCALE);
    scale_force = fmin(scale_force, MAX_SCALE);

    double scale_mz = 1.0;
    if (current_rms_mz > EPS)
        scale_mz = static_cast<double>(target_rms_mz) / current_rms_mz;
    if (!isfinite(scale_mz) || isnan(scale_mz))
        ROS_WARN("scale_mz is not finite or nan: %f", scale_mz);
        scale_mz = 1.0;
    scale_mz = fmax(scale_mz, MIN_SCALE);
    scale_mz = fmin(scale_mz, MAX_SCALE);

    // 如果两个 scale 都非常接近 1，则跳过乘法以节省开销
    bool doScale = (fabs(scale_force - 1.0) > 1e-6) || (fabs(scale_mz - 1.0) > 1e-6);

    if (doScale)
    {
#pragma omp parallel for
        for (int i = 0; i < N; ++i)
        {
            // 先把 Fx/Fy 按合力 scale 放大
            float fx = h_time_series[3 * i + 0];
            float fy = h_time_series[3 * i + 1];
            float mz = h_time_series[3 * i + 2];

            double new_fx = fx * scale_force;
            double new_fy = fy * scale_force;
            double new_mz = mz * scale_mz;

            // 若某个值不是有限的（极端情况），退回到安全值 0
            if (!isfinite(new_fx)) new_fx = 0.0;
            if (!isfinite(new_fy)) new_fy = 0.0;
            if (!isfinite(new_mz)) new_mz = 0.0;

            h_time_series[3 * i + 0] = static_cast<float>(new_fx);
            h_time_series[3 * i + 1] = static_cast<float>(new_fy);
            h_time_series[3 * i + 2] = static_cast<float>(new_mz);
        }
    }
    cudaMemcpy(d_time_series, h_time_series.data(), sizeof(float) * N * 3, cudaMemcpyHostToDevice);

    cufftDestroy(plan);
}


Eigen::Vector3f CudaWaveForceGenerator::getWaveForceGlobal(float t)
{
    int idx = static_cast<int>(t / dt) % N;
    float h_force[3];
    cudaMemcpy(h_force, d_time_series + 3 * idx, sizeof(float) * 3, cudaMemcpyDeviceToHost);
    return Eigen::Vector3f(h_force[0], h_force[1], h_force[2]);
}

__host__ Eigen::Vector3f CudaWaveForceGenerator::getWaveForce(const Eigen::Matrix<float, 6, 1> &state, float t)
{
    float psi = state(2);
    float cpsi = cosf(psi);
    float spsi = sinf(psi);

    Eigen::Vector3f global_force = getWaveForceGlobal(t);
    Eigen::Matrix2f R;
    R << cpsi, spsi,
        -spsi, cpsi;

    Eigen::Vector2f Fg(global_force(0), global_force(1));
    Eigen::Vector2f Fb = R.transpose() * Fg;

    Eigen::Vector3f tau;
    tau << Fb(0), Fb(1), global_force(2);
    return tau;
}
