#include "extwave.cuh"
#include <cufft.h>
#include <cmath>
#include <vector>
#include <random>
#include <ros/ros.h>

constexpr float g = 9.81f;
constexpr float rho = 997.0f;

// ------------------------ JONSWAP 波谱 ------------------------
static float jonswap(float omega, float alpha, float omega_p, float gamma)
{
    if(omega <= 0.f) return 0.f;
    float sigma = (omega <= omega_p) ? 0.07f : 0.09f;
    float r = expf(-powf((omega - omega_p),2)/(2*sigma*sigma*omega_p*omega_p));
    float S = alpha * g*g / powf(omega,5.f) * expf(-1.25f * powf(omega_p/omega,4.f));
    return S * powf(gamma, r);
}

// ------------------------ RAO 插值 ------------------------
static float interpRAO(const std::vector<float> &freqs, const std::vector<float> &RAO, float omega)
{
    if(omega <= freqs.front()) return RAO.front();
    if(omega >= freqs.back()) return RAO.back();
    auto it = std::upper_bound(freqs.begin(), freqs.end(), omega);
    size_t idx = it - freqs.begin();
    float w1 = freqs[idx-1], w2 = freqs[idx];
    float r1 = RAO[idx-1], r2 = RAO[idx];
    float t = (omega - w1)/(w2 - w1);
    return r1*(1-t) + r2*t;
}

// ------------------------ 构造函数 ------------------------
CudaWaveForceGenerator::CudaWaveForceGenerator(
    int N_, float dt_, float Hs_, float Tp_, float waveDirRad,
    float L_, float B_, float draft_, float waterDepth_
): N(N_), dt(dt_), Hs(Hs_), Tp(Tp_), wave_dir(waveDirRad),
   L(L_), B(B_), draft(draft_), water_depth(waterDepth_)
{
    T = N*dt;

    // RAO 简单经验公式（小长方体, surge/sway）
    freq_vals.resize(10);
    RAO_surge.resize(10);
    RAO_sway.resize(10);
    for(int i=0;i<10;i++){
        float f = 0.05f + i*0.05f;
        float omega = 2*M_PI*f;
        freq_vals[i] = omega;
        RAO_surge[i] = 0.3f / sqrt(1+pow(omega*Tp/1.2f,2)); // 经验
        RAO_sway[i]  = 0.2f / sqrt(1+pow(omega*Tp/1.2f,2));
    }

    // allocate device memory
    allocateCudaMemory();

    // ------------------------ 生成波谱时间序列 ------------------------
    float df = 1.0f/T;
    float omega_p = 2*M_PI/Tp;
    float alpha = 5.f/16.f * Hs*Hs*powf(omega_p,4)/ (g*g);
    float gamma = 3.3f;

    std::vector<thrust::complex<float>> h_spec_ux(N);
    std::vector<thrust::complex<float>> h_spec_uy(N);
    std::vector<thrust::complex<float>> h_spec_ax(N);
    std::vector<thrust::complex<float>> h_spec_ay(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> ud(0.f, 2*M_PI);

    for(int k=1;k<N/2;k++){
        float f = k*df;
        float omega = 2*M_PI*f;
        float S = jonswap(omega, alpha, omega_p, gamma);
        if(S<=0.f) continue;

        float RAO_x = interpRAO(freq_vals, RAO_surge, omega);
        float RAO_y = interpRAO(freq_vals, RAO_sway, omega);

        float amp_eta = sqrt(2*S*df); // RMS缩放
        float phi = ud(rng);

        // 位移 -> 速度/加速度
        thrust::complex<float> spec_eta = thrust::polar(amp_eta, phi);
        thrust::complex<float> spec_ux  = spec_eta * thrust::complex<float>(0, omega) * RAO_x;
        thrust::complex<float> spec_uy  = spec_eta * thrust::complex<float>(0, omega) * RAO_y;
        thrust::complex<float> spec_ax  = spec_eta * thrust::complex<float>(-omega*omega, 0) * RAO_x;
        thrust::complex<float> spec_ay  = spec_eta * thrust::complex<float>(-omega*omega, 0) * RAO_y;

        h_spec_ux[k] = spec_ux;  h_spec_ux[N-k] = thrust::conj(spec_ux);
        h_spec_uy[k] = spec_uy;  h_spec_uy[N-k] = thrust::conj(spec_uy);
        h_spec_ax[k] = spec_ax;  h_spec_ax[N-k] = thrust::conj(spec_ax);
        h_spec_ay[k] = spec_ay;  h_spec_ay[N-k] = thrust::conj(spec_ay);
    }

    CUDA_CHECK(cudaMemcpy(d_spectrum_ux, h_spec_ux.data(), sizeof(thrust::complex<float>)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spectrum_uy, h_spec_uy.data(), sizeof(thrust::complex<float>)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spectrum_ax, h_spec_ax.data(), sizeof(thrust::complex<float>)*N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spectrum_ay, h_spec_ay.data(), sizeof(thrust::complex<float>)*N, cudaMemcpyHostToDevice));

    runIFFTAndStoreTimeSeries();
}

// ------------------------ 内存管理 ------------------------
void CudaWaveForceGenerator::allocateCudaMemory()
{
    CUDA_CHECK(cudaMalloc(&d_spectrum_ux, sizeof(thrust::complex<float>)*N));
    CUDA_CHECK(cudaMalloc(&d_spectrum_uy, sizeof(thrust::complex<float>)*N));
    CUDA_CHECK(cudaMalloc(&d_spectrum_ax, sizeof(thrust::complex<float>)*N));
    CUDA_CHECK(cudaMalloc(&d_spectrum_ay, sizeof(thrust::complex<float>)*N));
    CUDA_CHECK(cudaMalloc(&d_time_series, sizeof(float)*N*4));
}

void CudaWaveForceGenerator::freeCudaMemory()
{
    if(d_spectrum_ux){cudaFree(d_spectrum_ux); d_spectrum_ux=nullptr;}
    if(d_spectrum_uy){cudaFree(d_spectrum_uy); d_spectrum_uy=nullptr;}
    if(d_spectrum_ax){cudaFree(d_spectrum_ax); d_spectrum_ax=nullptr;}
    if(d_spectrum_ay){cudaFree(d_spectrum_ay); d_spectrum_ay=nullptr;}
    if(d_time_series){cudaFree(d_time_series); d_time_series=nullptr;}
}

// ------------------------ IFFT ------------------------
__global__ void packIFFTToTimeSeries(thrust::complex<float>* ux, thrust::complex<float>* uy,
                                     thrust::complex<float>* ax, thrust::complex<float>* ay,
                                     float* out, int N)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=N) return;
    float invN = 1.0f/N;
    out[4*i+0] = ux[i].real()*invN;
    out[4*i+1] = uy[i].real()*invN;
    out[4*i+2] = ax[i].real()*invN;
    out[4*i+3] = ay[i].real()*invN;
}

void CudaWaveForceGenerator::runIFFTAndStoreTimeSeries()
{
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_spectrum_ux),
                                  reinterpret_cast<cufftComplex*>(d_spectrum_ux),
                                  CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_spectrum_uy),
                                  reinterpret_cast<cufftComplex*>(d_spectrum_uy),
                                  CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_spectrum_ax),
                                  reinterpret_cast<cufftComplex*>(d_spectrum_ax),
                                  CUFFT_INVERSE));
    CUFFT_CHECK(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(d_spectrum_ay),
                                  reinterpret_cast<cufftComplex*>(d_spectrum_ay),
                                  CUFFT_INVERSE));
    CUFFT_CHECK(cufftDestroy(plan));

    int threads = 256;
    int blocks = (N+threads-1)/threads;
    packIFFTToTimeSeries<<<blocks, threads>>>(d_spectrum_ux,d_spectrum_uy,d_spectrum_ax,d_spectrum_ay,d_time_series,N);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ------------------------ 获取时间序列 ------------------------
Eigen::Vector4f CudaWaveForceGenerator::getWaveKinematicsGlobal(float t)
{
    int idx = static_cast<int>(t/dt) % N;
    if(idx<0) idx+=N;
    float h[4];
    CUDA_CHECK(cudaMemcpy(h, d_time_series+4*idx, sizeof(float)*4, cudaMemcpyDeviceToHost));
    return Eigen::Vector4f(h[0],h[1],h[2],h[3]);
}

// ------------------------ 获取船体波浪力 ------------------------
Eigen::Vector3f CudaWaveForceGenerator::getWaveForce(const Eigen::Matrix<float,6,1> &state, float t)
{
    float psi = state(2);
    float u_ship = state(3), v_ship = state(4);

    Eigen::Vector4f kin = getWaveKinematicsGlobal(t);
    Eigen::Vector2f u_wave_body;
    u_wave_body <<  cos(psi)*kin(0) + sin(psi)*kin(1),
                   -sin(psi)*kin(0) + cos(psi)*kin(1);
    Eigen::Vector2f udot_wave_body;
    udot_wave_body << cos(psi)*kin(2) + sin(psi)*kin(3),
                      -sin(psi)*kin(2) + cos(psi)*kin(3);

    Eigen::Vector2f u_rel = u_wave_body - Eigen::Vector2f(u_ship,v_ship);

    float Fx = rho*water_depth*B*udot_wave_body(0);
    float Fy = rho*water_depth*L*udot_wave_body(1);
    float N  = 0.5f*rho*water_depth*L*B*udot_wave_body.norm(); // 简化偏航

    return Eigen::Vector3f(Fx,Fy,N);
}

// ------------------------ 析构 ------------------------
CudaWaveForceGenerator::~CudaWaveForceGenerator()
{
    freeCudaMemory();
}
