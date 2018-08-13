#include "ScatteringWrapper.hpp"
#include "Error.hpp"

#include <cfloat>
#include <ctime>
#include <random>

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            __global__ void testRandCoord(Vec3<float> *orig, Vec3<float> *vel,
                                          float *int_times, Vec3<float> *pos,
                                          float *scat_times, float *rands,
                                          const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::randCoord(orig[index], vel[index], &(int_times[2*index]), pos[index], scat_times[index], rands[index]);
                }
            }

            __global__ void testIsoScatterKernel(Vec3<float> *vel, float *rands,
                                                 const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::isotropicScatteringKernel(vel[index], &(rands[2*index]));
                }
            }

            void randTest(Vec3<float> &orig, Vec3<float> &vel,
                          float *int_times, Vec3<float> &pos,
                          float &scat_times)
            {
                pos = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                scat_times = -5;
                std::mt19937 rng(time(NULL));
                std::uniform_real_distribution<float> dist(0.f, 1.f);
                float r = dist(rng);
                Vec3<float> *d_orig, *d_vel, *d_pos;
                float *d_itimes, *d_stimes, *d_rand;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_pos, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_itimes, 2*sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_stimes, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_rand, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, &orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_pos, &pos, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_itimes, int_times, 2*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_stimes, &scat_times, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_rand, &r, sizeof(float), cudaMemcpyHostToDevice) );
                testRandCoord<<<1, 1>>>(d_orig, d_vel, d_itimes, d_pos, d_stimes, d_rand, 1);
                CudaErrchk( cudaMemcpy(&pos, d_pos, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&scat_times, d_stimes, sizeof(float), cudaMemcpyDeviceToHost) );
            }

            void scatteringSiteTest(Vec3<float> &orig, Vec3<float> &vel, 
                                    float *int_times, Vec3<float> &pos,
                                    float &scat_times)
            {
                pos = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                scat_times = -5;
                std::mt19937 rng(time(NULL));
                std::uniform_real_distribution<float> dist(0.f, 1.f);
                float r = dist(rng);
                Vec3<float> *d_orig, *d_vel, *d_pos;
                float *d_itimes, *d_stimes, *d_rand;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_pos, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_itimes, 2*sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_stimes, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_rand, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, &orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_pos, &pos, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_itimes, int_times, 2*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_stimes, &scat_times, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_rand, &r, sizeof(float), cudaMemcpyHostToDevice) );
                kernels::calcScatteringSites<<<1, 1>>>(d_itimes, d_orig, d_vel, d_pos, d_stimes, d_rand, 1);
                CudaErrchk( cudaMemcpy(&pos, d_pos, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&scat_times, d_stimes, sizeof(float), cudaMemcpyDeviceToHost) );
            }

            void isoScatterTest(Vec3<float> &vel)
            {
                float *rands = new float[2];
                std::mt19937 rng(time(NULL));
                std::uniform_real_distribution<float> dist(0.f, 1.f);
                rands[0] = dist(rng);
                rands[1] = dist(rng);
                Vec3<float> *d_vel;
                float *d_rand;
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_rand, 2*sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_rand, rands, 2*sizeof(float), cudaMemcpyHostToDevice) );
                testIsoScatterKernel<<<1, 1>>>(d_vel, d_rand, 1);
                CudaErrchk( cudaMemcpy(&vel, d_vel, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                delete [] rands;
            }

            void scatterTest(const int key, const float &time,
                             Vec3<float> &vel)
            {
                float *rands = new float[2];
                //std::mt19937 rng(time(NULL));
                std::mt19937 rng;
                std::uniform_real_distribution<float> dist(0.f, 1.f);
                rands[0] = dist(rng);
                rands[1] = dist(rng);
                Vec3<float> *d_vel;
                float *d_rand, *d_times;
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_rand, 2*sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_times, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_rand, rands, 2*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_times, &time, sizeof(float), cudaMemcpyHostToDevice) );
                kernels::scatter<<<1, 1>>>(key, d_times, d_vel, d_rand, 1);
                CudaErrchkNoCode();
                CudaErrchk( cudaMemcpy(&vel, d_vel, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                delete [] rands;
            }

        }

    }

}
