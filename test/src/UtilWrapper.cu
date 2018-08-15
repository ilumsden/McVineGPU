#include "UtilWrapper.hpp"

#include "Error.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            namespace kernels = mcvine::gpu::kernels;

            __global__ void testQuadratic(float *a, float *b, float *c,
                                          float *x0, float *x1,
                                          bool *solved, const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    solved[index] = kernels::solveQuadratic(a[index], b[index],
                                                            c[index], x0[index],
                                                            x1[index]);
                }
            }

            void testInitArray(std::vector<float> &data, const float val)
            {
                data.clear();
                data.resize(10);
                float *d_data;
                CudaErrchk( cudaMalloc(&d_data, 10*sizeof(float)) );
                kernels::initArray<float><<<1, 10>>>(d_data, 10, val);
                CudaErrchkNoCode();
                float *d = data.data();
                CudaErrchk( cudaMemcpy(d, d_data, 10*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_data) );
            }

            void testSolveQuadratic(float a, float b, float c,
                                    float &x0, float &x1, bool &solved)
            {
                float *d_a, *d_b, *d_c;
                float *d_x0, *d_x1;
                bool *d_solve;
                CudaErrchk( cudaMalloc(&d_a, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_b, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_c, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_x0, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_x1, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_solve, sizeof(bool)) );
                CudaErrchk( cudaMemcpy(d_a, &a, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_b, &b, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_x0, &x0, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_x1, &x1, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_solve, &solved, sizeof(bool), cudaMemcpyHostToDevice) );
                testQuadratic<<<1, 1>>>(d_a, d_b, d_c, d_x0, d_x1, d_solve, 1);
                CudaErrchkNoCode();
                CudaErrchk( cudaMemcpy(&x0, d_x0, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&x1, d_x1, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&solved, d_solve, sizeof(bool), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_a) );
                CudaErrchk( cudaFree(d_b) );
                CudaErrchk( cudaFree(d_c) );
                CudaErrchk( cudaFree(d_x0) );
                CudaErrchk( cudaFree(d_x1) );
                CudaErrchk( cudaFree(d_solve) );
            }

            void testSimplifyPairs(std::vector<float> &times,
                                   std::vector< Vec3<float> > &coords, 
                                   const int input_groups,
                                   const int numOutputs)
            {
                // To simulate the behavior of the actual code, 
                // the coordinate groupsize is considered to be 2.
                int size_times = times.size();
                int size_coords = coords.size();
                int threads = size_coords / 2;
                float *d_times, *s_times;
                Vec3<float> *d_coords, *s_coords;
                CudaErrchk( cudaMalloc(&d_times, size_times*sizeof(float)) );
                CudaErrchk( cudaMalloc(&s_times, ((size_times*numOutputs)/input_groups)*sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_coords, size_coords*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&s_coords, ((size_coords*numOutputs)/2)*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(d_times, times.data(), size_times*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_coords, coords.data(), size_coords*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                kernels::simplifyTimePointPairs<<<1, threads>>>(d_times, d_coords,
                                                                threads, 
                                                                input_groups, 2,
                                                                numOutputs,
                                                                s_times, s_coords);
                CudaErrchkNoCode();
                times.resize((size_times*numOutputs)/input_groups);
                coords.resize((size_coords*numOutputs)/2);
                float *dt = times.data();
                Vec3<float> *dc = coords.data();
                CudaErrchk( cudaMemcpy(dt, s_times, ((size_times*numOutputs)/input_groups)*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(dc, s_coords, ((size_coords*numOutputs)/2)*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_times) );
                CudaErrchk( cudaFree(s_times) );
                CudaErrchk( cudaFree(d_coords) );
                CudaErrchk( cudaFree(s_coords) );
            }

            void testForceIntOrder(std::vector<float> &ts,
                                   std::vector< Vec3<float> > &coords)
            {
                float *d_times;
                Vec3<float> *d_coords;
                CudaErrchk( cudaMalloc(&d_times, (int)(ts.size())*sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_coords, (int)(coords.size())*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(d_times, ts.data(), (int)(ts.size())*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_coords, coords.data(), (int)(coords.size())*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                kernels::forceIntersectionOrder<<<1, ((int)(ts.size())/2)>>>(d_times, d_coords, ((int)(ts.size())/2));
                CudaErrchkNoCode();
                float *t = ts.data();
                Vec3<float> *dc = coords.data();
                CudaErrchk( cudaMemcpy(t, d_times, (int)(ts.size())*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(dc, d_coords, (int)(coords.size())*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_times) );
                CudaErrchk( cudaFree(d_coords) );
            }

            void testPropagate(Vec3<float> &orig, float &time,
                               Vec3<float> &new_orig, float &new_time)
            {
                Vec3<float> *d_orig, *d_pos;
                float *d_time, *d_ntime;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_pos, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_time, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_ntime, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, &orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_pos, &new_orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_time, &time, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_ntime, &new_time, sizeof(float), cudaMemcpyHostToDevice) );
                kernels::propagate<<<1, 1>>>(d_orig, d_time, d_pos, d_ntime, 1);
                CudaErrchkNoCode();
                CudaErrchk( cudaMemcpy(&orig, d_orig, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&time, d_time, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_pos) );
                CudaErrchk( cudaFree(d_time) );
                CudaErrchk( cudaFree(d_ntime) );
            }

            void testUpdateProbability(float &prob,
                                       Vec3<float> &p1, Vec3<float> &p0)
            {
                float *d_prob;
                Vec3<float> *d_p1, *d_p0;
                CudaErrchk( cudaMalloc(&d_prob, sizeof(float)) );
                CudaErrchk( cudaMalloc(&d_p1, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_p0, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(d_prob, &prob, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_p1, &p1, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_p0, &p0, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                kernels::updateProbability<<<1, 1>>>(d_prob, d_p1, d_p0, 1, 1, atten, 1);
                CudaErrchkNoCode();
                CudaErrchk( cudaMemcpy(&prob, d_prob, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_prob) );
                CudaErrchk( cudaFree(d_p1) );
                CudaErrchk( cudaFree(d_p0) );
            }

        }

    }

}
