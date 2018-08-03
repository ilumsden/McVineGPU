#include "IntersectWrapper.hpp"
#include "Error.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            namespace kernels = mcvine::gpu::kernels;

            __global__ void testIntRectangle(float *ts, Vec3<float> *pts,
                                             const Vec3<float> *orig,
                                             const Vec3<float> *vel,
                                             const float X, const float Y, const float Z,
                                             const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::intersectRectangle(ts[index], pts[index], orig[index], X/2, vel[index], Y, Z, 2, 0);
                }
            }

            __global__ void testIntCylSide(float *ts, Vec3<float> *pts,
                                           const Vec3<float> *orig,
                                           const Vec3<float> *vel,
                                           const float r, const float h, const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::intersectCylinderSide(&(ts[2*index]), &(pts[2*index]), orig[index], vel[index], r, h, 0);
                }
            }

            __global__ void testIntCylTopBottom(float *ts, Vec3<float> *pts,
                                                const Vec3<float> *orig,
                                                const Vec3<float> *vel,
                                                const float r, const float h, const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::intersectCylinderTopBottom($(ts[2*index]), $(pts[2*index]), orig[index], vel[index], r, h, 0);
                }
            }

            __global__ void testIntTriangle(float *ts, Vec3<float> *pts,
                                            const Vec3<float> *orig,
                                            const Vec3<float> *vel,
                                            const Vec3<float> *verts,
                                            const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    kernels::intersectTriangle(ts[index], pts[index], orig[index], vel[index], verts[0], verts[1], verts[2], 0);
                }
            }

            void rectangleTest(Vec3<float> &orig, Vec3<float> &vel, float &time, Vec3<float> &point)
            {
                time = -5; point = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                float X = -0.002; float Y = 0.05; float Z = 0.1;
                Vec3<float> *d_orig, *d_vel, *pts;
                float *ts;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&pts, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&ts, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(pts, &point, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(ts, &time, sizeof(float), cudaMemcpyHostToDevice) );
                testIntRectangle<<<1, 1>>>(ts, pts, d_orig, d_vel, X, Y, Z, 1);
                CudaErrchk( cudaMemcpy(&time, ts, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&point, pts, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_vel) );
                CudaErrchk( cudaFree(pts) );
                CudaErrchk( cudaFree(ts) );
            }

            void cylinderSideTest(Vec3<float> &orig, Vec3<float> &vel, float *time, Vec3<float> *point)
            {
                time[0] = -5; time[1] = -5;
                point[0] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                point[1] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                float radius = 0.002; float height = 0.1;
                Vec3<float> *d_orig, *d_vel, *pts;
                float *ts;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&pts, 2*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&ts, 2*sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(pts, &point, 2*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(ts, &time, 2*sizeof(float), cudaMemcpyHostToDevice) );
                testIntCylSide<<<1, 1>>>(ts, pts, d_orig, d_vel, radius, height, 1);
                CudaErrchk( cudaMemcpy(&time, ts, 2*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&point, pts, 2*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_vel) );
                CudaErrchk( cudaFree(pts) );
                CudaErrchk( cudaFree(ts) );
            }

            void cylinderEndTest(Vec3<float> &orig, Vec3<float> &vel, float *time, Vec3<float> *point)
            {
                time[0] = -5; time[1] = -5;
                point[0] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                point[1] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                float radius = 0.002; float height = 0.1;
                Vec3<float> *d_orig, *d_vel, *pts;
                float *ts;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&pts, 2*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&ts, 2*sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(pts, &point, 2*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(ts, &time, 2*sizeof(float), cudaMemcpyHostToDevice) );
                testIntCylTopBottom<<<1, 1>>>(ts, pts, d_orig, d_vel, radius, height, 1);
                CudaErrchk( cudaMemcpy(&time, ts, 2*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&point, pts, 2*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_vel) );
                CudaErrchk( cudaFree(pts) );
                CudaErrchk( cudaFree(ts) );
            }

            void triangleTest(Vec3<float> &orig, Vec3<float> &vel, float &time, Vec3<float> &point)
            {
                time = -5; point = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                Vec3<float> verts[3] = { Vec3<float>(0, 0, 0), Vec3<float>(-0.001, 0.025, -0.1), Vec3<float>(-0.001, -0.025, -0.1) };
                Vec3<float> *d_orig, *d_vel, *d_verts, *pts;
                float *ts;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_verts, 3*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&pts, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&ts, sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_verts, &(verts[0]), 3*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(pts, &point, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(ts, &time, sizeof(float), cudaMemcpyHostToDevice) );
                testIntRectangle<<<1, 1>>>(ts, pts, d_orig, d_vel, X, Y, Z, 1);
                CudaErrchk( cudaMemcpy(&time, ts, sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(&point, pts, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_vel) );
                CudaErrchk( cudaFree(d_verts) );
                CudaErrchk( cudaFree(pts) );
                CudaErrchk( cudaFree(ts) );
            }

            void 3DTest(const int key, float *time, Vec3<float> *point)
            {
                std::vector<float> sd;
                Vec3<float> orig, vel;
                int groupsize = 1;
                switch(key)
                {
                    case 0: 
                        sd.push_back(0.002); sd.push_back(0.05); sd.push_back(0.1);
                        groupsize = 6;
                        orig = Vec3<float>(-0.002, 0, 0.05);
                        vel = Vec3<float>(0.001, 0, -0.05);
                        break;
                    case 1:
                        sd.push_back(0.05); sd.push_back(0.1);
                        groupsize = 4;
                        orig = Vec3<float>(0.1, -0.025, -0.05);
                        break;
                    case 2:
                        sd.push_back(0.002); sd.push_back(0.05); sd.push_back(0.1);
                        groupsize = 5;
                        orig = Vec3<float>(-0.001, 0.03, -0.16);
                        vel = Vec3<float>(0.0005, -0.01, 0.06);
                        break;
                    case 3:
                        sd.push_back(0.1);
                        orig = Vec3<float>(-0.5, 0, 0);
                        vel = Vec3<float>(1, 0, 0);
                        groupsize = 2;
                        break;
                    default:
                        sd.push_back(-1);
                        orig = Vec3<float>(-5, -5, -5);
                        vel = Vec3<float>(-1, -1, -1);
                }
                for (int i = 0; i < groupsize; i++)
                {
                    time[i] = -5;
                }
                point[0] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                point[1] = Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX);
                Vec3<float> *d_orig, *d_vel, *pts;
                float *shapeData, *ts;
                CudaErrchk( cudaMalloc(&d_orig, sizeof(Vec3<float)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float)) );
                CudaErrchk( cudaMalloc(&pts, 2*sizeof(Vec3<float)) );
                CudaErrchk( cudaMalloc(&shapeData, ((int)(sd.size()))*sizeof(float)) );
                CudaErrchk( cudaMalloc(&ts, groupsize*sizeof(float)) );
                CudaErrchk( cudaMemcpy(d_orig, &orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(pts, point, 2*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(shapeData, &(sd[0]), ((int)(sd.size()))*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(ts, time, groupsize*sizeof(float), cudaMemcpyHostToDevice) );
                kernels::intersect<<<1, 1>>>(key, d_orig, d_vel, shapeData, 1, ts, pts);
                CudaErrchk( cudaMemcpy(time, ts, groupsize*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(point, pts, 2*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaFree(d_orig) );
                CudaErrchk( cudaFree(d_vel) );
                CudaErrchk( cudaFree(pts) );
                CudaErrchk( cudaFree(shapeData) );
                CudaErrchk( cudaFree(ts) );
            }

        }

    }

}
