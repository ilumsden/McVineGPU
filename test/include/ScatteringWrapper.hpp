#ifndef SCATTERING_KERNEL_TEST_HPP
#define SCATTERING_KERNEL_TEST_HPP

#include "ScatteringKernels.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {
            template<typename T>
            bool isBetween(T val, T lower, T upper)
            {
                return (val > lower) && (val < upper);
            }


            namespace kernels = mcvine::gpu::kernels;

            __global__ void testRandCoord(Vec3<float> *orig, Vec3<float> *vel,
                                          float *int_times, Vec3<float> *pos,
                                          float *scat_times, float *rands,
                                          const int N);

            __global__ void testIsoScatterKernel(Vec3<float> *vel, float *rands,
                                                 const int N);

            void randTest(Vec3<float> &orig, Vec3<float> &vel,
                          float *int_times, Vec3<float> &pos,
                          float &scat_times);

            void scatteringSiteTest(Vec3<float> &orig, Vec3<float> &vel, 
                                    float *int_times, Vec3<float> &pos,
                                    float &scat_times);

            void isoScatterTest(Vec3<float> &vel);

            void scatterTest(const int key, const float &time,
                             Vec3<float> &vel);

        }

    }

}

#endif
