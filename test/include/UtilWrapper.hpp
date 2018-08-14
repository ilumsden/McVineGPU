#ifndef UTIL_KERNEL_TEST_HPP
#define UTIL_KERNEL_TEST_HPP

#include <vector>

#include "UtilKernels.hpp"
#include "SystemVars.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            __global__ void testQuadratic(float *a, float *b, float *c,
                                          float *x0, float *x1, 
                                          bool *solved);

            void testInitArray(std::vector<float> &data, const float val);

            void testSolveQuadratic(float a, float b, float c,
                                    float &x0, float &x1, bool &solved);

            void testSimplifyPairs(std::vector<float> &times,
                                   std::vector< Vec3<float> > &coords,
                                   const int input_groups,
                                   const int numOutputs);

            void testForceIntOrder(std::vector<float> &ts,
                                   std::vector< Vec3<float> > &coords);

            void testPropagate(Vec3<float> &orig, float &time,
                               Vec3<float> &new_orig, float &new_time);

            void testUpdateProbability(float &prob,
                                       Vec3<float> &p1, Vec3<float> &p0);

        }

    }

}

#endif
