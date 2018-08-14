#include <cmath>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "UtilWrapper.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            TEST(UtilKernelTest, TestInitArray)
            {
                std::vector<float> data;
                const float val = 5.f;
                testInitArray(data, val);
                for (float d : data)
                {
                    EXPECT_NEAR(d, val, 1e-6);
                }
            }

            TEST(UtilKernelTest, TestSolveQuad2Sol)
            {
                float a = 1.f;
                float b = 0.f;
                float c = -1.f;
                float x0, x1;
                bool solved;
                testSolveQuadratic(a, b, c, x0, x1, solved);
                ASSERT_TRUE(solved);
                EXPECT_NEAR(x0, -1.f, 1e-6);
                EXPECT_NEAR(x1, 1.f, 1e-6);
            }

            TEST(UtilKernelTest, TestSolveQuad1Sol)
            {
                float a = 1.f;
                float b = -8.f;
                float c = 16.f;
                float x0, x1;
                bool solved;
                testSolveQuadratic(a, b, c, x0, x1, solved);
                ASSERT_TRUE(solved);
                EXPECT_NEAR(x0, 4.f, 1e-6);
                EXPECT_NEAR(x1, 4.f, 1e-6);
            }

            TEST(UtilKernelTest, TestSolveQuadNoSol)
            {
                float a = 1.f;
                float b = -4.f;
                float c = 10.f;
                float x0, x1;
                bool solved;
                testSolveQuadratic(a, b, c, x0, x1, solved);
                ASSERT_FALSE(solved);
            }

            TEST(UtilKernelTest, TestSimpPairsGroup6)
            {
                std::vector<float> times { 0.7, -1, 0.5, -1, -1, -1 };
                std::vector< Vec3<float> > coords { Vec3<float>(1, 0, 0),
                                                    Vec3<float>(-1, 0, 0) };
                testSimplifyPairs(times, coords, 6, 2);
                EXPECT_EQ((int)(times.size()), 2);
                EXPECT_EQ((int)(coords.size()), 2);
                EXPECT_NEAR(times[0], 0.7, 1e-6);
                EXPECT_NEAR(times[1], 0.5, 1e-6);
                EXPECT_TRUE(coords[0] == Vec3<float>(1, 0, 0));
                EXPECT_TRUE(coords[1] == Vec3<float>(-1, 0, 0));
            }

            TEST(UtilKernelTest, TestSimpPairsGroup5)
            {
                std::vector<float> times { 0.1, -1, -1, -1, -1 };
                std::vector< Vec3<float> > coords { Vec3<float>(0, 0, -5),
                                                    Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX) };
                testSimplifyPairs(times, coords, 5, 1);
                EXPECT_EQ((int)(times.size()), 1);
                EXPECT_EQ((int)(coords.size()), 1);
                EXPECT_NEAR(times[0], 0.1, 1e-6);
                EXPECT_TRUE(coords[0] == Vec3<float>(0, 0, -5));
            }

            TEST(UtilKernelTest, TestForceIntOrder)
            {
                std::vector<float> times { 0.7, 0.5 };
                std::vector< Vec3<float> > coords { Vec3<float>(1, 0, 0),
                                                    Vec3<float>(-1, 0, 0) };
                testForceIntOrder(times, coords);
                EXPECT_NEAR(times[0], 0.5, 1e-6);
                EXPECT_NEAR(times[1], 0.7, 1e-6);
                EXPECT_TRUE(coords[0] == Vec3<float>(-1, 0, 0));
                EXPECT_TRUE(coords[1] == Vec3<float>(1, 0, 0));
            }

            TEST(UtilKernelTest, TestPropagate)
            {
                Vec3<float> orig(0, 0, 0);
                Vec3<float> new_orig(1, 3, 2);
                float time = 0.25;
                float new_time = 0.25;
                testPropagate(orig, time, new_orig, new_time);
                EXPECT_TRUE(orig == new_orig);
                EXPECT_NEAR(time, 0.5, 1e-6);
            }

            TEST(UtilKernelTest, TestUpdateProb)
            {
                float prob = 1;
                Vec3<float> p1(2, 1, 1);
                Vec3<float> p0(1, 0, 0);
                testUpdateProbability(prob, p1, p0);
                EXPECT_NEAR(prob, exp(-(1/atten)), 1e-6);
            }

        }

    }

}
