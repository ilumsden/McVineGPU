#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "Ray.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            TEST(RayTest, ThreeConstructor)
            {
                Ray test(3, 4, 5);
                EXPECT_FLOAT_EQ(test.origin[0], 3.f);
                EXPECT_FLOAT_EQ(test.origin[1], 4.f);
                EXPECT_FLOAT_EQ(test.origin[2], 5.f);
                EXPECT_FLOAT_EQ(test.vel[0], 0.f);
                EXPECT_FLOAT_EQ(test.vel[1], 0.f);
                EXPECT_FLOAT_EQ(test.vel[2], 0.f);
                EXPECT_FLOAT_EQ(test.t, 0.f);
                EXPECT_FLOAT_EQ(test.prob, 1.f);
            }

            TEST(RayTest, SixConstructor)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                EXPECT_FLOAT_EQ(test.origin[0], 3.f);
                EXPECT_FLOAT_EQ(test.origin[1], 4.f);
                EXPECT_FLOAT_EQ(test.origin[2], 5.f);
                EXPECT_FLOAT_EQ(test.vel[0], 1.f);
                EXPECT_FLOAT_EQ(test.vel[1], 2.f);
                EXPECT_FLOAT_EQ(test.vel[2], 3.f);
                EXPECT_FLOAT_EQ(test.t, 0.f);
                EXPECT_FLOAT_EQ(test.prob, 1.f);
            }

            TEST(RayTest, EightConstructor)
            {
                Ray test(3, 4, 5, 1, 2, 3, 0.5, 0.1);
                EXPECT_FLOAT_EQ(test.origin[0], 3.f);
                EXPECT_FLOAT_EQ(test.origin[1], 4.f);
                EXPECT_FLOAT_EQ(test.origin[2], 5.f);
                EXPECT_FLOAT_EQ(test.vel[0], 1.f);
                EXPECT_FLOAT_EQ(test.vel[1], 2.f);
                EXPECT_FLOAT_EQ(test.vel[2], 3.f);
                EXPECT_FLOAT_EQ(test.t, 0.5f);
                EXPECT_FLOAT_EQ(test.prob, 0.1f);
            }

            TEST(RayTest, VecConstructor)
            {
                Vec3<float> orig(3, 4, 5);
                Vec3<float> v(1, 2, 3);
                Ray test(orig, v, 0.5, 0.1);
                EXPECT_FLOAT_EQ(test.origin[0], 3.f);
                EXPECT_FLOAT_EQ(test.origin[1], 4.f);
                EXPECT_FLOAT_EQ(test.origin[2], 5.f);
                EXPECT_FLOAT_EQ(test.vel[0], 1.f);
                EXPECT_FLOAT_EQ(test.vel[1], 2.f);
                EXPECT_FLOAT_EQ(test.vel[2], 3.f);
                EXPECT_FLOAT_EQ(test.t, 0.5f);
                EXPECT_FLOAT_EQ(test.prob, 0.1f);
            }

            TEST(RayTest, setVelocities)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.setVelocities(11, 12, 13);
                EXPECT_FLOAT_EQ(test.vel[0], 11.f);
                EXPECT_FLOAT_EQ(test.vel[1], 12.f);
                EXPECT_FLOAT_EQ(test.vel[2], 13.f);
            }

            TEST(RayTest, setTime)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.setTime(10);
                EXPECT_FLOAT_EQ(test.t, 10.f);
            }

            TEST(RayTest, setProb)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.setProbability(0.005);
                EXPECT_FLOAT_EQ(test.prob, 0.005f);
            }

            TEST(RayTest, FourFltUpdate)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.update(5, 12, 13, 3);
                EXPECT_FLOAT_EQ(test.origin[0], 5.f);
                EXPECT_FLOAT_EQ(test.origin[1], 12.f);
                EXPECT_FLOAT_EQ(test.origin[2], 13.f);
                EXPECT_FLOAT_EQ(test.t, 3.f);
            }

            TEST(RayTest, ThreeFltUpdate)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.update(11, 12, 13);
                EXPECT_FLOAT_EQ(test.vel[0], 11.f);
                EXPECT_FLOAT_EQ(test.vel[1], 12.f);
                EXPECT_FLOAT_EQ(test.vel[2], 13.f);
            }

            TEST(RayTest, OneFltUpdate)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.update(0.005);
                EXPECT_FLOAT_EQ(test.prob, 0.005f);
            }

            TEST(RayTest, EightFltUpdate)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                test.update(5, 12, 13, 11, 12, 13, 3, 0.005);
                EXPECT_FLOAT_EQ(test.origin[0], 5.f);
                EXPECT_FLOAT_EQ(test.origin[1], 12.f);
                EXPECT_FLOAT_EQ(test.origin[2], 13.f);
                EXPECT_FLOAT_EQ(test.vel[0], 11.f);
                EXPECT_FLOAT_EQ(test.vel[1], 12.f);
                EXPECT_FLOAT_EQ(test.vel[2], 13.f);
                EXPECT_FLOAT_EQ(test.t, 3.f);
                EXPECT_FLOAT_EQ(test.prob, 0.005f);
            }

            TEST(RayTest, VecUpdate)
            {
                Ray test(3, 4, 5, 1, 2, 3);
                Vec3<float> t0(5, 12, 13);
                Vec3<float> t1(11, 12, 13);
                test.update(t0, t1, 3, 0.005);
                EXPECT_FLOAT_EQ(test.origin[0], 5.f);
                EXPECT_FLOAT_EQ(test.origin[1], 12.f);
                EXPECT_FLOAT_EQ(test.origin[2], 13.f);
                EXPECT_FLOAT_EQ(test.vel[0], 11.f);
                EXPECT_FLOAT_EQ(test.vel[1], 12.f);
                EXPECT_FLOAT_EQ(test.vel[2], 13.f);
                EXPECT_FLOAT_EQ(test.t, 3.f);
                EXPECT_FLOAT_EQ(test.prob, 0.005f);
            }

        }

    }

}
