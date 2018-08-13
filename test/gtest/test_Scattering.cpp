#include <cstdlib>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "ScatteringWrapper.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            TEST(ScatteringKernelTest, Rand)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                // Times are based on intersection with a cube with
                // edge length 0.2
                float *int_times = new float[2];
                int_times[0] = 0.4;
                int_times[1] = 0.6;
                Vec3<float> pos;
                float scat_times;
                randTest(orig, vel, int_times, pos, scat_times);
                ASSERT_PRED3(isBetween<float>, scat_times, int_times[0], int_times[1]);
                Vec3<float> comp_vel = (pos - orig) / scat_times;
                EXPECT_NEAR(comp_vel[0], vel[0], 1e-6);
                EXPECT_NEAR(comp_vel[1], vel[1], 1e-6);
                EXPECT_NEAR(comp_vel[2], vel[2], 1e-6);
                delete [] int_times;
            }

            TEST(ScatteringKernelTest, ScatteringSiteIntersection)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                // Times are based on intersection with a cube with
                // edge length 0.2
                float *int_times = new float[2];
                int_times[0] = 0.4;
                int_times[1] = 0.6;
                Vec3<float> pos;
                float scat_times;
                scatteringSiteTest(orig, vel, int_times, pos, scat_times);
                ASSERT_PRED3(isBetween<float>, scat_times, int_times[0], int_times[1]);
                Vec3<float> comp_vel = (pos - orig) / scat_times;
                EXPECT_NEAR(comp_vel[0], vel[0], 1e-6);
                EXPECT_NEAR(comp_vel[1], vel[1], 1e-6);
                EXPECT_NEAR(comp_vel[2], vel[2], 1e-6);
                delete [] int_times;
            }

            TEST(ScatteringKernelTest, ScatteringSiteNoIntersection)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                // Times are based on intersection with a cube with
                // edge length 0.2
                float *int_times = new float[2];
                int_times[0] = -1;
                int_times[1] = -1;
                Vec3<float> pos;
                float scat_times;
                scatteringSiteTest(orig, vel, int_times, pos, scat_times);
                EXPECT_NEAR(scat_times, -5.f, 1e-6);
                EXPECT_NEAR(pos[0], FLT_MAX, 1e-6);
                EXPECT_NEAR(pos[1], FLT_MAX, 1e-6);
                EXPECT_NEAR(pos[2], FLT_MAX, 1e-6);
                delete [] int_times;
            }

            TEST(ScatteringKernelTest, IsoScattering)
            {
                Vec3<float> vel(5, 8, 3);
                Vec3<float> cpy = vel;
                isoScatterTest(vel);
                EXPECT_NEAR(vel.length(), cpy.length(), 1e-6);
            }

            TEST(ScatteringKernelTest, GeneralScatteringForIsoIntersection)
            {
                int key = 0;
                float time = 0.5;
                Vec3<float> vel(5, 8, 3);
                Vec3<float> cpy = vel;
                scatterTest(key, time, vel);
                EXPECT_NEAR(vel.length(), cpy.length(), 1e-6);
                EXPECT_FALSE(vel == cpy);
                //EXPECT_FALSE(abs(vel[0] - cpy[0]) < 1e-6);
                //EXPECT_FALSE(abs(vel[1] - cpy[1]) < 1e-6);
                //EXPECT_FALSE(abs(vel[2] - cpy[2]) < 1e-6);
            }

            TEST(ScatteringKernelTest, GeneralScatteringForIsoNoIntersection)
            {
                int key = 0;
                int time = -1;
                Vec3<float> vel(5, 8, 3);
                Vec3<float> cpy = vel;
                scatterTest(key, time, vel);
                EXPECT_NEAR(vel.length(), cpy.length(), 1e-6);
                EXPECT_TRUE(vel == cpy);
                //EXPECT_TRUE(abs(vel[0] - cpy[0]) < 1e-6);
                //EXPECT_TRUE(abs(vel[1] - cpy[1]) < 1e-6);
                //EXPECT_TRUE(abs(vel[2] - cpy[2]) < 1e-6);
            }

        }

    }

}
