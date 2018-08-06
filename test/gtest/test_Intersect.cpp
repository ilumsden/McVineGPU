#include "gtest/gtest.h"
#include "IntersectWrapper.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            TEST(IntersectTest, Rectangle2DIntersect)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                float time;
                Vec3<float> point;
                rectangleTest(orig, vel, time, point);
                EXPECT_NEAR(time, 0.499, 1e-6);
                EXPECT_NEAR(point[0], -0.001, 1e-6);
                EXPECT_NEAR(point[1], 0.f, 1e-6);
                EXPECT_NEAR(point[2], 0.f, 1e-6);
            }

            TEST(IntersectTest, Rectangle2DNoIntersect)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel = Vec3<float>(0, 0, 1);
                float time;
                Vec3<float> point;
                rectangleTest(orig, vel, time, point);
                EXPECT_NEAR(time, -1, 1e-6);
                EXPECT_NEAR(point[0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[2], FLT_MAX, 1e-6);
            }

            TEST(IntersectTest, CylinderSide2DTwoIntersect)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderSideTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], 0.45, 1e-6);
                EXPECT_NEAR(point[0][0], -0.05, 1e-6);
                EXPECT_NEAR(point[0][1], 0.f, 1e-6);
                EXPECT_NEAR(point[0][2], 0.f, 1e-6);
                EXPECT_NEAR(time[1], 0.55, 1e-6);
                EXPECT_NEAR(point[1][0], 0.05, 1e-6);
                EXPECT_NEAR(point[1][1], 0.f, 1e-6);
                EXPECT_NEAR(point[1][2], 0.f, 1e-6);
                delete [] time;
                delete [] point;
            }


            TEST(IntersectTest, CylinderSide2DNoIntersect)
            {
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(0, 1, 0);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderSideTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], -1.f, 1e-6);
                EXPECT_NEAR(point[0][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[0][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[0][2], FLT_MAX, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(point[1][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][2], FLT_MAX, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, CylinderSide2DOneIntersect)
            {
                Vec3<float> orig(0.1, -0.025, -0.05);
                Vec3<float> vel(-0.05, 0.025, 0.05);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderSideTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], 1.f, 1e-6);
                EXPECT_NEAR(point[0][0], 0.05, 1e-6);
                EXPECT_NEAR(point[0][1], 0.f, 1e-6);
                EXPECT_NEAR(point[0][2], 0.f, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(point[1][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][2], FLT_MAX, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, CylinderEnd2DTwoIntersect)
            {
                Vec3<float> orig(0, 0, 0.5);
                Vec3<float> vel(0, 0, -1);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderEndTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], 0.45, 1e-6);
                EXPECT_NEAR(point[0][0], 0.f, 1e-6);
                EXPECT_NEAR(point[0][1], 0.f, 1e-6);
                EXPECT_NEAR(point[0][2], 0.05, 1e-6);
                EXPECT_NEAR(time[1], 0.55, 1e-6);
                EXPECT_NEAR(point[1][0], 0.f, 1e-6);
                EXPECT_NEAR(point[1][1], 0.f, 1e-6);
                EXPECT_NEAR(point[1][2], -0.05, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, CylinderEnd2DNoIntersect)
            {
                Vec3<float> orig(0, 0, 0.5);
                Vec3<float> vel(1, 0, 0);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderEndTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], -1.f, 1e-6);
                EXPECT_NEAR(point[0][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[0][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[0][2], FLT_MAX, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(point[1][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][2], FLT_MAX, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, CylinderEnd2DOneIntersect)
            {
                Vec3<float> orig(0.1, -0.025, -0.05);
                Vec3<float> vel(-0.05, 0.025, 0.05);
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                cylinderEndTest(orig, vel, time, point);
                EXPECT_NEAR(time[0], 2.f, 1e-6);
                EXPECT_NEAR(point[0][0], 0.f, 1e-6);
                EXPECT_NEAR(point[0][1], 0.025, 1e-6);
                EXPECT_NEAR(point[0][2], 0.05, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(point[1][0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1][2], FLT_MAX, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, Triangle2DIntersect)
            {
                Vec3<float> orig(-0.001, 0.03, -0.16);
                Vec3<float> vel(0.0005, -0.01, 0.06);
                float time;
                Vec3<float> point;
                triangleTest(orig, vel, time, point);
                EXPECT_NEAR(time, 2.f, 1e-6);
                EXPECT_NEAR(point[0], 0.f, 1e-6);
                EXPECT_NEAR(point[1], 0.01, 1e-6);
                EXPECT_NEAR(point[2], -0.04, 1e-6);
            }

            TEST(IntersectTest, Triangle2DNoIntersect)
            {
                Vec3<float> orig(-0.001, 0.03, -0.16);
                Vec3<float> vel(-1, 0, 0);
                float time;
                Vec3<float> point;
                triangleTest(orig, vel, time, point);
                EXPECT_NEAR(time, -1.f, 1e-6);
                EXPECT_NEAR(point[0], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1], FLT_MAX, 1e-6);
                EXPECT_NEAR(point[1], FLT_MAX, 1e-6);
            }

            TEST(IntersectTest, Box3D)
            {
                int key = 0;
                float *time = new float[6];
                Vec3<float> *point = new Vec3<float>[2];
                solidTest(key, time, point);
                EXPECT_NEAR(time[0], -1.f, 1e-6);
                EXPECT_NEAR(time[1], 2.f, 1e-6);
                EXPECT_NEAR(time[2], -1.f, 1e-6);
                EXPECT_NEAR(time[3], 1.f, 1e-6);
                EXPECT_NEAR(time[4], -1.f, 1e-6);
                EXPECT_NEAR(time[5], -1.f, 1e-6);
                EXPECT_NEAR(point[0][0], 0.f, 1e-6);
                EXPECT_NEAR(point[0][1], 0.f, 1e-6);
                EXPECT_NEAR(point[0][2], -0.05, 1e-6);
                EXPECT_NEAR(point[1][0], -0.001, 1e-6);
                EXPECT_NEAR(point[1][1], 0.f, 1e-6);
                EXPECT_NEAR(point[1][2], 0.f, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, Cylinder3D)
            {
                int key = 1;
                float *time = new float[4];
                Vec3<float> *point = new Vec3<float>[2];
                solidTest(key, time, point);
                EXPECT_NEAR(time[0], 2.f, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(time[2], 1.f, 1e-6);
                EXPECT_NEAR(time[3], -1.f, 1e-6);
                EXPECT_NEAR(point[0][0], 0.f, 1e-6);
                EXPECT_NEAR(point[0][1], 0.025, 1e-6);
                EXPECT_NEAR(point[0][2], 0.05, 1e-6);
                EXPECT_NEAR(point[1][0], 0.05, 1e-6);
                EXPECT_NEAR(point[1][1], 0.f, 1e-6);
                EXPECT_NEAR(point[1][2], 0.f, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, Pyramid3D)
            {
                int key = 2;
                float *time = new float[5];
                Vec3<float> *point = new Vec3<float>[2];
                solidTest(key, time, point);
                EXPECT_NEAR(time[0], 1.f, 1e-6);
                EXPECT_NEAR(time[1], -1.f, 1e-6);
                EXPECT_NEAR(time[2], -1.f, 1e-6);
                EXPECT_NEAR(time[3], -1.f, 1e-6);
                EXPECT_NEAR(time[4], 2.f, 1e-6);
                EXPECT_NEAR(point[0][0], -0.0005, 1e-6);
                EXPECT_NEAR(point[0][1], 0.02, 1e-6);
                EXPECT_NEAR(point[0][2], -0.1, 1e-6);
                EXPECT_NEAR(point[1][0], 0.f, 1e-6);
                EXPECT_NEAR(point[1][1], 0.01, 1e-6);
                EXPECT_NEAR(point[1][2], -0.04, 1e-6);
                delete [] time;
                delete [] point;
            }

            TEST(IntersectTest, Sphere3D)
            {
                int key = 3;
                float *time = new float[2];
                Vec3<float> *point = new Vec3<float>[2];
                solidTest(key, time, point);
                EXPECT_NEAR(time[0], 0.4, 1e-6);
                EXPECT_NEAR(time[1], 0.6, 1e-6);
                EXPECT_NEAR(point[0][0], -0.1, 1e-6);
                EXPECT_NEAR(point[0][1], 0.f, 1e-6);
                EXPECT_NEAR(point[0][2], 0.f, 1e-6);
                EXPECT_NEAR(point[1][0], 0.1, 1e-6);
                EXPECT_NEAR(point[1][1], 0.f, 1e-6);
                EXPECT_NEAR(point[1][2], 0.f, 1e-6);
                delete [] time;
                delete [] point;
            }

        }

    }

}
