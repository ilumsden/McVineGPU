#include <cmath>
#include <cuda_runtime.h>

#include "gtest/gtest.h"

#include "Box.hpp"
#include "Cylinder.hpp"
#include "Pyramid.hpp"
#include "Sphere.hpp"

#include "ShapeWrapper.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using Box = mcvine::gpu::composite::Box;
            using Cylinder = mcvine::gpu::composite::Cylinder;
            using Pyramid = mcvine::gpu::composite::Pyramid;
            using Sphere = mcvine::gpu::composite::Sphere;

            TEST(ShapeTest, BoxExtIntersect)
            {
                Box box(0.002, 0.05, 0.1);
                Vec3<float> orig(-0.002, 0, 0.05);
                Vec3<float> vel(0.001, 0, -0.05);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(box, 0, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 2);
                EXPECT_EQ((int)(coords.size()), 2);
                EXPECT_NEAR(times[0], 1.f, 1e-6);
                EXPECT_NEAR(times[1], 2.f, 1e-6);
                EXPECT_NEAR(coords[0][0], -0.001, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][0], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][2], -0.05, 1e-6);
            }

            TEST(ShapeTest, BoxIntIntersect)
            {
                Box box(0.002, 0.05, 0.1);
                Vec3<float> orig(0, 0, 0);
                Vec3<float> vel(0, 0, 1);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(box, 1, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 1);
                EXPECT_EQ((int)(coords.size()), 1);
                EXPECT_NEAR(times[0], 0.05, 1e-6);
                EXPECT_NEAR(coords[0][0], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], 0.05, 1e-6);
            }

            TEST(ShapeTest, CylExtIntersect)
            {
                Cylinder cyl(0.05, 0.1);
                Vec3<float> orig(0.1, -0.025, -0.05);
                Vec3<float> vel(-0.05, 0.025, 0.05);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(cyl, 0, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 2);
                EXPECT_EQ((int)(coords.size()), 2);
                EXPECT_NEAR(times[0], 1.f, 1e-6);
                EXPECT_NEAR(times[1], 2.f, 1e-6);
                EXPECT_NEAR(coords[0][0], 0.05, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][0], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][1], 0.025, 1e-6);
                EXPECT_NEAR(coords[1][2], 0.05, 1e-6);
            }

            TEST(ShapeTest, CylIntIntersect)
            {
                Cylinder cyl(0.05, 0.1);
                Vec3<float> orig(0, 0, 0);
                Vec3<float> vel(1, 0, 0);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(cyl, 1, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 1);
                EXPECT_EQ((int)(coords.size()), 1);
                EXPECT_NEAR(times[0], 0.05, 1e-6);
                EXPECT_NEAR(coords[0][0], 0.05, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], 0.f, 1e-6);
            }

            TEST(ShapeTest, PyrExtIntersect)
            {
                Pyramid pyr(0.002, 0.05, 0.1);
                Vec3<float> orig(-0.001, 0.03, -0.16);
                Vec3<float> vel(0.0005, -0.01, 0.06);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(pyr, 0, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 2);
                EXPECT_EQ((int)(coords.size()), 2);
                EXPECT_NEAR(times[0], 1.f, 1e-6);
                EXPECT_NEAR(times[1], 2.f, 1e-6);
                EXPECT_NEAR(coords[0][0], -0.0005, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.02, 1e-6);
                EXPECT_NEAR(coords[0][2], -0.1, 1e-6);
                EXPECT_NEAR(coords[1][0], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][1], 0.01, 1e-6);
                EXPECT_NEAR(coords[1][2], -0.04, 1e-6);
            }

            TEST(ShapeTest, PyrIntIntersect)
            {
                Pyramid pyr(0.002, 0.05, 0.1);
                Vec3<float> orig(0, 0, 0);
                Vec3<float> vel(0, 0, -1);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(pyr, 1, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 1);
                EXPECT_EQ((int)(coords.size()), 1);
                EXPECT_NEAR(times[0], 0.1, 1e-6);
                EXPECT_NEAR(coords[0][0], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], -0.1, 1e-6);
            }

            TEST(ShapeTest, SphExtIntersect)
            {
                Sphere sph(0.1);
                Vec3<float> orig(-0.5, 0, 0);
                Vec3<float> vel(1, 0, 0);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(sph, 0, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 2);
                EXPECT_EQ((int)(coords.size()), 2);
                EXPECT_NEAR(times[0], 0.4, 1e-6);
                EXPECT_NEAR(times[1], 0.6, 1e-6);
                EXPECT_NEAR(coords[0][0], -0.1, 1e-6);
                EXPECT_NEAR(coords[0][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[0][2], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][0], 0.1, 1e-6);
                EXPECT_NEAR(coords[1][1], 0.f, 1e-6);
                EXPECT_NEAR(coords[1][2], 0.f, 1e-6);
            }

            TEST(ShapeTest, SphIntIntersect)
            {
                Sphere sph(0.1);
                Vec3<float> orig(0, 0, 0);
                Vec3<float> vel(1, 1, 1);
                std::vector<float> times;
                std::vector< Vec3<float> > coords;
                runIntersection(sph, 1, orig, vel, times, coords);
                EXPECT_EQ((int)(times.size()), 1);
                EXPECT_EQ((int)(coords.size()), 1);
                EXPECT_NEAR(times[0], 0.1/sqrt(3), 1e-6);
                EXPECT_NEAR(coords[0][0], 0.1/sqrt(3), 1e-6);
                EXPECT_NEAR(coords[0][1], 0.1/sqrt(3), 1e-6);
                EXPECT_NEAR(coords[0][2], 0.1/sqrt(3), 1e-6);
            }

        }

    }

}
