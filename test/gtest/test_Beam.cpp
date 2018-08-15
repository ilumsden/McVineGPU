#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include "Beam.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            TEST(BeamTest, ConstructorTestIdeal)
            {
                std::vector< std::shared_ptr<Ray> > rays;
                rays.push_back(std::make_shared<Ray>(0, 0, 0, 1, 1, 1));
                Beam beam(rays, 10, 10);
                for (int i = 0; i < 10; i++)
                {
                    EXPECT_NEAR(beam.origins[i][0], 0.f, 1e-6);
                    EXPECT_NEAR(beam.origins[i][1], 0.f, 1e-6);
                    EXPECT_NEAR(beam.origins[i][2], 0.f, 1e-6);
                    EXPECT_NEAR(beam.vel[i][0], 1.f, 1e-6);
                    EXPECT_NEAR(beam.vel[i][1], 1.f, 1e-6);
                    EXPECT_NEAR(beam.vel[i][2], 1.f, 1e-6);
                    EXPECT_NEAR(beam.times[i], 0.f, 1e-6);
                    EXPECT_NEAR(beam.probs[i], 1.f, 1e-6);
                }
                EXPECT_EQ(beam.numBlocks, 1);
                EXPECT_EQ(beam.blockSize, 10);
                EXPECT_EQ(beam.rayptr, &rays);
            }

            TEST(BeamTest, ConstructorTestReal)
            {
                std::vector< std::shared_ptr<Ray> > rays;
                for (int i = 0; i < 5; i++)
                {
                    rays.push_back(std::make_shared<Ray>(0, 0, 0, 1, 1, 1));
                }
                for (int i = 5; i < 10; i++)
                {
                    rays.push_back(std::make_shared<Ray>(1, 1, 1, 2, 2, 2));
                }
                Beam beam(rays, -1, 10);
                for (int i = 0; i < 10; i++)
                {
                    if (i < 5)
                    {
                        EXPECT_NEAR(beam.origins[i][0], 0.f, 1e-6);
                        EXPECT_NEAR(beam.origins[i][1], 0.f, 1e-6);
                        EXPECT_NEAR(beam.origins[i][2], 0.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][0], 1.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][1], 1.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][2], 1.f, 1e-6);
                        EXPECT_NEAR(beam.times[i], 0.f, 1e-6);
                        EXPECT_NEAR(beam.probs[i], 1.f, 1e-6);
                    }
                    else
                    {
                        EXPECT_NEAR(beam.origins[i][0], 1.f, 1e-6);
                        EXPECT_NEAR(beam.origins[i][1], 1.f, 1e-6);
                        EXPECT_NEAR(beam.origins[i][2], 1.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][0], 2.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][1], 2.f, 1e-6);
                        EXPECT_NEAR(beam.vel[i][2], 2.f, 1e-6);
                        EXPECT_NEAR(beam.times[i], 0.f, 1e-6);
                        EXPECT_NEAR(beam.probs[i], 1.f, 1e-6);
                    }
                }
                EXPECT_EQ(beam.numBlocks, 1);
                EXPECT_EQ(beam.blockSize, 10);
                EXPECT_EQ(beam.rayptr, &rays);
            }

            TEST(BeamTest, UpdateRayTestIdeal)
            {
                std::vector< std::shared_ptr<Ray> > rays;
                rays.push_back(std::make_shared<Ray>(0, 0, 0, 1, 1, 1));
                Beam beam(rays, 10, 10);
                ASSERT_NE(beam.rayptr, nullptr);
                for (int i = 0; i < 10; i++)
                {
                    beam.origins[i] = Vec3<float>(1, 1, 1);
                    beam.vel[i] = Vec3<float>(2, 2, 2);
                    beam.times[i] = 0.5;
                    beam.probs[i] = 0.5;
                }
                beam.updateRays();
                for (auto r : rays)
                {
                    EXPECT_NEAR(r->origin[0], 1.f, 1e-6);
                    EXPECT_NEAR(r->origin[1], 1.f, 1e-6);
                    EXPECT_NEAR(r->origin[2], 1.f, 1e-6);
                    EXPECT_NEAR(r->vel[0], 2.f, 1e-6);
                    EXPECT_NEAR(r->vel[1], 2.f, 1e-6);
                    EXPECT_NEAR(r->vel[2], 2.f, 1e-6);
                    EXPECT_NEAR(r->t, 0.5, 1e-6);
                    EXPECT_NEAR(r->prob, 0.5, 1e-6);
                }
            }

            TEST(BeamTest, UpdateRayTestReal)
            {
                std::vector< std::shared_ptr<Ray> > rays;
                for (int i = 0; i < 5; i++)
                {
                    rays.push_back(std::make_shared<Ray>(0, 0, 0, 1, 1, 1));
                }
                for (int i = 5; i < 10; i++)
                {
                    rays.push_back(std::make_shared<Ray>(1, 1, 1, 2, 2, 2));
                }
                Beam beam(rays, -1, 10);
                ASSERT_NE(beam.rayptr, nullptr);
                for (int i = 0; i < 10; i++)
                {
                    beam.origins[i] = Vec3<float>(2, 2, 2);
                    beam.vel[i] = Vec3<float>(3, 3, 3);
                    beam.times[i] = 0.5;
                    beam.probs[i] = 0.5;
                }
                beam.updateRays();
                for (auto r : rays)
                {
                    EXPECT_NEAR(r->origin[0], 2.f, 1e-6);
                    EXPECT_NEAR(r->origin[1], 2.f, 1e-6);
                    EXPECT_NEAR(r->origin[2], 2.f, 1e-6);
                    EXPECT_NEAR(r->vel[0], 3.f, 1e-6);
                    EXPECT_NEAR(r->vel[1], 3.f, 1e-6);
                    EXPECT_NEAR(r->vel[2], 3.f, 1e-6);
                    EXPECT_NEAR(r->t, 0.5, 1e-6);
                    EXPECT_NEAR(r->prob, 0.5, 1e-6);
                }
            }

        }

    }

}
