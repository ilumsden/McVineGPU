#include "gtest/gtest.h"
#include <cmath>
#include <cuda_runtime.h>

#include "AbstractScattererWrapper.hpp"
#include "Beam.hpp"
#include "Box.hpp"
#include "ScatteringWrapper.hpp"
#include "SystemVars.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using Box = mcvine::gpu::composite::Box;

            class AbsScattererTest : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        box = std::make_shared<Box>(0.002, 0.05, 0.1);
                        std::vector< std::shared_ptr<Ray> > rays;
                        rays.push_back(std::make_shared<Ray>(-0.002, 0, 0, 0.001, 0, 0));
                        beam = std::make_shared<Beam>(rays, 10, 10);
                    }

                    std::shared_ptr<Box> box;
                    std::shared_ptr<Beam> beam;
            };

            TEST_F(AbsScattererTest, ExtIntersectTest)
            {
                AbsScattererWrapper scatter(beam, box);
                std::vector<float> int_times;
                std::vector< Vec3<float> > int_coords;
                scatter.testHandleExtInt(int_times, int_coords);
                for (int i = 0; i < 10; i++)
                {
                    EXPECT_NEAR(int_times[2*i], 1.f, 1e-6);
                    EXPECT_NEAR(int_times[2*i+1], 3.f, 1e-6);
                    EXPECT_NEAR(int_coords[2*i][0], -0.001, 1e-6);
                    EXPECT_NEAR(int_coords[2*i][1], 0.f, 1e-6);
                    EXPECT_NEAR(int_coords[2*i][2], 0.f, 1e-6);
                    EXPECT_NEAR(int_coords[2*i+1][0], 0.001, 1e-6);
                    EXPECT_NEAR(int_coords[2*i+1][1], 0.f, 1e-6);
                    EXPECT_NEAR(int_coords[2*i+1][2], 0.f, 1e-6);
                }
            }

            TEST_F(AbsScattererTest, ScatterSitesTest)
            {
                AbsScattererWrapper scatter(beam, box);
                std::vector<float> int_times;
                std::vector< Vec3<float> > int_coords;
                for (int i = 0; i < 10; i++)
                {
                    int_times.push_back(1);
                    int_times.push_back(3);
                    int_coords.push_back(Vec3<float>(-0.001, 0, 0));
                    int_coords.push_back(Vec3<float>(0.001, 0, 0));
                }
                scatter.testScatterSites(int_times, int_coords);
                for (int i = 0; i < 10; i++)
                {
                    EXPECT_PRED3(isBetween<float>, beam->origins[i][0], -0.001, 0.001);
                    EXPECT_NEAR(beam->origins[i][1], 0.f, 1e-6);
                    EXPECT_NEAR(beam->origins[i][2], 0.f, 1e-6);
                    EXPECT_PRED3(isBetween<float>, beam->times[i], 1.f, 3.f);
                    EXPECT_PRED3(isBetween<float>, beam->probs[i], exp(-(0.002/atten)), 1.f);
                }
            }

            TEST_F(AbsScattererTest, IntIntersectTest)
            {
                float init_prob = exp(-(0.001/atten));
                std::vector< std::shared_ptr<Ray> > rays;
                rays.push_back(std::make_shared<Ray>(0, 0, 0, 0, 0, 0.05, 2, init_prob));
                std::shared_ptr<Beam> beam2 = std::make_shared<Beam>(rays, 10, 10);
                AbsScattererWrapper scatter(beam2, box);
                scatter.testHandleIntIntersect();
                float final_prob = init_prob * exp(-(0.05/atten));
                for (int i = 0; i < 10; i++)
                {
                    EXPECT_NEAR(beam2->origins[i][0], 0.f, 1e-6);
                    EXPECT_NEAR(beam2->origins[i][1], 0.f, 1e-6);
                    EXPECT_NEAR(beam2->origins[i][2], 0.05, 1e-6);
                    EXPECT_NEAR(beam2->times[i], 3.f, 1e-6);
                    EXPECT_NEAR(beam2->probs[i], final_prob, 1e-6);
                }
            }

        }

    }

}
