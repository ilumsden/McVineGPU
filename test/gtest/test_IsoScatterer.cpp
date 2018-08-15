#include "gtest/gtest.h"
#include <cmath>
#include <cuda_runtime.h>

#include "Box.hpp"
#include "IsoScattererWrapper.hpp"
#include "SystemVars.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using Box = mcvine::gpu::composite::Box;

            bool onBoxSurface(const std::shared_ptr<Box> box, const Vec3<float> &orig)
            {
                if ((abs(orig[0] - (box->data[0]/2)) < 1e-6) &&
                    (abs(orig[1]) < (box->data[1]/2)) &&
                    (abs(orig[2]) < (box->data[2]/2)))
                {
                    return true;
                }
                else if ((abs(orig[0]) < (box->data[0]/2)) &&
                         (abs(orig[1] - (box->data[1]/2)) < 1e-6) &&
                         (abs(orig[2]) < (box->data[2]/2)))
                {
                    return true;
                }
                else if ((abs(orig[0]) < (box->data[0]/2)) &&
                         (abs(orig[1]) < (box->data[1]/2)) &&
                         (abs(orig[2] - (box->data[2]/2)) < 1e-6))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }

            TEST(IsoScattererTest, ScatteringVelTest)
            {
                std::shared_ptr<Box> box = std::make_shared<Box>(0.002, 0.05, 0.1);
                std::vector< std::shared_ptr<Ray> > rays;
                // Initial Position assumed to be (-2, 0, 0)
                rays.push_back(std::make_shared<Ray>(0, 0, 0, 1, 0, 0, 2, exp(-(0.001/atten))));
                std::shared_ptr<Beam> beam = std::make_shared<Beam>(rays, 10, 10);
                IsoScattererWrapper scatter(beam, box);
                scatter.testFindScatterVels();
                for (int i = 0; i < 10; i++)
                {
                    EXPECT_NEAR(beam->vel[i].length(), 1.f, 1e-6);
                    EXPECT_TRUE(beam->vel[i] != Vec3<float>(1, 0, 0));
                }
            }

            TEST(IsoScattererTest, ScatterTest)
            {
                std::shared_ptr<Box> box = std::make_shared<Box>(0.002, 0.05, 0.1);
                std::vector< std::shared_ptr<Ray> > rays;
                rays.push_back(std::make_shared<Ray>(-2, 0, 0, 1, 0, 0));
                std::shared_ptr<Beam> beam = std::make_shared<Beam>(rays, 10, 10);
                IsoScattererWrapper scatter(beam, box);
                scatter.scatter();
                Vec3<float> init_vel(1, 0, 0);
                for (int i = 0; i < 10; i++)
                {
                    Vec3<float> orig = beam->origins[i];
                    EXPECT_PRED2(onBoxSurface, box, orig);
                    EXPECT_NEAR(beam->vel[i].length(), init_vel.length(), 1e-6);
                    EXPECT_TRUE(beam->vel[i] != init_vel);
                    EXPECT_TRUE(beam->times[i] > 0);
                    EXPECT_TRUE(beam->probs[i] < 1);
                }
            }

        }

    }

}
