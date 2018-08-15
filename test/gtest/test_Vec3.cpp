#include <cmath>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "Vec3.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            class Vec3Test : public ::testing::Test
            {
                protected:
                    void SetUp() override
                    {
                        v0_ = Vec3<float>(3, 4, 5);
                        v1_ = Vec3<float>(5, 12, 13);
                    }

                    Vec3<float> v0_;
                    Vec3<float> v1_;
            };

            TEST_F(Vec3Test, ExplicitConstructor)
            {
                Vec3<float> test(7, 24, 25);
                EXPECT_FLOAT_EQ(test[0], 7.f);
                EXPECT_FLOAT_EQ(test[1], 24.f);
                EXPECT_FLOAT_EQ(test[2], 25.f);
            }

            TEST_F(Vec3Test, CopyConstructor)
            {
                Vec3<float> test(v0_);
                EXPECT_FLOAT_EQ(test[0], v0_[0]);
                EXPECT_FLOAT_EQ(test[1], v0_[1]);
                EXPECT_FLOAT_EQ(test[2], v0_[2]);
            }
            
            TEST_F(Vec3Test, AssignmentOp)
            {
                v1_ = v0_;
                EXPECT_FLOAT_EQ(v1_[0], v0_[0]);
                EXPECT_FLOAT_EQ(v1_[1], v0_[1]);
                EXPECT_FLOAT_EQ(v1_[2], v0_[2]);
            }

            TEST_F(Vec3Test, Size)
            {
                EXPECT_EQ((int)(v0_.size()), 3);
            }

            TEST_F(Vec3Test, getX)
            {
                EXPECT_FLOAT_EQ(v0_.getX(), 3.f);
            }

            TEST_F(Vec3Test, getY)
            {
                EXPECT_FLOAT_EQ(v0_.getY(), 4.f);
            }

            TEST_F(Vec3Test, getZ)
            {
                EXPECT_FLOAT_EQ(v0_.getZ(), 5.f);
            }

            TEST_F(Vec3Test, setX)
            {
                v0_.setX(31);
                EXPECT_FLOAT_EQ(v0_.getX(), 31.f);
            }

            TEST_F(Vec3Test, setY)
            {
                v0_.setY(480);
                EXPECT_FLOAT_EQ(v0_.getY(), 480.f);
            }

            TEST_F(Vec3Test, setZ)
            {
                v0_.setZ(481);
                EXPECT_FLOAT_EQ(v0_.getZ(), 481.f);
            } 

            TEST_F(Vec3Test, normalize)
            {
                float norm = sqrt(pow(3.f, 2) + pow(4.f, 2) + pow(5.f, 2));
                v0_.normalize();
                EXPECT_FLOAT_EQ(v0_[0], (3.f/norm));
                EXPECT_FLOAT_EQ(v0_[1], (4.f/norm));
                EXPECT_FLOAT_EQ(v0_[2], (5.f/norm));
            }

            TEST_F(Vec3Test, length)
            {
                EXPECT_FLOAT_EQ(v0_.length(), sqrt(pow(3.f, 2) + pow(4.f, 2) + pow(5.f, 2)));
            }

            TEST_F(Vec3Test, AddOp)
            {
                Vec3<float> test = v0_ + v1_;
                EXPECT_FLOAT_EQ(test[0], 8.f);
                EXPECT_FLOAT_EQ(test[1], 16.f);
                EXPECT_FLOAT_EQ(test[2], 18.f);
            }

            TEST_F(Vec3Test, AddEqOp)
            {
                v0_ += v1_;
                EXPECT_FLOAT_EQ(v0_[0], 8.f);
                EXPECT_FLOAT_EQ(v0_[1], 16.f);
                EXPECT_FLOAT_EQ(v0_[2], 18.f);
            }

            TEST_F(Vec3Test, SubOp)
            {
                Vec3<float> test = v1_ - v0_;
                EXPECT_FLOAT_EQ(test[0], 2.f);
                EXPECT_FLOAT_EQ(test[1], 8.f);
                EXPECT_FLOAT_EQ(test[2], 8.f);
            }

            TEST_F(Vec3Test, NegateOp)
            {
                v1_ = -v0_;
                EXPECT_FLOAT_EQ(v1_[0], -3.f);
                EXPECT_FLOAT_EQ(v1_[1], -4.f);
                EXPECT_FLOAT_EQ(v1_[2], -5.f);
            }

            TEST_F(Vec3Test, SubEqOp)
            {
                v1_ -= v0_;
                EXPECT_FLOAT_EQ(v1_[0], 2.f);
                EXPECT_FLOAT_EQ(v1_[1], 8.f);
                EXPECT_FLOAT_EQ(v1_[2], 8.f);
            }

            TEST_F(Vec3Test, ConstMulOp)
            {
                Vec3<float> test = v0_*5;
                EXPECT_FLOAT_EQ(test[0], 15.f);
                EXPECT_FLOAT_EQ(test[1], 20.f);
                EXPECT_FLOAT_EQ(test[2], 25.f);
            }

            TEST_F(Vec3Test, CrossProductOp)
            {
                Vec3<float> test = v0_ * v1_;
                EXPECT_FLOAT_EQ(test[0], -8.f);
                EXPECT_FLOAT_EQ(test[1], -14.f);
                EXPECT_FLOAT_EQ(test[2], 16.f);
            }

            TEST_F(Vec3Test, ConstMulEqOp)
            {
                v0_ *= 5;
                EXPECT_FLOAT_EQ(v0_[0], 15.f);
                EXPECT_FLOAT_EQ(v0_[1], 20.f);
                EXPECT_FLOAT_EQ(v0_[2], 25.f);
            }

            TEST_F(Vec3Test, CrossProductEqOp)
            {
                v0_ *= v1_;
                EXPECT_FLOAT_EQ(v0_[0], -8.f);
                EXPECT_FLOAT_EQ(v0_[1], -14.f);
                EXPECT_FLOAT_EQ(v0_[2], 16.f);
            }

            TEST_F(Vec3Test, ConstDivideOp)
            {
                Vec3<float> test = v0_ / 2;
                EXPECT_FLOAT_EQ(test[0], (3.f/2.f));
                EXPECT_FLOAT_EQ(test[1], 2.f);
                EXPECT_FLOAT_EQ(test[2], (5.f/2.f));
            }

            TEST_F(Vec3Test, ConstDivideEqOp)
            {
                v0_ /= 2;
                EXPECT_FLOAT_EQ(v0_[0], (3.f/2.f));
                EXPECT_FLOAT_EQ(v0_[1], 2.f);
                EXPECT_FLOAT_EQ(v0_[2], (5.f/2.f));
            }

            TEST_F(Vec3Test, DotProductOp)
            {
                float test = v0_ | v1_;
                EXPECT_FLOAT_EQ(test, 128.f);
            }

            TEST_F(Vec3Test, EqualsComp)
            {
                EXPECT_FALSE(v0_ == v1_);
                EXPECT_TRUE(v0_ == Vec3<float>(3, 4, 5));
            }

            TEST_F(Vec3Test, NEqualsComp)
            {
                EXPECT_TRUE(v0_ != v1_);
                EXPECT_FALSE(v0_ != Vec3<float>(3, 4, 5));
            }

            TEST_F(Vec3Test, ReadAccessOp)
            {
                EXPECT_FLOAT_EQ(v0_[0], 3.f);
                EXPECT_FLOAT_EQ(v0_[1], 4.f);
                EXPECT_FLOAT_EQ(v0_[2], 5.f);
            }

            TEST_F(Vec3Test, WriteAccessOp)
            {
                v0_[0] = 21.f;
                EXPECT_FLOAT_EQ(v0_[0], 21.f);
                v0_[1] = 220.f;
                EXPECT_FLOAT_EQ(v0_[1], 220.f);
                v0_[2] = 221.f;
                EXPECT_FLOAT_EQ(v0_[2], 221.f);
            }

        }

    }

}
