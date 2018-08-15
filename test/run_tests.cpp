#include "gtest/gtest.h"
#include "SystemVars.hpp"

float mcvine::gpu::atten;

int main(int argc, char** argv)
{
    mcvine::gpu::atten = 0.01;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
