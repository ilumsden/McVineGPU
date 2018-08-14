#ifndef SHAPE_WRAPPER_HPP
#define SHAPE_WRAPPER_HPP

#include <vector>

#include "AbstractShape.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using AbstractShape = mcvine::gpu::composite::AbstractShape;

            void runIntersection(AbstractShape &shape, const int key,
                                 Vec3<float> &orig, Vec3<float> &vel,
                                 std::vector<float> &int_times,
                                 std::vector< Vec3<float> > &int_coords);

        }

    }

}

#endif
