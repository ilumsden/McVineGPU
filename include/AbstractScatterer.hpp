#ifndef ABS_SCATTER_HPP
#define ABS_SCATTER_HPP

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>
#include <fstream>

#if defined(RANDTEST)
#include <cmath>
#endif

#include "Beam.hpp"
#include "Box.hpp"
#include "Cylinder.hpp"
#include "Error.hpp"
#include "Pyramid.hpp"
#include "ScatteringKernels.hpp"
#include "Sphere.hpp"
#include "SystemVars.hpp"

typedef mcvine::gpu::composite::AbstractShape AbstractShape;

namespace mcvine
{

    namespace gpu
    {

        namespace scatter
        {

            class AbstractScatterer
            {
                public:
                    AbstractScatterer(std::shared_ptr<Beam> b, 
                                      std::shared_ptr<AbstractShape> s)
                    : beam(b), shape(s), type(-1) { ; }

                    virtual ~AbstractScatterer() { ; }

                    virtual void scatter() = 0;

                protected:

                    void handleExteriorIntersect(std::vector<float> &int_times,
                                                 std::vector< Vec3<float> > &int_coords);

                    void findScatteringSites(const std::vector<float> &int_times,
                                             const std::vector< Vec3<float> > &int_coords);

                    virtual void findScatteringVels() = 0;

                    void handleInteriorIntersect();

                    std::shared_ptr<Beam> beam;

                    std::shared_ptr<AbstractShape> shape;

                    int type;
            };

        }

    }

}

#endif
