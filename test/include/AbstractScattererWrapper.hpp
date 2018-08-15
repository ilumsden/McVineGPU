#ifndef ABS_SCATTER_TEST_HPP
#define ABS_SCATTER_TEST_HPP

#include "AbstractScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using AbstractScatterer = mcvine::gpu::scatter::AbstractScatterer;

            class AbsScattererWrapper : public AbstractScatterer
            {
                public:
                    AbsScattererWrapper(std::shared_ptr<Beam> b,
                                        std::shared_ptr<AbstractShape> s)
                    : AbstractScatterer(b, s) { ; }
                    
                    virtual void scatter() override { ; }

                    void testHandleExtInt(std::vector<float> &int_times,
                                          std::vector< Vec3<float> > &int_coords)
                    {
                        handleExteriorIntersect(int_times, int_coords);
                    }

                    void testScatterSites(const std::vector<float> &int_times,
                                          const std::vector< Vec3<float> > &int_coords)
                    {
                        findScatteringSites(int_times, int_coords);
                    }

                    void testHandleIntIntersect()
                    {
                        handleInteriorIntersect();
                    }

                protected:

                    virtual void findScatteringVels() override { ; }
            };

        }

    }

}

#endif
