#ifndef ISO_SCATTER_WRAPPER_HPP
#define ISO_SCATTER_WRAPPER_HPP

#include "Box.hpp"
#include "IsotropicScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            using IsotropicScatterer = mcvine::gpu::scatter::IsotropicScatterer;

            class IsoScattererWrapper : public IsotropicScatterer
            {
                public:
                    IsoScattererWrapper(std::shared_ptr<Beam> b,
                                        std::shared_ptr<AbstractShape> s)
                    : IsotropicScatterer(b, s) { ; }

                    void testFindScatterVels()
                    {
                        findScatteringVels();
                    }
            };

        }

    }

}

#endif
