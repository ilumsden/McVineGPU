#ifndef ISO_SCATTER_HPP
#define ISO_SCATTER_HPP

#include "AbstractScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace scatter
        {

            class IsotropicScatterer : public AbstractScatterer
            {
                public:
                    IsotropicScatterer(std::shared_ptr<Beam> b,
                                       std::shared_ptr<AbstractShape> s)
                    : AbstractScatterer(b, s), type(0) { ; }

                    ~IsotropicScatterer() { ; }

                    virtual void scatter() override;

                protected:

                    virtual void findScatteringVels() override;
            };

        }

    }

}

#endif
