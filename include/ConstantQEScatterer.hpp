#ifndef CONST_QE_SCATTER_HPP
#define CONST_QE_SCATTER_HPP

#include "AbstractScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace scatter
        {

            class ConstantQEScatterer : public AbstractScatterer
            {
                public:
                    ConstantQEScatterer(std::shared_ptr<Beam> b,
                                        std::shared_ptr<AbstractShape> s,
                                        const float kQ, const float kE)
                    : AbstractScatterer(b, s), Q(kQ), E(kE) { type = 1; }

                    virtual ~ConstantQEScatterer() { ; }

                    virtual void scatter() override;

                protected:

                    virtual void findScatteringVels() override;

                private:

                    const float Q, E;
            };

        }

    }

}

#endif
