#ifndef MPI_RUNNER_HPP
#define MPI_RUNNER_HPP

#include <mpi.h>
//#include <boost/mpi.hpp>
//#include <boost/serialization/vector.hpp>
//#include <boost/serialization/shared_ptr.hpp>

#include <functional>
#include <memory>

#include "AbstractShape.hpp"

namespace mcvine
{

    namespace gpu
    {

        using AbstractShape = mcvine::gpu::composite::AbstractShape;

        void runMPI(int *argc, char ***argv, 
                    std::function<void(std::shared_ptr<AbstractShape>, const int)> simFunc,
                    std::shared_ptr<AbstractShape> &shape,
                    const int numNeutrons, const int blockSize);

    }

}

#endif
