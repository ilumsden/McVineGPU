#ifndef _CUDA_DRIVER_H_
#define _CUDE_DRIVER_H_

#include <memory>
#include <vector>

#include "Box.hpp"
#include "Ray.hpp"

class CudaDriver
{
    public:
        void operator()(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > rays);
    private:
        void cudaHandler(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > rays);
};

#endif
