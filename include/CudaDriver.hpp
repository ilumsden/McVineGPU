#ifndef _CUDA_DRIVER_H_
#define _CUDE_DRIVER_H_

#include <memory>
#include <vector>

#include "Box.hpp"
#include "Ray.hpp"

class CudaDriver
{
    public:
        void operator()(std::shared_ptr<Box> b, std::vector< std::shared_ptr<Ray> > &rays);
    private:
        void handleRectIntersect(const std::shared_ptr<const Box> b, 
                                 const std::vector< std::shared_ptr<Ray> > &rays,
                                 std::vector<float> &host_time);
        void findScatteringSites(const std::shared_ptr<const Box> b,
                                 const std::vector< std::shared_ptr<Ray> > &rays,
                                 const std::vector<float> &int_times, std::vector<float> &sites);
};

#endif
