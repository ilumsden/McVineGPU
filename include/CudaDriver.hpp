#ifndef _CUDA_DRIVER_H_
#define _CUDE_DRIVER_H_

#include <memory>
#include <vector>

#include "Box.hpp"
#include "Ray.hpp"

class CudaDriver
{
    public:
        void operator()(std::shared_ptr<Box> &b, const std::vector< std::shared_ptr<Ray> > &rays);
    private:
        void handleRectIntersect(std::shared_ptr<Box> &b, 
                                 const std::vector< std::shared_ptr<Ray> > &rays,
                                 std::vector<float> &host_time, 
                                 std::vector<float> &int_coords);
        void findScatteringSites(std::shared_ptr<Box> &b,
                                 const std::vector< std::shared_ptr<Ray> > &rays,
                                 const std::vector<float> &int_times, 
                                 const std::vector<float> &int_coords,
                                 std::vector<float> &sites);

        //void allocInitialData(const std::vector< std::shared_ptr<Ray> > &rays);
        //void freeInitialData();

        //float *rx, *ry, *rz, *vx, *vy, *vz;
        //int N;
};

#endif
