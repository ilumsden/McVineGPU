#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include <cfloat>

#include "AbstractShape.hpp"
#include "Kernels.hpp"

struct Cylinder : public AbstractShape
{
    Cylinder() { type = "Cylinder"; }
    
    Cylinder(const double kRadius, const double kHeight)
    {
        radius = kRadius;
        height = kHeight;
        type = "Cylinder";
    }

    ~Cylinder() { ; }

    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) override;

    double radius, height;
};

#endif
