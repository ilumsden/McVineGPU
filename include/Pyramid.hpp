#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include "AbstractShape.hpp"

struct Pyramid : public AbstractShape
{
    Pyramid() { type = "Pyramid"; }

    Pyramid(double X, double Y, double h)
    {
        edgeX = X; edgeY = Y; height = h;
        type = "Pyramid";
    }

    ~Pyramid() { ; }

    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) override;

    double edgeX, edgeY, height;
};

#endif
