#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "AbstractShape.hpp"

struct Sphere : public AbstractShape
{
    Sphere() { type = "Sphere"; }

    Sphere(const double r) { radius = r; type = "Sphere"; }

    ~Sphere() { ; }

    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) override;

    double radius;
};

#endif
