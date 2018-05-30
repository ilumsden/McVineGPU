#ifndef ABSTRACT_SHAPE_HPP
#define ABSTRACT_SHAPE_HPP

#include <memory>
#include <string>
#include <vector>

#include "Ray.hpp"

struct AbstractShape
{
    AbstractShape() { type = "Shape"; }

    virtual ~AbstractShape() { ; }

    //virtual void accept(UnaryVisitor &v) = 0;

    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) = 0;

    std::string type;
};

#endif
