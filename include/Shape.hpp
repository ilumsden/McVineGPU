#ifndef SHAPE_HPP
#define SHAPE_HPP

#include <memory>
#include <string>
#include <vector>

#include "Ray.hpp"

class Shape
{
    public:
        Shape() { type = "Shape"; }
        virtual ~Shape() { ; }
        //virtual void accept(UnaryVisitor &v) = 0;
        virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                               float *d_vx, float *d_vy, float *d_vz,
                               const int N, const int blockSize, const int numBlocks,
                               std::vector<float> &int_times, std::vector<float> &int_coords) = 0;
    protected:
        std::string type;
};

#endif
