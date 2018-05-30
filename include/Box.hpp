#ifndef BOX_HPP
#define BOX_HPP

//#include "WhateverTheVisitorFileIsCalled"
#include "AbstractShape.hpp"

struct Box : public AbstractShape
{
    Box() { type = "Box"; }

    Box(const double a, const double b, const double c)
    {
        X=a; Y=b; Z=c;
        type = "Box";
    }

    ~Box() { ; }

    //virtual void accept(UnaryVisitor &v) override;
    
    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) override;

    double X, Y, Z;
};

#endif
