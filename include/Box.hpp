#ifndef BOX_HPP
#define BOX_HPP

//#include "WhateverTheVisitorFileIsCalled"
#include "Shape.hpp"

class Box : public Shape
{
    public:
        Box() { ; }
        Box(const double a, const double b, const double c);
        Box(const double amin, const double bmin, const double cmin,
            const double amax, const double bmax, const double cmax);
        //virtual void accept(UnaryVisitor &v) override;
        virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                               float *d_vx, float *d_vy, float *d_vz,
                               const int N, const int blockSize, const int numBlocks,
                               std::vector<float> &int_times, std::vector<float> &int_coords) override;
        double getX() const { return x; }
        double getY() const { return y; }
        double getZ() const { return z; }
        double getXmin() const { return xmin; }
        double getXmax() const { return xmax; }
        double getYmin() const { return ymin; }
        double getYmax() const { return ymax; }
        double getZmin() const { return zmin; }
        double getZmax() const { return zmax; }
    protected:
        double x, y, z;
        double xmin, xmax, ymin, ymax, zmin, zmax;
};

#endif
