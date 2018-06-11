#ifndef BOX_HPP
#define BOX_HPP

//#include "WhateverTheVisitorFileIsCalled"
#include "AbstractShape.hpp"

/* The struct defining the Box primitive.
 * It inherits from AbstractShape to ensure it contains the
 * correct signature for the intersect function.
 */
struct Box : public AbstractShape
{
    // Default Constructor
    Box() { type = "Box"; }

    /* "Explicit" Constructor.
     * This function takes three doubles and sets the side lengths
     * with them.
     */
    Box(const double a, const double b, const double c)
    {
        X=a; Y=b; Z=c;
        type = "Box";
    }

    ~Box() { ; }

    // See the corresponding comment in AbstractShape.hpp
    //virtual void accept(UnaryVisitor &v) override;
    
    /* The function that handles the calculation of the intersection
     * points and times between the Box object and the neutrons represented
     * by d_rx, d_ry, d_rz, d_vx, d_vy, and d_vz.
     */
    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) override;

    // These members store the Box's side lengths in the X, Y, and Z directions.
    double X, Y, Z;
};

#endif
