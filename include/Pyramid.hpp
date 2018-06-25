#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include "AbstractShape.hpp"

/* The struct defining the Pyramid primitive.
 * It inherits from AbstractShape to ensure it contains the
 * correct signature for the intersect function.
 */
struct Pyramid : public AbstractShape
{
    // Default Constructor
    Pyramid() { type = "Pyramid"; }

    /* "Explicit" Constructor
     * This function takes three doubles and sets the base dimensions
     * and height with them.
     */
    Pyramid(double X, double Y, double h)
    {
        edgeX = X; edgeY = Y; height = h;
        type = "Pyramid";
    }

    ~Pyramid() { ; }

    /* The function that handles the calculation of the intersection points
     * and times between the Pyramid object and the neutrons represented
     * by d_origins and d_vel.
     */
    virtual void intersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, 
                           std::vector< Vec3<float> > &int_coords) override;

    /* These members store the Pyramid's base dimensions (in X and Y
     * components) and height.
     */
    double edgeX, edgeY, height;
};

#endif
