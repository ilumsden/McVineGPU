#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include <cfloat>

#include "AbstractShape.hpp"
#include "Kernels.hpp"

/* The struct defining the Cylinder primitive.
 * It inherits from AbstractShape to ensure it contains the
 * correct signature for the intersect function.
 */
struct Cylinder : public AbstractShape
{
    // Default Constructor
    Cylinder() { type = "Cylinder"; }
    
    /* "Explicit" Constructor
     * This function takes two doubles and uses them to set
     * the Cylinder's radius and height.
     */
    Cylinder(const double kRadius, const double kHeight)
    {
        radius = kRadius;
        height = kHeight;
        type = "Cylinder";
    }

    ~Cylinder() { ; }

    /* The function that handles the calculation of the intersection
     * points and times between the Cylinder object and the neutrons
     * represented by d_rx, d_ry, d_rz, d_vx, d_vy, and d_vz.
     */
    virtual void intersect(//float *d_rx, float *d_ry, float *d_rz,
                           //float *d_vx, float *d_vy, float *d_vz,
                           Vec3<float> *d_origins, Vec3<float> *d_vel,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, 
                           std::vector< Vec3<float> > &int_coords) override;//std::vector<float> &int_coords) override;

    // These members store the Cylinder's radius and height.
    double radius, height;
};

#endif
