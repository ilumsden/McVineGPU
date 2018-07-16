#ifndef CYLINDER_HPP
#define CYLINDER_HPP

#include <cfloat>

#include "AbstractShape.hpp"

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
     * represented by d_origins and d_vel when the neutrons start
     * outside the Cylinder.
     */
    virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;

    /* The function that handles the calculation of the intersection
     * points and times between the Cylinder object and the neutrons
     * represented by d_origins and d_vel when the neutrons start
     * inside the Cylinder.
     */
    virtual void interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;

    // These members store the Cylinder's radius and height.
    double radius, height;
};

#endif
