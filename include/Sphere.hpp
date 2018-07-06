#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "AbstractShape.hpp"

/* The struct defining the Sphere primitive.
 * It inherits from AbstractShape to ensure it contains the
 * correct signature for the intersect function.
 */
struct Sphere : public AbstractShape
{
    // Default Constructor
    Sphere() { type = "Sphere"; }

    /* "Explicit" Constructor
     * This function takes a double which is used to set the Sphere's
     * radius.
     */
    Sphere(const double r) { radius = r; type = "Sphere"; }

    ~Sphere() { ; }

    /* The function that handles the calculation of the intersection
     * points and times between the Sphere object and the neutrons
     * represented by d_origins and d_vel when the neutrons start
     * outside the Sphere.
     */
    virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;

    /* The function that handles the calculation of the intersection
     * points and times between the Sphere object and the neutrons
     * represented by d_origins and d_vel when the neutrons start
     * inside the Sphere.
     */
    virtual void interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;

    // This member stores the Sphere's radius.
    double radius;
};

#endif
