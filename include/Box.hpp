#ifndef BOX_HPP
#define BOX_HPP

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
    
    /* The function that handles the calculation of the intersection
     * points and times between the Box object and the neutrons represented
     * by d_origins and d_vel when the neutrons start outside the Box.
     */
    virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times,
                                   std::vector< Vec3<float> > &int_coords) override;

    /* The function that handles the calculation of the intersection
     * points and times between the Box object and the neutrons represented
     * by d_origins and d_vel when the neutrons start inside the Box.
     */
    virtual void interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times,
                                   std::vector< Vec3<float> > &int_coords) override;

    // These members store the Box's side lengths in the X, Y, and Z directions.
    double X, Y, Z;
};

#endif
