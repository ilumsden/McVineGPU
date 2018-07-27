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
    Cylinder() 
    { 
        type = "Cylinder"; 
        data = new float[2];
        data[0] = 0; data[1] = 0;
    }
    
    /* "Explicit" Constructor
     * This function takes two floats and uses them to set
     * the Cylinder's radius and height.
     */
    Cylinder(const float kRadius, const float kHeight)
    {
        type = "Cylinder";
        data = new float[2];
        data[0] = kRadius;
        data[1] = kHeight;
    }

    ~Cylinder() { delete [] data; }

    /* The function that handles the calculation of the intersection
     * points and times between the Cylinder object and the neutrons
     * represented by d_origins and d_vel when the neutrons start
     * outside the Cylinder.
     */
    virtual void exteriorIntersect(std::vector<Vec3<float>*> &d_origins, 
                                   std::vector<Vec3<float>*> &d_vel,
                                   const int blockSize, 
                                   const std::vector<int> &numBlocks,
                                   const std::vector<int> &steps,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;

    /* The function that handles the calculation of the intersection
     * points and times between the Cylinder object and the neutrons
     * represented by d_origins and d_vel when the neutrons start
     * inside the Cylinder.
     */
    virtual void interiorIntersect(std::vector<Vec3<float>*> &d_origins, 
                                   std::vector<Vec3<float>*> &d_vel,
                                   const int blockSize,
                                   const std::vector<int> &numBlocks,
                                   const std::vector<int> &steps,
                                   std::vector<float> &int_times, 
                                   std::vector< Vec3<float> > &int_coords) override;
};

#endif
