#ifndef ABSTRACT_SHAPE_HPP
#define ABSTRACT_SHAPE_HPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Error.hpp"
#include "IntersectKernels.hpp"
#include "Ray.hpp"
#include "SystemVars.hpp"

extern std::unordered_map<std::string, int> interKeyDict;

/* Abstract Shape is the parent interface for the primitive
 * solids defined in McVine.
 * It is a struct instead of a class so that it replicates how
 * shapes are handled in McVine.
 */
struct AbstractShape
{
    AbstractShape(); 

    virtual ~AbstractShape() { ; }

    /* This pure virtual function will be used by the primitive
     * shapes to handle the calculation of intersection points.
     * It takes arrays of the initial position and velocity data
     * for the neutrons being considered as parameters. It also takes
     * the size of these arrays (N), the CUDA thread and block 
     * parameters (blockSize and numBlocks), and two vectors passed
     * by-reference to store the intersection times and
     * coordinates (int_times and int_coords).
     */
    virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times,
                                   std::vector< Vec3<float> > &int_coords) = 0;

    virtual void interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                   const int N, const int blockSize, const int numBlocks,
                                   std::vector<float> &int_times,
                                   std::vector< Vec3<float> > &int_coords) = 0;
 
    /* Type is a string stating which primitive the object is.
     * This member might be removed later if it ends up being
     * unnecessary.
     */
    std::string type;

    float *data;
};

#endif
