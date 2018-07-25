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

typedef std::pair<intersectStart_t, int> devicePair;

extern std::unordered_map<std::string, devicePair> funcPtrDict;

/* Abstract Shape is the parent interface for the primitive
 * solids defined in McVine.
 * It is a struct instead of a class so that it replicates how
 * shapes are handled in McVine.
 */
struct AbstractShape
{
    AbstractShape() 
    { 
        type = "Shape"; 
        if (funcPtrDict.empty())
        {
            funcPtrDict.insert(
                std::make_pair<std::string, devicePair>("Box", 
                std::make_pair<intersectStart_t, int>(NULL, 6)));
            funcPtrDict.insert(
                std::make_pair<std::string, devicePair>("Cylinder", 
                std::make_pair<intersectStart_t, int>(NULL, 4)));
            funcPtrDict.insert(
                std::make_pair<std::string, devicePair>("Pyramid", 
                std::make_pair<intersectStart_t, int>(NULL, 5)));
            funcPtrDict.insert(
                std::make_pair<std::string, devicePair>("Sphere", 
                std::make_pair<intersectStart_t, int>(NULL, 2)));
            CudaErrchk( cudaMemcpyFromSymbol(&(std::get<0>(funcPtrDict["Box"])), boxInt, sizeof(intersectStart_t)) );
            CudaErrchk( cudaMemcpyFromSymbol(&(std::get<0>(funcPtrDict["Cylinder"])), cylInt, sizeof(intersectStart_t)) );
            CudaErrchk( cudaMemcpyFromSymbol(&(std::get<0>(funcPtrDict["Pyramid"])), pyrInt, sizeof(intersectStart_t)) );
            CudaErrchk( cudaMemcpyFromSymbol(&(std::get<0>(funcPtrDict["Sphere"])), sphInt, sizeof(intersectStart_t)) );
        }
    }

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
