#ifndef ABSTRACT_SHAPE_HPP
#define ABSTRACT_SHAPE_HPP

#include <memory>
#include <string>
#include <vector>

#include "Ray.hpp"

/* Abstract Shape is the parent interface for the primitive
 * solids defined in McVine.
 * It is a struct instead of a class so that it replicates how
 * shapes are handled in McVine.
 */
struct AbstractShape
{
    AbstractShape() { type = "Shape"; }

    virtual ~AbstractShape() { ; }

    /* This visitor-pattern acceptor function was initially created
     * as a means for handling Unary operations on shapes.
     * After looking into the process used in McVine, this function will
     * likely be deleted.
     */
    //virtual void accept(UnaryVisitor &v) = 0;

    /* This pure virtual function will be used by the primitive
     * shapes to handle the calculation of intersection points.
     * It takes arrays of the initial position and velocity data
     * for the neutrons being considered as parameters. It also takes
     * the size of these arrays (N), the CUDA thread and block 
     * parameters (blockSize and numBlocks), and two vectors passed
     * by-reference to store the intersection times and
     * coordinates (int_times and int_coords).
     */
    virtual void intersect(float *d_rx, float *d_ry, float *d_rz,
                           float *d_vx, float *d_vy, float *d_vz,
                           const int N, const int blockSize, const int numBlocks,
                           std::vector<float> &int_times, std::vector<float> &int_coords) = 0;

    /* Type is a string stating which primitive the object is.
     * This member might be removed later if it ends up being
     * unnecessary.
     */
    std::string type;
};

#endif
