#ifndef _CUDA_DRIVER_H_
#define _CUDA_DRIVER_H_

#include <memory>
#include <vector>

#include "Box.hpp"
#include "Cylinder.hpp"
#include "Pyramid.hpp"
#include "Ray.hpp"
#include "Sphere.hpp"
#include "SystemVars.hpp"

/* This class controlls the execution of CUDA kernels.
 * Currently, it is mostly ment to be a convenient device for testing
 * the kernels and ensuring good data flow when multiple CUDA-based
 * operations are performed by a single test.
 */
class CudaDriver
{
    public:
        /* CudaDriver Constructor
         * This constructor generates host- and device-side arrays
         * of the initial position and velocity data that defines the
         * neutrons. It also stores the size of these arrays into a
         * member variable (N) for easy access. Finally, it uses the
         * bS parameter to set the members that dictate the CUDA
         * kernel launch parameters (blockSize and numBlocks).
         */
        CudaDriver(std::vector< std::shared_ptr<Ray> > &rays, 
                   std::shared_ptr<AbstractShape> &shape, int bS);

        /* This function deallocates the memory allocated
         * in the constructor.
         */
        ~CudaDriver();

        /* This function provides an easy mechanism to run through all
         * the calculations currently implemented.
         */
        void runCalculations();//std::shared_ptr<AbstractShape> &b);

        /* This function prints the data stored for each neutron
         * based on the following member arrays: origins, vel,
         * times, and probs. 
         */
        void printData(const std::string &fname=std::string());    

    private:

        // This function is used to initiate the intersection calculation.
        void handleExteriorIntersect(//std::shared_ptr<AbstractShape> &b, 
                                     std::vector<float> &host_time, 
                                     std::vector< Vec3<float> > &int_coords);

        // This function is used to initiate the scattering site calculation.
        void findScatteringSites(const std::vector<float> &int_times, 
                                 const std::vector< Vec3<float> > &int_coords);

        // This function is used to initiate the elastic scattering calculation.
        void findScatteringVels();

        void handleInteriorIntersect();

        // These members store the host-side copies of the neutron data.
        Vec3<float> *origins, *vel;
        float *times, *probs;
        // These members store the device-side copies of the neutron data.
        Vec3<float> *d_origins, *d_vel;
        float *d_times, *d_probs;
        // This member stores a pointer to the vector of rays.
        std::vector< std::shared_ptr<Ray> > *rayptr;
        // This member stores a pointer to the scattering body.
        std::shared_ptr<AbstractShape> b;
        // This int stores the number of neutrons (size of the above data).
        int N;
        // These are the CUDA launch parameters. 
        int blockSize, numBlocks;
};

#endif
