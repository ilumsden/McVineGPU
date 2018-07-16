#include "ScatteringKernels.hpp"

__device__ void randCoord(Vec3<float> &orig, Vec3<float> &vel,
                          float *time,
                          Vec3<float> &pos,
                          float &scat_time,
                          float rand)
{
    /* Instead of pasing the initial ray data, the two intersection
     * points and times are used to recalculate the velocities.
     */
    float dt = time[1] - time[0];
    // cuRand is used to generate a random time between 0 and dt.
    rand *= dt;
    scat_time = rand + time[0];
    /* Basic kinematics are used to calculate the coordinates of
     * the randomly chosen scattering site.
     */
    pos = orig + (vel*scat_time);
}

__global__ void calcScatteringSites(float *ts, 
                                    Vec3<float> *orig, Vec3<float> *vel,
                                    Vec3<float> *pos, float *scat_times,
                                    float *rands, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // This is done to prevent excess threads from interfering in the code.
    if (index < N)
    {
        /* If the intersection times for the neutron are the default
         * value of -5, there was no intersection, so the function
         * terminates.
         */
        if (ts[2*index] != -5 && ts[2*index+1] != -5)
        {
            /* The randCoord function is called to determine the
             * scattering site.
             */
            randCoord(orig[index], vel[index], &(ts[2*index]), pos[index], scat_times[index], rands[index]);
        }
    }
}

__global__ void elasticScatteringKernel(const float *ray_time,
                                        Vec3<float> *vel,
                                        float *rands,
                                        const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N && ray_time[index] > 0)
    {
        /* The z coordinate of the unit velocity vector is chosen
         * at random and manipulated so that it falls between
         * -1 and 1.
         */
        float z = rands[2*index];
        z *= 2;
        z -= 1;
        /* The spherical coordinate `phi` is randomly chosen
         * and manipulated so that it falls between 0 and 2*Pi.
         */
        float phi = rands[2*index+1];
        phi *= 2*PI;
        /* Since z is based on a unit vector, theta is calculated
         * using standard Cartesian-to-Spherical coordinate conversion
         * with r = 1.
         */
        float theta = acosf(z);
        /* Since the scattering is elastic, the post-scattering velocity
         * must have the same magnitude as the pre-scattering velocity.
         * As a result, the pre-scattering magnitude must be recorded.
         */
        float r = vel[index].length();        
        /* The post-scattering velocity is calculated using standard
         * Spherical-to-Cartesian conversion.
         */
        vel[index][0] = r * cosf(phi) * sinf(theta);
        vel[index][1] = r * sinf(phi) * sinf(theta);
        vel[index][2] = r * z;
    }
}
