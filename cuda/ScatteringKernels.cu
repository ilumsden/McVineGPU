#include "ScatteringKernels.hpp"

__device__ void randCoord(Vec3<float> &orig, Vec3<float> &vel,
                          float *time,
                          Vec3<float> &pos,
                          float &scat_time,
                          curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* Instead of pasing the initial ray data, the two intersection
     * points and times are used to recalculate the velocities.
     */
    float dt = time[1] - time[0];
    // cuRand is used to generate a random time between 0 and dt.
    float randt = curand_uniform(&(state[index]));
    randt *= dt;
    scat_time = randt + time[0];
    /* Basic kinematics are used to calculate the coordinates of
     * the randomly chosen scattering site.
     */
    pos = orig + (vel*scat_time);
}

__global__ void calcScatteringSites(float *ts, 
                                    Vec3<float> *orig, Vec3<float> *vel,
                                    Vec3<float> *pos, float *scat_times,
                                    curandState *state, const int N)
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
            /* The randCoord function assumes that the first time
             * is smaller than the second. If this is not the
             * case, the times and the corresponding intersection
             * coordinates are swapped.
             */
            /*if (ts[2*index] > ts[2*index+1])
            {
                float tmpt;
                tmpt = ts[2*index];
                ts[2*index] = ts[2*index+1];
                ts[2*index+1] = tmpt;
            }*/
            /* The randCoord function is called to determine the
             * scattering site.
             */
            randCoord(orig[index], vel[index], &(ts[2*index]), pos[index], scat_times[index], state);
        }
    }
}

__global__ void elasticScatteringKernel(const float *ray_time,
                                        Vec3<float> *vel,
                                        curandState *state,
                                        const int N)
{
    /* To start each curandState will be used to generate the random
     * z and phi values.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N && ray_time[index] > 0)
    {
        float z = curand_uniform(&(state[index]));
        z *= 2;
        z -= 1;
        float phi = curand_uniform(&(state[index]));
        phi *= 2*PI;
        float theta = acosf(z);
        float r = vel[index].length();        
        vel[index][0] = r * cosf(phi) * sinf(theta);
        vel[index][1] = r * sinf(phi) * sinf(theta);
        vel[index][2] = r * z;
    }
}
