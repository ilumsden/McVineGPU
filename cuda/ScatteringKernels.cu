#include "ScatteringKernels.hpp"

__device__ void randCoord(Vec3<float> *inters, float *time,
                          Vec3<float> &pos,
                          curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /* Instead of pasing the initial ray data, the two intersection
     * points and times are used to recalculate the velocities.
     */
    float dt = time[1] - time[0];
    Vec3<float> m = (inters[1] - inters[0]) * (1.0/dt);
    // cuRand is used to generate a random time between 0 and dt.
    float randt = curand_uniform(&(state[index]));
    randt *= dt;
    /* Basic kinematics are used to calculate the coordinates of
     * the randomly chosen scattering site.
     */
    pos = inters[0] + (m*randt);
}

__global__ void calcScatteringSites(float *ts, Vec3<float> *int_pts,
                                    Vec3<float> *pos, curandState *state,
                                    const int N)
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
            if (ts[2*index] > ts[2*index+1])
            {
                float tmpt;
                Vec3<float> tmpv;
                tmpt = ts[2*index];
                ts[2*index] = ts[2*index+1];
                ts[2*index+1] = tmpt;
                tmpv = int_pts[2*index];
                int_pts[2*index] = int_pts[2*index+1];
                int_pts[2*index+1] = tmpv;
            }
            /* The randCoord function is called to determine the
             * scattering site.
             */
            randCoord(&(int_pts[2*index]), &(ts[2*index]), pos[index], state);
        }
    }
}

__global__ void elasticScatteringKernel(const Vec3<float> *initVel,
                                        Vec3<float> *postVel,
                                        curandState *state,
                                        const int N)
{
    /* To start each curandState will be used to generate the random
     * z and phi values.
     */
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        float z = curand_uniform(state[index]);
        z *= 2;
        z -= 1;
        float phi = curand_uniform(state[index]);
        phi *= 2*PI;
        float theta = acosf(z);
        float rho = sqrtf(initVel[0]*initVel[0] +
                          initVel[1]*initVel[1] +
                          initVel[2]*initVel[2]);
        postVel[index][0] = rho * sinf(phi) * cosf(theta);
        postVel[index][1] = rho * sinf(phi) * sinf(theta);
        postVel[index][2] = rho * z;
    }
}
