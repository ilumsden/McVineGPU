#include <cstdio>

#include "UtilKernels.hpp"

__device__ bool solveQuadratic(float a, float b, float c, float &x0, float &x1)
{
    // Calculates the discriminant and returns false if it is less than 0.
    float discr = b*b - 4*a*c;
    if (discr < 0)
    {
        return false;
    }
    else
    {
        /* This process ensures that there is little to no roundoff error
         * in the evaluation of the quadratic formula.
         * This process defines a value 
         * q = -0.5 * (b + sign(b)*sqrt(b^2 - 4ac)).
         * If you define x0 = q/a (producing the standard quadratic
         * formula), x1 can be defined as c/q by multiplying the
         * other form of the formula (+/- -> -sign(b)) by
         * ((-b + sign(b)*sqrt(discr))/(-b + sign(b)*sqrt(discr))).
         */
        float q = (b > 0) ? 
                  (-0.5 * (b + sqrtf(discr))) :
                  (-0.5 * (b - sqrtf(discr)));
        x0 = q/a;
        x1 = c/q;
    }
    // This simply ensures that x0 < x1.
    if (x0 > x1)
    {
        float tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    return true;
}

__global__ void simplifyTimes(const float *times, const int N, const int groupSize, float *simp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // This is done to prevent excess threads from interfering in the code.
    if (index < N)
    {
        int count = 0;
        for (int i = 0; i < groupSize; i++)
        {
            if (times[groupSize * index + i] != -1 && count < 2)
            {
                simp[2*index+count] = times[groupSize*index+i];
                count++;
            }
        }
    }
}

__global__ void forceIntersectionOrder(float *ts, Vec3<float> *coords,
                                       const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < 2*N)
    {
        /* If the first listed intersection comes after the second
         * listed intersection, the two intersection times and
         * coordinates are swapped.
         */
        if (ts[2*index] > ts[2*index+1])
        {
            float tmpt;
            Vec3<float> tmpc;
            tmpt = ts[2*index];
            ts[2*index] = ts[2*index+1];
            ts[2*index+1] = tmpt;
            tmpc = coords[2*index];
            coords[2*index] = coords[2*index+1];
            coords[2*index+1] = tmpc;
        }
    }
}

__global__ void prepRand(curandState *state, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(((seed << 10) + idx), 0, 0, &state[idx]); 
}

__global__ void propagate(Vec3<float> *orig, float *ray_times,
                          Vec3<float> *scat_pos, float *scat_times,
                          const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        /* Updates the neutron's main position and time
         * data with the values passed through `scat_pos`
         * and `scat_times`.
         */
        orig[index] = scat_pos[index];
        ray_times[index] = scat_times[index];
    }
}

__global__ void updateProbability(float *ray_prob,
                                  Vec3<float> *orig, Vec3<float> *int_coords,
                                  const float atten, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        /* Updates the neutron's probability attribute based on
         * the absorption associated with travalling through the
         * scattering body to the scattering site.
         */
        float d = (orig[index] - int_coords[2*index]).length();
        ray_prob[index] *= expf(-(d/atten));
    }
}
