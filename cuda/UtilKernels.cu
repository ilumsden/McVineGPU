#include <cstdio>

#include "UtilKernels.hpp"

#ifndef PI
#define PI 3.14159265358979323846f
#endif

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

__global__ void prepRand(curandState *state, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(((seed << 10) + idx), 0, 0, &state[idx]); 
}
