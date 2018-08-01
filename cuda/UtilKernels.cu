#include <cstdio>

#include "UtilKernels.hpp"

namespace mcvine
{

    namesapce gpu
    {

        namespace kernels
        {

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

            __global__ void simplifyTimePointPairs(const float *times,
                                                   const Vec3<float> *coords,
                                                   const int N,
                                                   const int inputGroupTime,
                                                   const int inputGroupCoord,
                                                   const int outputGroupSize,
                                                   float *simp_times,
                                                   Vec3<float> *simp_coords) 
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    float tmp_times[2] = {-5, -5};
                    Vec3<float> tmp_coords[2] = {Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX),                                     Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX)};
                    int count = 0;
                    for (int i = 0; i < inputGroupTime; i++)
                    {
                        if (times[inputGroupTime*index+i] != -1 && count < 2)
                        {
                            tmp_times[count] = times[inputGroupTime*index+i];
                            count++;
                        }
                    }
                    count = 0;
                    for (int i = 0; i < inputGroupCoord; i++)
                    {
                        if (coords[inputGroupCoord*index+i] != Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX)
                            && count < 2)
                        {
                            tmp_coords[count] = coords[inputGroupCoord*index+i];
                            count++;
                        }
                    }
                    count = 0;
                    for (int i = 0; i < 2; i++)
                    {
                        if (tmp_times[i] >= 0 && count < outputGroupSize)
                        {
                            simp_times[outputGroupSize*index+count] = tmp_times[i];
                            simp_coords[outputGroupSize*index+count] = tmp_coords[i];
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
                //curand_init((seed + idx), 0, 1, &state[idx]); 
                curand_init(seed, idx, 0, &state[idx]); 
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
                    ray_times[index] += scat_times[index];
                }
            }

            __global__ void updateProbability(float *ray_prob,
                                              Vec3<float> *p1, Vec3<float> *p0,
                                              const int p1GroupSize,
                                              const int p0GroupSize,
                                              const float atten, const int N)
            {
                int index = blockIdx.x * blockDim.x + threadIdx.x;
                if (index < N)
                {
                    /* Updates the neutron's probability attribute based on
                     * the absorption associated with travalling through the
                     * scattering body to the scattering site.
                     */
                    float d = (p1[p1GroupSize*index] - p0[p0GroupSize*index]).length();
                    ray_prob[index] *= expf(-(d/atten));
                }
            }

        }

    }

}
