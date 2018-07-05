#ifndef SCATTERING_KERNELS_HPP
#define SCATTERING_KERNELS_HPP

#include "UtilKernels.hpp"

/* This function generates a random point on the line between the two
 * points represented by inters. The point's coordinates are stored in
 * sx, sy, and sz. The state parameter is the cuRand state for the
 * thread that this function is called from.
 * This function can be called on device only.
 */
__device__ void randCoord(Vec3<float> &orig, Vec3<float> &vel,
                          float *int_times,
                          Vec3<float> &pos,
                          float &scat_time,
                          curandState *state);

/* This function uses the intersection points (int_pts)
 * and times (ts) calculated by the intersect functions to choose 
 * a random point within the sample from which scattering occurs. 
 * These random scattering points are stored in pos. As usual, N is the
 * number of neutrons used in the calculation. The state parameter is
 * an array of cuRand States that are used to randomly generate the
 * scattering points.
 * This function can be called from host.
 */
__global__ void calcScatteringSites(float *ts, 
                                    Vec3<float> *orig, Vec3<float> *vel,
                                    Vec3<float> *pos, float *scat_times,
                                    curandState *state, const int N);

__global__ void elasticScatteringKernel(const float *ray_time,
                                        Vec3<float> *vel,
                                        curandState *state,
                                        const int N);

#endif
