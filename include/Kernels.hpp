#ifndef KERNELS_HPP
#define KERNELS_HPP

/* This file lists the function declarations for all CUDA kernels.
 * It will likely be broken into several smaller, more specific
 * files later.
 */

#include <curand.h>
#include <curand_kernel.h>

#include "Vec3.hpp"

/* This function initializes the contents of the data array with the
 * value val.
 * This function can be called from host.
 */
template <typename T>
__global__ void initArray(T* data, const int size, const T val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
    {
        data[i] = val;
    }
}

/* This function calculates the dot product between vectors a and b.
 * This function can be called on device only.
 */
//__device__ float dot(float ax, float ay, float az,
//                     float bx, float by, float bz);

/* This function calculates the cross product between
 * vectors a and b, and it stores the result in vector c.
 * This function can be called on device only.
 */
//__device__ void cross(float ax, float ay, float az,
//                      float bx, float by, float bz,
//                      float *cx, float *cy, float *cz);

/* This function calculates the intersection time and point between
 * a neutron (represented by x, y, z, va, vb, and vc) and a
 * rectangle (represented by A, B, and zdiff). The remaining
 * parameters are used to ensure that the intersection time and
 * coordinates are placed in the correct spot in the ts and pts
 * arrays.
 * This function can be called on device only.
 */
__device__ void intersectRectangle(//float* ts, float* pts, 
                                   float* ts, Vec3<float>* pts,
                                   //float x, float y, float z, float zdiff,
                                   const Vec3<float> &orig, float zdiff,
                                   //float va, float vb, float vc,
                                   const Vec3<float> &vel,
                                   const float A, const float B,
                                   const int key, const int groupSize,
                                   const int off1, int &off2);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and the rounded
 * side of a Cylinder (represented by r and h). The offset parameter
 * is used to ensure the time and coordinate data is placed in the
 * correct spot in the ts and pts arrays.
 * This function can be called on device only.
 */
__device__ void intersectCylinderSide(float *ts, float *pts,
                                      float x, float y, float z,
                                      float vx, float vy, float vz,
                                      const float r, const float h,
                                      int &offset);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and the top and bottom
 * of a Cylinder (represented by r and h). The offset parameter is used to
 * ensure the time and coordinate data is placed in the correct spot in
 * the ts and pts arrays.
 * This function can be called on device only.
 */
__device__ void intersectCylinderTopBottom(float *ts, float *pts,
                                           float x, float y, float z,
                                           float vx, float vy, float vz,
                                           const float r, const float h,
                                           int &offset);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and a
 * triangle (represented by points a, b, and c). The off1 and off2
 * parameters are used to ensure the time and coordinate data is placed
 * in the correct spot in the ts and pts arrays.
 * This function can be called on device only.
 *
 * NOTE: this function is not yet working.
 */
__device__ void intersectTriangle(float *ts, float *pts,
                                  const float x, const float y, const float z,
                                  const float vx, const float vy, const float vz,
                                  const float aX, const float aY, const float aZ,
                                  const float bX, const float bY, const float bZ,
                                  const float cX, const float cY, const float cZ,
                                  const int off1, int &off2);

/*__device__ void calculateQuadCoef(float x, float vx, float vy, float vz,
                                  float dist, float &disc,
                                  float &a, float &b, float &c);*/

/* This function solves the quadratic equation given values a, b, and c.
 * The results of the equation are stored in x0 and x1.
 * This function can be called on device only.
 */
__device__ bool solveQuadratic(float a, float b, float c, float &x0, float &x1);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Box (represented by X, Y, and Z). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__global__ void intersectBox(//float* rx, float* ry, float* rz,
                             Vec3<float>* origins,
                             //float* vx, float* vy, float* vz,
                             Vec3<float>* vel,
                             const float X, const float Y, const float Z,
                             const int N, float* ts, //float* pts);
                             Vec3<float>* pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Cylinder (represented by r and h). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__global__ void intersectCylinder(float *rx, float *ry, float *rz,
                                  float *vx, float *vy, float *vz,
                                  const float r, const float h,
                                  const int N, float *ts, float *pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Pyramid (represented by X, Y, and H). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 *
 * NOTE: Because the intersectTriangle function is not yet working,
 *       this function is also not yet working.
 */
__global__ void intersectPyramid(float *rx, float *ry, float *rz,
                                 float *vx, float *vy, float *vz,
                                 const float X, const float Y, const float H,
                                 const int N, float *ts, float *pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Sphere (represented by radius). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__global__ void intersectSphere(float *rx, float *ry, float *rz,
                                float *vx, float *vy, float *vz,
                                const float radius,
                                const int N, float *ts, float *pts);

/* This function takes the times produced by the intersect functions above
 * for solids (i.e. Box, Sphere, Cylinder, etc.) and reduces the array so
 * that there are only 2 times per neutron. If there are no meaningful
 * times for a neturon, the times are simplified to 2 -1s. The simplified
 * data is stored in simp. N is the number of neutrons, and groupSize is
 * the number of times per neutron in ts.
 * This function can be called from host.
 */
__global__ void simplifyTimes(const float* ts, const int N, const int groupSize, float* simp);

/* This function seeds and initializes a cuRand random number generator
 * using the cuRand States stored in state and the seed value "seed."
 * This function can be called from host.
 */
__global__ void prepRand(curandState *state, int seed);

/* This function generates a random point on the line between the two
 * points represented by inters. The point's coordinates are stored in
 * sx, sy, and sz. The state parameter is the cuRand state for the
 * thread that this function is called from.
 * This function can be called on device only.
 */
__device__ void randCoord(float* inters, float* time, float *sx, float *sy, float *sz, curandState *state);

/* This function uses the intersection points (int_pts)
 * and times (ts) calculated by the intersect functions to choose 
 * a random point within the sample from which scattering occurs. 
 * These random scattering points are stored in pos. As usual, N is the
 * number of neutrons used in the calculation. The state parameter is
 * an array of cuRand States that are used to randomly generate the
 * scattering points.
 * This function can be called from host.
 */
__global__ void calcScatteringSites(float* ts, float* int_pts, 
                                    float* pos, curandState *state, const int N);

#endif
