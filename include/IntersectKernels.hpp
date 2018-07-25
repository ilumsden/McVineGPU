#ifndef INTERSECT_KERNELS_HPP
#define INTERSECT_KERNELS_HPP

#include "UtilKernels.hpp"

typedef void (*intersectStart_t)(Vec3<float>&, Vec3<float>&, const float*,
                                 float*, Vec3<float>*);

/* This function calculates the intersection time and point between
 * a neutron (represented by x, y, z, va, vb, and vc) and a
 * rectangle (represented by A, B, and zdiff). The remaining
 * parameters are used to ensure that the intersection time and
 * coordinates are placed in the correct spot in the ts and pts
 * arrays.
 * This function can be called on device only.
 */
__device__ void intersectRectangle(float &ts, Vec3<float> &pts,
                                   const Vec3<float> &orig, float zdiff,
                                   const Vec3<float> &vel,
                                   const float A, const float B,
                                   const int key, int &off);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and the rounded
 * side of a Cylinder (represented by r and h). The offset parameter
 * is used to ensure the time and coordinate data is placed in the
 * correct spot in the ts and pts arrays.
 * This function can be called on device only.
 */
__device__ void intersectCylinderSide(float *ts, Vec3<float> *pts,
                                      const Vec3<float> &orig,
                                      const Vec3<float> &vel,
                                      const float r, const float h,
                                      int &offset);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and the top and bottom
 * of a Cylinder (represented by r and h). The offset parameter is used to
 * ensure the time and coordinate data is placed in the correct spot in
 * the ts and pts arrays.
 * This function can be called on device only.
 */
__device__ void intersectCylinderTopBottom(float *ts, Vec3<float> *pts,
                                           const Vec3<float> &orig,
                                           const Vec3<float> &vel,
                                           const float r, const float h,
                                           int &offset);

/* This function calculates the intersection time and coordinates between
 * a neutron (represented by x, y, z, vx, vy, and vz) and a
 * triangle (represented by points a, b, and c). The off1 and off2
 * parameters are used to ensure the time and coordinate data is placed
 * in the correct spot in the ts and pts arrays.
 * This function can be called on device only.
 */
__device__ void intersectTriangle(float &ts, Vec3<float> &pts,
                                  const Vec3<float> &orig,
                                  const Vec3<float> &vel,
                                  const Vec3<float> &a,
                                  const Vec3<float> &b,
                                  const Vec3<float> &c,
                                  int &off);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Box (represented by X, Y, and Z). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__device__ void intersectBox(Vec3<float>& origins,
                             Vec3<float>& vel,
                             const float *shapeData,
                             float* ts, Vec3<float>* pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Cylinder (represented by r and h). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__device__ void intersectCylinder(Vec3<float> &origins, Vec3<float> &vel,
                                  const float *shapeData,
                                  float *ts, Vec3<float> *pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Pyramid (represented by X, Y, and H). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__device__ void intersectPyramid(Vec3<float> &origins, Vec3<float> &vel,
                                 const float *shapeData,
                                 float *ts, Vec3<float> *pts);

/* This function controlls the calculation of the intersections between
 * a collection of neutrons (represented by rx, ry, rz, vx, vy, and vz)
 * and a Sphere (represented by radius). N is simply the number of
 * neutrons being used in the calculation. The calculated intersection times
 * and coordinates are stored in the ts and pts arrays respectively.
 * This function can be called from host.
 */
__device__ void intersectSphere(Vec3<float> &origins, Vec3<float> &vel,
                                const float *shapeData,
                                float *ts, Vec3<float> *pts);

__device__ intersectStart_t boxInt = intersectBox;
__device__ intersectStart_t cylInt = intersectCylinder;
__device__ intersectStart_t pyrInt = intersectPyramid;
__device__ intersectStart_t sphInt = intersectSphere;

__global__ void intersect(intersectStart_t intPtr, Vec3<float> *origins, 
                          Vec3<float> *vel, const float *shapeData, 
                          const int N, const int groupSize,
                          float *ts, Vec3<float> *pts);

#endif
