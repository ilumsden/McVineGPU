#include "IntersectKernels.hpp"

__global__ void testIntRectangle(float *ts, Vec3<float> *pts,
                                 const Vec3<float> *orig,
                                 const Vec3<float> *vel,
                                 const float X, const float Y, const float Z,
                                 const int N);

__global__ void testIntCylSide(float *ts, Vec3<float> *pts,
                               const Vec3<float> *orig,
                               const Vec3<float> *vel,
                               const float r, const float h, const int N);

__global__ void testIntCylTopBottom(float *ts, Vec3<float> *pts,
                                    const Vec3<float> *orig,
                                    const Vec3<float> *vel,
                                    const float r, const float h, const int N);

__global__ void testIntTriangle(float *ts, Vec3<float> *pts, 
                                const Vec3<float> *orig,
                                const Vec3<float> *vel,
                                const Vec3<float> *verts, const int N);

void rectangleTest(float *times, Vec3<float> *points);

void cylinderSideTest(float *times, Vec3<float> *points);

void cylinderEndTest(float *times, Vec3<float> *points);

void triangleTest(float *times, Vec3<float> *points);

void 3DTest(const int key, float *times, Vec3<float> *points);
