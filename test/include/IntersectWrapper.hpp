#include "IntersectKernels.hpp"
#include <cfloat>
#include <vector>

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

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

            void rectangleTest(Vec3<float> &orig, Vec3<float> &vel, float &time, Vec3<float> &point);

            void cylinderSideTest(Vec3<float> &orig, Vec3<float> &vel, float *time, Vec3<float> *point);

            void cylinderEndTest(Vec3<float> &orig, Vec3<float> &vel, float *time, Vec3<float> *point);

            void triangleTest(Vec3<float> &orig, Vec3<float> &vel, float &time, Vec3<float> &point);

            void solidTest(const int key, float *time, Vec3<float> *point);

        }

    }

}
