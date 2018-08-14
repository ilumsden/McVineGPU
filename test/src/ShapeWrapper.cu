#include "ShapeWrapper.hpp"

#include "Error.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace test
        {

            void runIntersection(AbstractShape &shape, const int key,
                                 Vec3<float> &orig, Vec3<float> &vel,
                                 std::vector<float> &int_times,
                                 std::vector< Vec3<float> > &int_coords)
            {
                Vec3<float> *d_origins, *d_vel;
                CudaErrchk( cudaMalloc(&d_origins, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMalloc(&d_vel, sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(d_origins, &orig, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, &vel, sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                if (key == 0)
                {
                    shape.exteriorIntersect(d_origins, d_vel, 1, 1, 1, int_times, int_coords);
                }
                else
                {
                    shape.interiorIntersect(d_origins, d_vel, 1, 1, 1, int_times, int_coords);
                }
                CudaErrchk( cudaFree(d_origins) );
                CudaErrchk( cudaFree(d_vel) );
            }

        }

    }

}
