#include <cfloat>

#include "Error.hpp"
#include "Kernels.hpp"

#include "tests.hpp"
#include "test_kernels.hpp"

bool test_intersectTriangle()
{
    float *ts = new float[1];
    float *pts = new float[3];
    float *d_ts, *d_pts;
    CudaErrchk( cudaMalloc(&d_ts, sizeof(float)) );
    initArray<<<numBlocksSimple, blockSizeSimple>>>(d_ts, 1, -5);
    CudaErrchkNoCode();
    CudaErrchk( cudaMalloc(&d_pts, 3*sizeof(float)) );
    initArray<<<numBlocksSimple, blockSizeSimple>>>(d_pts, 3, FLT_MAX);
    CudaErrchkNoCode();
    triangleTest<<<numBlocksSimple, blockSizeSimple>>>(d_ts, d_pts);
    CudaErrchkNoCode();
    CudaErrchk( cudaMemcpy(ts, d_ts, sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(pts, d_pts, 3*sizeof(float), cudaMemcpyDeviceToHost) );
    assert(ts[0] == 1);
    assert(pts[0] == 0);
    assert(pts[1] == 0);
    assert(pts[2] == 1);
    return true;
}
