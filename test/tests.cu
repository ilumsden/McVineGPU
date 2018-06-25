#include <cfloat>
#include <cstdio>
#include <string>
#include <vector>

#include "Error.hpp"
#include "Kernels.hpp"

#include "tests.hpp"
#include "test_kernels.hpp"

bool test_intersectTriangle()
{
    //float *ts = new float[1];
    float *ts;
    //float *pts = new float[3];
    Vec3<float> *pts;
    float *d_ts;//, *d_pts;
    Vec3<float> *d_pts;
    CudaErrchk( cudaMalloc(&d_ts, sizeof(float)) );
    initArray<float><<<numBlocksSimple, blockSizeSimple>>>(d_ts, 1, -5);
    CudaErrchkNoCode();
    //CudaErrchk( cudaMalloc(&d_pts, 3*sizeof(float)) );
    CudaErrchk( cudaMalloc(&d_pts, sizeof(Vec3<float>)) );
    //initArray<<<numBlocksSimple, blockSizeSimple>>>(d_pts, 3, FLT_MAX);
    initArray< Vec3<float> ><<<numBlocksSimple, blockSizeSimple>>>(d_pts, 1, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
    CudaErrchkNoCode();
    triangleTest<<<numBlocksSimple, blockSizeSimple>>>(d_ts, d_pts);
    CudaErrchkNoCode();
    CudaErrchk( cudaMemcpy(ts, d_ts, sizeof(float), cudaMemcpyDeviceToHost) );
    //CudaErrchk( cudaMemcpy(pts, d_pts, 3*sizeof(float), cudaMemcpyDeviceToHost) );
    CudaErrchk( cudaMemcpy(pts, d_pts, sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
    assert(ts[0] == 1);
    assert((*pts)[0] == 0);
    assert((*pts)[1] == 0);
    assert((*pts)[2] == 1);
    return true;
}

bool test_Vec3()
{
    Vec3<float> vectors[12];
    vectors[0].setX(3); vectors[0].setY(3); vectors[0].setZ(3);
    vectors[1].setX(3); vectors[1].setY(3); vectors[1].setZ(3);
    vectors[2].setX(12); vectors[2].setY(15); vectors[2].setZ(11);
    vectors[3].setX(20); vectors[3].setY(17); vectors[3].setZ(18);
    vectors[4].setX(5); vectors[4].setY(6); vectors[4].setZ(7);
    vectors[5].setX(3); vectors[5].setY(4); vectors[5].setZ(5);
    vectors[6].setX(7); vectors[6].setY(5); vectors[6].setZ(9);
    vectors[7].setX(3); vectors[7].setY(5); vectors[7].setZ(1);
    vectors[8].setX(2); vectors[8].setY(6); vectors[8].setZ(4);
    vectors[9].setX(-2); vectors[9].setY(4); vectors[9].setZ(1);
    vectors[10].setX(1); vectors[10].setY(5); vectors[10].setZ(9);
    vectors[11].setX(8); vectors[11].setY(6); vectors[11].setZ(5);
    Vec3<float> *d_vectors;
    CudaErrchk( cudaMalloc(&d_vectors, 12*sizeof(Vec3<float>)) );
    CudaErrchk( cudaMemcpy(d_vectors, &(vectors[0]), 12*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
    bool *d_res;
    CudaErrchk( cudaMalloc(&d_res, 20*sizeof(bool)) );
    std::vector<std::string> errmsgs{ "== Operator Incorrect.",
                                      "!= Operator Incorrect.",
                                      "Assignment Operator (=) Incorrect.",
                                      "getX() Incorrect.",
                                      "getY() Incorrect.",
                                      "getZ() Incorrect.",
                                      "setX() Incorrect.",
                                      "setY() Incorrect.", 
                                      "setZ() Incorrect.",
                                      "normalize() Incorrect.",
                                      "+ Operator Incorrect.",
                                      "- Operator Incorrect.",
                                      "Negation Operator Incorrect.",
                                      "Scalar * Operator Incorrect.",
                                      "Vector * Operator Incorrect.",
                                      "+= Operator Incorrect.",
                                      "-= Operator Incorrect.",
                                      "Scalar *= Operator Incorrect.",
                                      "Vector *= Operator Incorrect.",
                                      "Dot Product (|) Operator Incorrect."};
    bool *res = (bool*)malloc(20*sizeof(bool));
    printf("CPU FLT_MIN = %e\n", FLT_MIN);
    vec3Test<<<numBlocksSimple, blockSizeSimple>>>(d_vectors, d_res);
    CudaErrchkNoCode();
    CudaErrchk( cudaMemcpy(res, d_res, 20*sizeof(bool), cudaMemcpyDeviceToHost) );
    bool noErrors = true;
    for (int i = 0; i < 21; i++)
    {
        if (res[i])
        {
            fprintf(stderr, "Error during Testing of Vec3:\n    %s\n", errmsgs[i].c_str());
            noErrors = false;
        }
    }
    return noErrors;
}
