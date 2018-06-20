#include "test_kernels.hpp"
#include "Kernels.hpp"

__global__ void triangleTest(float *ts, float *pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0)
    {
        int offset = 0;
        intersectTriangle(ts, pts,
                          0, 0, 0,
                          0, 0, 1,
                          0, 1, 1,
                          1, -1, 1,
                          -1, -1, 1,
                          0, offset);
    }
}

__global__ void vec3Test(Vec3<float> *vectors, bool *res)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    Vec3<float> opvec, cpy;
    float dp;
    switch (index)
    {
        case 0: 
            res[0] = !(vectors[0] == vectors[1]);
            break;
        case 1:
            res[1] = (vectors[0] != vectors[1]);
            break;
        case 3:
            cpy = vectors[3];
            res[2] = !(cpy == vectors[3]);
            break;
        case 4:
            res[3] = !(vectors[4].getX() == 5);
            res[4] = !(vectors[4].getY() == 6);
            res[5] = !(vectors[4].getZ() == 7);
            vectors[4].setX(10);
            vectors[4].setY(11);
            vectors[4].setZ(12);
            res[6] = !(vectors[4].getX() == 10);
            res[7] = !(vectors[4].getY() == 11);
            res[8] = !(vectors[4].getZ() == 12);
            break;
        case 5:
            // vectors[5] is originally <3, 4, 5>
            vectors[5].normalize();
            printf("normalize\n    xsol = %f\n    ysol = %f\n    zsol = %f\n", (3.0/(5.0*sqrtf(2))), (4.0/(5.0*sqrtf(2))), (5.0/(5.0*sqrtf(2)))); 
            res[9] = !((vectors[5].getX() == (3.0/(5.0*sqrtf(2.0)))) && (vectors[5].getY() == (4.0/(5.0*sqrtf(2.0)))) && (vectors[5].getZ() == (5.0/(5.0*sqrtf(2.0)))));
            break;
        case 6:
            /* vectors[6] = <7, 5, 9>
             * vectors[7] = <3, 5, 1>
             */
            opvec = vectors[6] + vectors[7];
            res[10] = !((opvec.getX() == 10) && (opvec.getY() == 10) && (opvec.getZ() == 10));
            opvec = vectors[6] - vectors[7];
            res[11] = !((opvec.getX() == 4) && (opvec.getY() == 0) && (opvec.getZ() == 8));
            opvec = -opvec;
            res[12] = !((opvec.getX() == -4) && (opvec.getY() == 0) && (opvec.getZ() == -8));
            opvec = vectors[6] * 5;
            res[13] = !((opvec.getX() == 35) && (opvec.getY() == 25) && (opvec.getZ() == 45));
            opvec = vectors[6] * vectors[7];
            res[14] = !((opvec.getX() == -40) && (opvec.getY() == 20) && (opvec.getZ() == 20));
            break;
        case 7:
            /* vectors[8] = <2, 6, 4>
             * vectors[9] = <-2, 4, 1>
             */
            vectors[8] += vectors[9];
            res[15] = !((vectors[8].getX() == 0) && (vectors[8].getY() == 10) && (vectors[8].getZ() == 5));
            vectors[8] -= vectors[9];
            res[16] = !((vectors[8].getX() == 2) && (vectors[8].getY() == 6) && (vectors[8].getZ() == 4));
            vectors[8] *= 2;
            res[17] = !((vectors[8].getX() == 4) && (vectors[8].getY() == 12) && (vectors[8].getZ() == 8));
            vectors[8] *= vectors[9];
            res[18] = !((vectors[8].getX() == -20) && (vectors[8].getY() == -20) && (vectors[8].getZ() == 40));
            break;
        case 8:
            /* vectors[10] = <1, 5, 9>
             * vectors[11] = <8, 6, 5>
             */
            dp = vectors[10] | vectors[11];
            res[19] = !(dp == 83);
        default: break;
    }
}
