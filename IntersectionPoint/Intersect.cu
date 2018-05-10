#include <thrust/device_vector.h>

__global__
void intersectRectangle(
    double* rx, double* ry, double* rz,
    double* vx, double* vy, double* vz,
    const double X, const double Y, const double Z, const int N,
    thrust::device_vector<double> &ts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    auto calct = [](double x, double y, double z, 
                    double va, double vb, double vc)
        {
            double t = (0-z)/vc;
            double r1x = x+va*t; r1y = y+vb*t;
            if (fabs(r1x) < X/2 && fabs(r1y) < Y/2)
            {
                ts.push_back(t);
            }
            else
            {
                ts.push_back(-1);
            }
        };
    for (int i = index; i < N; i += stride)
    {
        if (vz[i] != 0)
        {
            calct(rx[i], ry[i], rz[i]-Z/2, vx[i], vy[i], vz[i]);
            calct(rx[i], ry[i], rz[i]+Z/2, vx[i], vy[i], vz[i]);
        }
        if (vx[i] != 0)
        {
            calct(ry[i], rz[i], rx[i]-X/2, vy[i], vz[i], vx[i]);
            calct(ry[i], rz[i], rx[i]+X/2, vy[i], vz[i], vx[i]);
        }
        if (vy[i] != 0)
        {
            calct(rz[i], rx[i], ry[i]-Y/2, vz[i], vx[i], vy[i]);
            calct(rz[i], rx[i], ry[i]+Y/2, vz[i], vx[i], vy[i]);
        }
    }
}
