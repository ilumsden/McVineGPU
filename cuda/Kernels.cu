#include <cstdio>

#include "Kernels.hpp"

__global__ void initArray(float *data, int size, const float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += stride)
    {
        data[i] = val;
    }
}

__device__ float dot(float ax, float ay, float az,
                     float bx, float by, float bz)
{
    return ax*bx + ay*by + az*bz;
}

__device__ void cross(float ax, float ay, float az,
                      float bx, float by, float bz,
                      float *cx, float *cy, float *cz)
{
    *cx = ay*bz - az*by;
    *cy = az*bx - ax*bz;
    *cz = ax*by - by*bx;
    return;
}

/* This is a device-only helper function for determining the time
 * it takes a ray to intersect the rectangle specified by the `intersectRectangle`
 * function.
 * It is a CUDA version of the intersectRectangle function from ArrowIntersector.cc
 * in McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__device__ void intersectRectangle(float* ts, float* pts,
                                   float x, float y, float z, float zdiff,
                                   float va, float vb, float vc, 
                                   const float A, const float B,
                                   const int key, const int groupSize, 
                                   const int off1, int &off2)
{
    z -= zdiff;
    float t = (0-z)/vc;
    float r1x = x+va*t; 
    float r1y = y+vb*t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (fabsf(r1x) < (A/2) && fabsf(r1y) < (B/2))
    {
        float ix, iy, iz;
        if (key == 0)
        {
            ix = r1x;
            iy = r1y;
            iz = zdiff;
        }
        else if (key == 1)
        {
            iy = r1x;
            iz = r1y;
            ix = zdiff;
        }
        else
        {
            iz = r1x;
            ix = r1y;
            iy = zdiff;
        }
        if (off2 == 0 || off2 == 3)
        {
            pts[6*index + off2] = ix;
            pts[6*index + off2 + 1] = iy;
            pts[6*index + off2 + 2] = iz;
            off2 += 3;
            //printf("Rectangle: index = %i    off2 = %i\n", index, off2);
        }
        //ascii(r) = 114
        ts[off1 + index*groupSize] = t + 114;
    }
    else
    {
        ts[off1 + index*groupSize] = -1;
    }
}

__device__ void intersectCylinderSide(float *ts, float *pts,
                                      float x, float y, float z,
                                      float vx, float vy, float vz,
                                      const float r, const float h, 
                                      int &offset)
{
    float a = vx*vx + vy*vy;
    float b = x*vx + y*vy;
    float c = x*x+y*y - r*r;
    float k = b*b - a*c;
    float t;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < 0)
    {
        ts[4*index + 2] = -1;
        ts[4*index + 3] = -1;
        return;
    }
    else if (k == 0)
    {
        t = -b/a;
        ts[4*index + 3] = -1;
        if (fabsf(z+vz*t) < h/2)
        {
            ts[4*index + 2] = t;
            if (offset == 0 || offset == 3)
            {
                pts[6*index + offset] = x+vx*t;
                pts[6*index + offset + 1] = y+vy*t;
                pts[6*index + offset + 2] = z+vz*t;
                offset += 3;
            }
        }
        else
        {
            ts[4*index + 2] = -1;
        }
    }
    __syncthreads();
    int i = 2;
    k = sqrtf(k);
    t = (-b+k)/a;
    if (fabsf(z+vz*t) < h/2)
    {
        ts[4*index + i] = t;
        i++;
        if (offset == 0 || offset == 3)
        {
            pts[6*index + offset] = x+vx*t;
            pts[6*index + offset + 1] = y+vy*t;
            pts[6*index + offset + 2] = z+vz*t;
            offset += 3;
        }
    }
    t = (-b-k)/a;
    if (fabsf(z+vz*t) < h/2)
    {
        ts[4*index + i] = t;
        i++;
        if (offset == 0 || offset == 3)
        {
            pts[6*index + offset] = x+vx*t;
            pts[6*index + offset + 1] = y+vy*t;
            pts[6*index + offset + 2] = z+vz*t;
            offset += 3;
        }
    }
    if (i < 4)
    {
        for (int j = i; j < 4; j++)
        {
            ts[4*index + j] = -1;
        }
    }
    __syncthreads();
}

__device__ void intersectCylinderTopBottom(float *ts, float *pts,
                                           float x, float y, float z,
                                           float vx, float vy, float vz,
                                           const float r, const float h,
                                           int &offset)
{
    float r2 = r*r;
    float hh = h/2;
    float x1, y1;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float t = (hh-z)/vz;
    x1 = x + vx*t;
    y1 = y + vy*t;
    if (x1*x1 + y1*y1 <= r2)
    {
        ts[4*index] = t;
        if (offset == 0 || offset == 3)
        {
            pts[6*index + offset] = x1;
            pts[6*index + offset + 1] = y1;
            pts[6*index + offset + 2] = hh;
            offset += 3;
        }
    }
    else
    {
        ts[4*index] = -1;
    }
    t = (-hh-z)/vz;
    x1 = x + vx*t;
    y1 = y + vy*t;
    if (x1*x1 + y1*y1 <= r2)
    {
        ts[4*index + 1] = t;
        if (offset == 0 || offset == 3)
        {
            pts[6*index + offset] = x1;
            pts[6*index + offset + 1] = y1;
            pts[6*index + offset + 2] = -hh;
            offset += 3;
        }
    }
    else
    {
        ts[4*index + 1] = -1;
    }
}

__device__ void intersectTriangle(float *ts, float *pts,
                                  const float x, const float y, const float z,
                                  const float vx, const float vy, const float vz,
                                  const float aX, const float aY, const float aZ, 
                                  const float bX, const float bY, const float bZ,
                                  const float cX, const float cY, const float cZ,
                                  const int off1, int &off2)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float abX = bX - aX, abY = bY - aY, abZ = bZ - aZ;
    float acX = cX - aX, acY = cY - aY, acZ = cZ - aZ;
    float nX, nY, nZ;
    cross(abX, abY, abZ, acX, acY, acZ, &nX, &nY, &nZ);
    float nLength = fabsf(nX)*fabsf(nX)+fabsf(nY)*fabsf(nY)+fabsf(nZ)*fabsf(nZ);
    nLength = sqrtf(nLength);
    nX /= nLength; nY /= nLength; nZ /= nLength;
    float d = dot(nX, nY, nZ, aX, aY, aZ);
    float v_p = dot(nX, nY, nZ, vx, vy, vz);
    if (fabsf(v_p) < 1e-10)
    {
        ts[5*index + off1] = -1;
        return;
    }
    float r_p = dot(nX, nY, nZ, x, y, z);
    float t = (d - r_p)/v_p;
    //printf("index = %i\n    abX = %f abY = %f abZ = %f\n    acX = %f acY = %f acZ = %f\n    nX = %f nY = %f nZ = %f\n    d = %f r_p = %f v_p = %f\n    t = %f\n", index, abX, abY, abZ, acX, acY, acZ, nX, nY, nZ, d, r_p, v_p, t);
    float pX = x + vx*t, pY = y + vy*t, pZ = z + vz*t;
    float apX = pX - aX, apY = pY - aY, apZ = pZ - aZ;
    float ncX, ncY, ncZ;
    cross(nX, nY, nZ, acX, acY, acZ, &ncX, &ncY, &ncZ);
    float c1 = dot(apX, apY, apZ, ncX, ncY, ncZ)/dot(abX, abY, abZ, ncX, ncY, ncZ);
    if (c1 < 0)
    {
        ts[5*index + off1] = -1;
        return;
    }
    float nbX, nbY, nbZ;
    cross(nX, nY, nZ, abX, abY, abZ, &nbX, &nbY, &nbZ);
    float c2 = dot(apX, apY, apZ, nbX, nbY, nbZ)/dot(acX, acY, acZ, nbX, nbY, nbZ);
    if (c2 < 0)
    {
        ts[5*index + off1] = -1;
        return;
    }
    if (c1+c2 > 1)
    {
        ts[5*index + off1] = -1;
        return;
    }
    // Set time to actual value and record pX, pY, and pZ as int pts.
    // ascii(T) = 84
    ts[5*index + off1] = t + 84;
    if (off2 == 0 || off2 == 3)
    {
        pts[6*index + off2] = pX;
        pts[6*index + off2 + 1] = pY;
        pts[6*index + off2 + 2] = pZ;
        //printf("index = %i: time = %f\n    x = %f y = %f z = %f\n    vx = %f vy = %f vz = %f\n    pX = %f pY = %f pZ = %f\n    pts[%i] = %f pts[%i] = %f pts[%i] = %f\n", index, t, x, y, z, vx, vy, vz, pX, pY, pZ, 6*index+off2, pts[6*index + off2], 6*index+off2+1, pts[6*index + off2+1], 6*index+off2+2, pts[6*index + off2+2]);
        off2 += 3;
        //printf("Triangle: index = %i    off2 = %i\n", index, off2);
    }
    __syncthreads();
    return;
}

/*__device__ void calculateQuadCoef(float x, float vx, float vy, float vz,
                                  float dist, float &disc,
                                  float &a, float &b, float &c)
{
    a = 1 + (vy/vx)*(vy/vx) + (vz/vx)*(vz/vx);
    b = -2*(1 + ((x*vy*vy)/(vx*vx)) + ((x*vz*vz)/(vx*vx)));
    c = x*x + ((x*vy)/vx)*((x*vy)/vx) + ((x*vz)/vx)*((x*vz)/vx);
    c -= dist*dist;
    disc = b*b - 4*a*c;
    return;
}*/

__device__ bool solveQuadratic(float a, float b, float c, float &x0, float &x1)
{
    float discr = b*b - 4*a*c;
    if (discr < 0)
    {
        return false;
    }
    else
    {
        // Done to avoid "catastrophic cancellation"
        float q = (b > 0) ? 
                  (-0.5 * (b + sqrtf(discr))) :
                  (-0.5 * (b - sqrtf(discr)));
        x0 = q/a;
        x1 = c/q;
    }
    if (x0 > x1)
    {
        float tmp = x0;
        x0 = x1;
        x1 = tmp;
    }
    return true;
}

/* This is a global CUDA function for controlling the calculation of intersection
 * times. It is a CUDA version of the visit function from ArrowIntersector.cc in
 * McVine (mcvine/packages/mccomposite/lib/geometry/visitors/ArrowIntersector.cc).
 */
__global__ void intersectBox(float* rx, float* ry, float* rz,
                             float* vx, float* vy, float* vz,
                             const float X, const float Y, const float Z, 
                             const int N, float* ts, float* pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int offset = 0;
        if (vz[index] != 0)
        {
            intersectRectangle(ts, pts, rx[index], ry[index], rz[index], Z/2, vx[index], vy[index], vz[index], X, Y, 0, 6, 0, offset);
            intersectRectangle(ts, pts, rx[index], ry[index], rz[index], -Z/2, vx[index], vy[index], vz[index], X, Y, 0, 6, 1, offset);
        }
        else
        {
            ts[index*6] = -1;
            ts[index*6 + 1] = -1;
        }
        if (vx[index] != 0)
        {
            intersectRectangle(ts, pts, ry[index], rz[index], rx[index], X/2, vy[index], vz[index], vx[index], Y, Z, 1, 6, 2, offset);
            intersectRectangle(ts, pts, ry[index], rz[index], rx[index], -X/2, vy[index], vz[index], vx[index], Y, Z, 1, 6, 3, offset);
        }
        else
        {
            ts[index*6 + 2] = -1;
            ts[index*6 + 3] = -1;
        }
        if (vy[index] != 0)
        {
            intersectRectangle(ts, pts, rz[index], rx[index], ry[index], Y/2, vz[index], vx[index], vy[index], Z, X, 2, 6, 4, offset);
            intersectRectangle(ts, pts, rz[index], rx[index], ry[index], -Y/2, vz[index], vx[index], vy[index], Z, X, 2, 6, 5, offset);
        }
        else
        {
            ts[index*6 + 4] = -1;
            ts[index*6 + 5] = -1;
        }
    }
}

__global__ void intersectCylinder(float *rx, float *ry, float *rz,
                                  float *vx, float *vy, float *vz,
                                  const float r, const float h,
                                  const int N, float *ts, float *pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int offset = 0;
        intersectCylinderTopBottom(ts, pts, rx[index], ry[index], rz[index], vx[index], vy[index], vz[index], r, h, offset);
        intersectCylinderSide(ts, pts, rx[index], ry[index], rz[index], vx[index], vy[index], vz[index], r, h, offset);
    }
}

__global__ void intersectPyramid(float *rx, float *ry, float *rz,
                                 float *vx, float *vy, float *vz,
                                 const float X, const float Y, const float H,
                                 const int N, float *ts, float *pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int offset = 0;
        if (vz[index] != 0)
        {
            intersectRectangle(ts, pts, rx[index], ry[index], rz[index], -H, vx[index], vy[index], vz[index], X, Y, 0, 5, 0, offset);
        }
        intersectTriangle(ts, pts,
                          rx[index], ry[index], rz[index],
                          vz[index], vy[index], vz[index],
                          0, 0, 0, X/2, Y/2, -H, X/2, -Y/2, -H,
                          1, offset);
        intersectTriangle(ts, pts,
                          rx[index], ry[index], rz[index],
                          vz[index], vy[index], vz[index],
                          0, 0, 0, X/2, -Y/2, -H, -X/2, -Y/2, -H,
                          2, offset);
        intersectTriangle(ts, pts,
                          rx[index], ry[index], rz[index],
                          vz[index], vy[index], vz[index],
                          0, 0, 0, -X/2, -Y/2, -H, -X/2, Y/2, -H,
                          3, offset);
        intersectTriangle(ts, pts,
                          rx[index], ry[index], rz[index],
                          vz[index], vy[index], vz[index],
                          0, 0, 0, -X/2, Y/2, -H, X/2, Y/2, -H,
                          4, offset);
        //printf("index = %i:\n    ts[%i] = %f ts[%i] = %f ts[%i] = %f ts[%i] = %f ts[%i] = %f\n    rx[%i] = %f ry[%i] = %f rz[%i] = %f\n    vx[%i] = %f vy[%i] = %f vz[%i] = %f\n    pts[%i] = %f pts[%i] = %f pts[%i] = %f\n    pts[%i] = %f pts[%i] = %f pts[%i] = %f\n", index, 5*index, ts[5*index], 5*index+1, ts[5*index+1], 5*index+2, ts[5*index+2], 5*index+3, ts[5*index+3], 5*index+4, ts[5*index+4], index, rx[index], index, ry[index], index, rz[index], index, vx[index], index, vy[index], index, vz[index], 6*index, pts[6*index], 6*index+1, pts[6*index+1], 6*index+2, pts[6*index+2], 6*index+3, pts[6*index+3], 6*index+4, pts[6*index+4], 6*index+5, pts[6*index+5]);
    }
}

__global__ void intersectSphere(float *rx, float *ry, float *rz,
                                float *vx, float *vy, float *vz,
                                const float radius,
                                const int N, float *ts, float *pts)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        /*float rdotv = dot(rx[index], ry[index], rz[index], vx[index], vy[index], vz[index]);
        float rvX, rvY, rvZ;
        cross(rx[index], ry[index], rz[index], vx[index], vy[index], vz[index], &rvX, &rvY, &rvZ);
        float v2 = dot(vx[index], vy[index], vz[index], vx[index], vy[index], vz[index]);
        float rvDot = dot(rvX, rvY, rvZ, rvX, rvY, rvZ);
        float b2m4ac = v2*radius*radius;
        b2m4ac -= rvDot;
        if (b2m4ac < 0)
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }
        float sqrt_b2m4ac = sqrtf(b2m4ac);
        float d[2];
        d[0] = -(rdotv + sqrt_b2m4ac) / v2;
        d[1] = -(rdotv - sqrt_b2m4ac) / v2;
        if (d[0] > d[1])
        {
            int tmp = d[0];
            d[0] = d[1];
            d[1] = tmp;
        }
        float x1, x2, X, Y, Z;
        //float a, b, c, disc;
        float t, t1, t2;*/
        float a = dot(vx[index], vy[index], vz[index],
                      vx[index], vy[index], vz[index]);
        float b = 2 * dot(rx[index], ry[index], rz[index],
                          vx[index], vy[index], vz[index]);
        float c = dot(rx[index], ry[index], rz[index],
                      rx[index], ry[index], rz[index]);
        c -= radius*radius;
        float t0, t1;
        if (!solveQuadratic(a, b, c, t0, t1))
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }
        else
        {
            if (t0 < 0)
            {
                ts[2*index] = -1;
            }
            else
            {
                ts[2*index] = t0;
                pts[6*index] = rx[index] + vx[index] * t0;
                pts[6*index+1] = ry[index] + vy[index] * t0;
                pts[6*index+2] = rz[index] + vz[index] * t0;
            }
            if (t1 < 0)
            {
                ts[2*index+1] = -1;
            }
            else
            {
                ts[2*index + 1] = t1;
                pts[6*index+3] = rx[index] + vx[index] * t1;
                pts[6*index+4] = ry[index] + vy[index] * t1;
                pts[6*index+5] = rz[index] + vz[index] * t1;
            }
        }
        /*for (int i = 0; i < 6; i += 3)
        {
            calculateQuadCoef(rx[index], vx[index], vy[index], vz[index],
                              d[i/3], disc, a, b, c);
            if (disc < 0)
            {
                ts[2*index] = -1;
                ts[2*index + 1] = -1;
                return;
            }
            x1 = (-b + sqrtf(disc))/(2*a);
            x2 = (-b - sqrtf(disc))/(2*a);
            t1 = (x1 - rx[index])/vx[index];
            t2 = (x2 - rx[index])/vx[index];
            if (t1 < 0 && t2 < 0)
            {
                ts[2*index] = -1;
                ts[2*index + 1] = -1;
                return;
            }
            else
            {
                if (t1 < 0 && t2 >= 0)
                {
                    printf("t = t2 = %f X = x2 = %f\n", t2, x2);
                    t = t2;
                    X = x2;
                }
                else if (t1 >= 0 && t2 < 0)
                {
                    printf("t = t1 = %f X = x1 = %f\n", t1, x1);
                    t = t1;
                    X = x1;
                }
                else
                {
                    // This printf is temporary. It will be replaced later.
                    printf("Both times for X were positive or 0.\n    Time 1 = %f Time 2 = %f\n", t1, t2);
                    ts[2*index] = 1789;
                    ts[2*index+1] = 1789;
                    pts[6*index] = 1789;
                    pts[6*index+1] = 1789;
                    pts[6*index+2] = 1789;
                    pts[6*index+3] = 1789;
                    pts[6*index+4] = 1789;
                    pts[6*index+5] = 1789;
                    return;
                }
            }
            __syncthreads();
            Y = ry[index] + t*vy[index];
            Z = rz[index] + t*vz[index];
            ts[2*index + (i/3)] = t;
            pts[6*index + i] = X;
            pts[6*index + i+1] = Y;
            pts[6*index + i+2] = Z;
        }*/
        __syncthreads();
        /*calculateQuadCoef(ry[index], vy[index], vx[index], vz[index],
                          d1, disc, a, b, c);
        if (disc < 0)
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }
        y1 = (-b + sqrtf(disc))/(2*a);
        y2 = (-b - sqrtf(disc))/(2*a);
        t1 = (y1 - ry[index])/vy[index];
        t2 = (y2 - ry[index])/vy[index];
        if ((t1 < 0 && t2 < 0) || (t1 != t && t2 != t));
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }
        calculateQuadCoef(rz[index], vz[index], vx[index], vy[index],
                          d1, disc, a, b, c);
        if (disc < 0)
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }
        z1 = (-b + sqrtf(disc))/(2*a);
        z2 = (-b - sqrtf(disc))/(2*a);
        t1 = (z1 - rz[index])/vz[index];
        t2 = (z2 - rz[index])/vz[index];
        if ((t1 < 0 && t2 < 0) || (t1 != t && t2 != t));
        {
            ts[2*index] = -1;
            ts[2*index + 1] = -1;
            return;
        }*/
    }
}

__global__ void simplifyTimes(const float *times, const int N, const int groupSize, float *simp)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        int count = 0;
        for (int i = 0; i < groupSize; i++)
        {
            if (times[groupSize * index + i] != -1 && count < 2)
            {
                simp[2*index+count] = times[groupSize*index+i];
                count++;
            }
        }
    }
}

__global__ void prepRand(curandState *state, int seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(((seed << 10) + idx), 0, 0, &state[idx]); 
}

__device__ void randCoord(float* inters, float* time , float *sx, float *sy, float *sz, curandState *state)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = time[1] - time[0];
    float mx = (inters[3] - inters[0])/dt;
    float my = (inters[4] - inters[1])/dt;
    float mz = (inters[5] - inters[2])/dt;
    float randt = curand_uniform(&(state[index]));
    randt *= dt;
    *sx = inters[0] + mx*randt;
    *sy = inters[1] + my*randt;
    *sz = inters[2] + mz*randt;
}

__global__ void calcScatteringSites(float* ts, float* int_pts, float* pos, curandState *state, const int N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
        if (ts[2*index] != -5 && ts[2*index+1] != -5)
        {
            if (ts[2*index] > ts[2*index+1])
            {
                float tmpt, tmpc;
                tmpt = ts[2*index];
                ts[2*index] = ts[2*index+1];
                ts[2*index+1] = tmpt;
                for (int i = 6*index; i < 6*index+3; i++)
                {
                    tmpc = int_pts[i];
                    int_pts[i] = int_pts[i + 3];
                    int_pts[i + 3] = tmpc;
                }
            }
            randCoord(&(int_pts[6*index]), &(ts[2*index]), &(pos[3*index + 0]), &(pos[3*index + 1]), &(pos[3*index + 2]), state);
        }
        /*else
        {
            pos[3*index + 0] = 20 * X;
            pos[3*index + 1] = 20 * Y;
            pos[3*index + 2] = 20 * Z;
        }*/
    }
}
