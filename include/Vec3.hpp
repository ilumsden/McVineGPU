#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <cmath>

template <typename T>
class Vec3
{
    public:
        Vec3(T a=T(), T b=T(), T c=T()) : x(a), y(b), z(c) { ; }
        ~Vec3() { ; }
        __host__ __device__ std::size_t size() { return 3; }
        __host__ __device__ T getX() const { return x; }
        __host__ __device__ T getY() const { return y; }
        __host__ __device__ T getZ() const { return z; }
        __host__ __device__ void setX(T a) { x = a; }
        __host__ __device__ void setY(T b) { y = b; }
        __host__ __device__ void setZ(T c) { z = c; }
        __host__ __device__ void normalize();
        __host__ __device__ T length() const;
        __host__ __device__ Vec3<T> operator+(const Vec3<T> &b) const;
        __host__ __device__ void operator+=(const Vec3<T> &b);
        __host__ __device__ Vec3<T> operator-(const Vec3<T> &b) const;
        __host__ __device__ Vec3<T> operator-() const;
        __host__ __device__ void operator-=(const Vec3<T> &b);
        __host__ __device__ Vec3<T> operator*(const T n) const;
        __host__ __device__ Vec3<T> operator*(const Vec3<T> &b) const;
        __host__ __device__ void operator*=(const T n);
        __host__ __device__ void operator*=(const Vec3<T> &b);
        __host__ __device__ const T & operator[](int i) const;
        __host__ __device__ T & operator[](int i);
        __host__ __device__ T operator|(const Vec3<T> &b) const;
    private:
        T x, y, z;
};

template <typename T>
__host__ __device__ void Vec3::normalize()
{
    *this *= (1.0 / this->length());
}

template <typename T>
__host__ __device__ T Vec3::length() const
{
    return std::sqrt( std::abs(x)*std::abs(x) + std::abs(y)*std::abs(y) + std::abs(z)*std::abs(z) );
}

template <typename T>
__host__ __device__ Vec3<T> Vec3::operator+(const Vec3<T> &b) const
{
    return Vec3<T>(x+b.getX(), y+b.getY(), z+b.getZ());
}

template <typename T>
__host__ __device__ void Vec3::operator+=(const Vec3<T> &b)
{
    x += b.getX();
    y += b.getY();
    z += b.getZ();
}

template <typename T>
__host__ __device__ Vec3<T> Vec3::operator-(const Vec3<T> &b) const
{
    return Vec3<T>(x-b.getX(), y-b.getY(), z-b.getZ());
}

template <typename T>
__host__ __device__ void Vec3::operator-=(const Vec3<T> &b) const
{
    x -= b.getX();
    y -= b.getY();
    z -= b.getZ();
}

#endif
