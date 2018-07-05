#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "SystemVars.hpp"

/* The Vec3 class is a templated class that represents
 * a three-dimensional vector and the operations that can
 * be applied on said vector.
 */
template <typename T>
class Vec3
{
    public:
        __host__ __device__ Vec3(T a=T(), T b=T(), T c=T());
        __host__ __device__ Vec3(const Vec3<T> &b);
        __host__ __device__ const Vec3<T> & operator=(const Vec3<T> &b);
        // Default destructor for Vec3
        __host__ __device__ ~Vec3() { ; }
        // Returns the size of the vector (always 3)
        __host__ __device__ std::size_t size() { return 3; }
        /* These three functions are simple "getter" functions
         * that return the values of x, y, and z respectively.
         * They return the values by-value, so they should
         * ideally only be used for simpler types (i.e int,
         * float, double, etc.).
         */
        __host__ __device__ T getX() const { return m_data[0]; }
        __host__ __device__ T getY() const { return m_data[1]; }
        __host__ __device__ T getZ() const { return m_data[2]; }
        /* These three functions are simple "setter" functions
         * that set the values of x, y, and z respectively.
         */
        __host__ __device__ void setX(T a) { m_data[0] = a; }
        __host__ __device__ void setY(T b) { m_data[1] = b; }
        __host__ __device__ void setZ(T c) { m_data[2] = c; }
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
        __host__ __device__ T operator|(const Vec3<T> &b) const;
        __host__ __device__ bool operator==(const Vec3<T> &b) const;
        __host__ __device__ bool operator!=(const Vec3<T> &b) const;
        __host__ __device__ const T & operator[](int i) const;
        __host__ __device__ T & operator[](int i);
    private:
        T m_data[3];
};

/* The main constructor for Vec3.
 * By default, this function will set the members to the 
 * default value of datatype "T". If values are providied, they
 * will be used instead.
 * This function is able to run on both host (CPU) and device (GPU).
 */
template <typename T>
__host__ __device__ Vec3<T>::Vec3(T a, T b, T c)
{
    m_data[0] = a;
    m_data[1] = b;
    m_data[2] = c;
}

/* The copy constructor for Vec3.
 * This function copies the contents of Vec3 object b
 * into the newly created Vec3 object.
 */
template <typename T>
__host__ __device__ Vec3<T>::Vec3(const Vec3<T> &b)
{
    m_data[0] = b[0];
    m_data[1] = b[1];
    m_data[2] = b[2];
}

/* This function is an overloaded assignment operator to
 * allowed for copying of a Vec3 object to another existing
 * Vec3 object.
 */
template <typename T>
__host__ __device__ const Vec3<T> & Vec3<T>::operator=(const Vec3<T> &b)
{
    m_data[0] = b[0];
    m_data[1] = b[1];
    m_data[2] = b[2];
    return *this;
}

/* This function normalizes the vector.
 * All that it does is multiplies each element in the Vec3
 * object by 1/length, where length is the length of the vector.
 */
template <typename T>
__host__ __device__ void Vec3<T>::normalize()
{
    *this *= (1.0 / length());
    printf("x = %f\ny = %f\nz = %f\n", m_data[0], m_data[1], m_data[2]);
}

// This function calculates and returns the length of the vector.
template <typename T>
__host__ __device__ T Vec3<T>::length() const
{
#if defined(__CUDA_ARCH__)
    return sqrt( fabs(m_data[0])*fabs(m_data[0]) + fabs(m_data[1])*fabs(m_data[1]) + fabs(m_data[2])*fabs(m_data[2]) );
#else
    return std::sqrt( std::abs(m_data[0])*std::abs(m_data[0]) + std::abs(m_data[1])*std::abs(m_data[1]) + std::abs(m_data[2])*std::abs(m_data[2]) );
#endif
}

/* This function performs element-wise addition on two vectors and 
 * stores the result in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator+(const Vec3<T> &b) const
{
    return Vec3<T>(m_data[0]+b[0], m_data[1]+b[1], m_data[2]+b[2]);
}

/* This function performs element-wise addition on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator+=(const Vec3<T> &b)
{
    m_data[0] += b[0];
    m_data[1] += b[1];
    m_data[2] += b[2];
}

/* This function performs element-wise substraction on two vectors
 * and stores the result in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator-(const Vec3<T> &b) const
{
    return Vec3<T>(m_data[0]-b[0], m_data[1]-b[1], m_data[2]-b[2]);
}

/* This function negates each component of the vector and
 * stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator-() const
{
    return Vec3<T>(-m_data[0], -m_data[1], -m_data[2]);
}

/* This function performs element-wise subtraction on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator-=(const Vec3<T> &b)
{
    m_data[0] -= b[0];
    m_data[1] -= b[1];
    m_data[2] -= b[2];
}

/* This function performs a scalar product on the vector and
 * stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator*(const T n) const
{
    return Vec3<T>(n*m_data[0], n*m_data[1], n*m_data[2]);
}

/* This function performs a vector (cross) product on two vectors
 * and stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator*(const Vec3<T> &b) const
{
    return Vec3<T>(m_data[1]*b[2] - m_data[2]*b[1],
                   m_data[2]*b[0] - m_data[0]*b[2],
                   m_data[0]*b[1] - m_data[1]*b[0]);
}

/* This function performs a scalar product on the vector and
 * stores the resulting vector in the vector used for the operation.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator*=(const T n)
{
    m_data[0] *= n;
    m_data[1] *= n;
    m_data[2] *= n;
}

/* This function performs a vector (cross) product on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator*=(const Vec3<T> &b)
{
    float tmp[3];
    for (int i = 0; i < 3; i++)
    {
        tmp[i] = m_data[i];
    }
    m_data[0] = tmp[1]*b[2] - tmp[2]*b[1];
    m_data[1] = tmp[2]*b[0] - tmp[0]*b[2];
    m_data[2] = tmp[0]*b[1] - tmp[1]*b[0];
}

/* This function calculates the dot product of two vectors and
 * returns the result.
 */
template <typename T>
__host__ __device__ T Vec3<T>::operator|(const Vec3<T> &b) const
{
    return m_data[0]*b[0] + m_data[1]*b[1] + m_data[2]*b[2];
}

/* This function is a basic comparison operator that returns true
 * if the two vectors are equal (components are equal) and false
 * otherwise.
 */
template <typename T>
__host__ __device__ bool Vec3<T>::operator==(const Vec3<T> &b) const
{
    return (m_data[0] == b[0]) && (m_data[1] == b[1]) && (m_data[2] == b[2]);
}

/* This function is a basic comparison operator that returns true
 * if the two vectors are not equal (components are not equal) and
 * false otherwise.
 */
template <typename T>
__host__ __device__ bool Vec3<T>::operator!=(const Vec3<T> &b) const
{
    return (m_data[0] != b[0]) || (m_data[1] != b[1]) || (m_data[2] != b[2]);
}

/* This function is an alternate "getter" function that uses array
 * lookup notation. It returns a CONST reference to the data accessed.
 * If the index is out-of-bounds, an error message will be printed.
 * How the program exits depends on whether the error occured on the
 * CPU or GPU.
 */
template <typename T>
__host__ __device__ const T & Vec3<T>::operator[](int i) const
{
    if (i >= 3)
    {
/* If the error occurs on the GPU, the PTX assembly command "trap"
 * will be injected, causing the GPU code to immediately end.
 * This will also cause the kernel to return a cudaErrorUnknown
 * (errcode 30).
 */
#if defined(__CUDA_ARCH__)
        printf("Vec3: Index %i out of bounds.\n", i);
        __threadfence();
        asm("trap;");
// If the error occurs on the CPU, an out_of_range exception is thrown.
#else
        fprintf(stderr, "Vec3: Index %i out of bounds.\n", i);
        throw std::out_of_range("out of bounds");
#endif
    }
    return m_data[i];
}

/* This function is the same as the array lookup overload above,
 * except that it returns a NON-CONST reference to the data accessed.
 */
template <typename T>
__host__ __device__ T & Vec3<T>::operator[](int i)
{
    if (i >= 3)
    {
#if defined(__CUDA_ARCH__)
        printf("Vec3: Index %i out of bounds.\n", i);
        __threadfence();
        asm("trap;");
#else
        fprintf(stderr, "Vec3: Index %i out of bounds.\n", i);
        throw std::out_of_range("out of bounds");
#endif
    }
    return m_data[i];
}

#endif
