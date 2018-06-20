#ifndef VECTOR3_HPP
#define VECTOR3_HPP

#include <cmath>
#include <cstdio>
#include <stdexcept>

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
        __host__ __device__ T getX() const { return x; }
        __host__ __device__ T getY() const { return y; }
        __host__ __device__ T getZ() const { return z; }
        /* These three functions are simple "setter" functions
         * that set the values of x, y, and z respectively.
         */
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
        __host__ __device__ T operator|(const Vec3<T> &b) const;
        __host__ __device__ bool operator==(const Vec3<T> &b) const;
        __host__ __device__ bool operator!=(const Vec3<T> &b) const;
        __host__ __device__ const T & operator[](int i) const;
        __host__ __device__ T & operator[](int i);
    private:
        /* x, y, and z reference the three elements of the m_data
         * array. They are references, and, as such, any changes
         * made to x, y, or z will affect the corresponding elements
         * in m_data.
         */
        T &x, &y, &z;
        T m_data[3];
};

/* The main constructor for Vec3.
 * By default, this function will set the members to the 
 * default value of datatype "T". If values are providied, they
 * will be used instead.
 * This function is able to run on both host (CPU) and device (GPU).
 */
template <typename T>
__host__ __device__ Vec3<T>::Vec3(T a=T(), T b=T(), T c=T())
    : x(m_data[0]), y(m_data[1]), z(m_data[2])
{
    x = a;
    y = b;
    z = c;
}

/* The copy constructor for Vec3.
 * This function copies the contents of Vec3 object b
 * into the newly created Vec3 object.
 */
template <typename T>
__host__ __device__ Vec3<T>::Vec3(const Vec3<T> &b)
    : x(m_data[0]), y(m_data[1]), z(m_data[2])
{
    x = b[0];
    y = b[1];
    z = b[2];
}

/* This function is an overloaded assignment operator to
 * allowed for copying of a Vec3 object to another existing
 * Vec3 object.
 */
template <typename T>
__host__ __device__ const Vec3<T> & operator=(const Vec3<T> &b)
{
    x = b[0];
    y = b[1];
    z = b[2];
    return *this;
}

/* This function normalizes the vector.
 * All that it does is multiplies each element in the Vec3
 * object by 1/length, where length is the length of the vector.
 */
template <typename T>
__host__ __device__ void Vec3<T>::normalize()
{
    *this *= (1.0 / this->length());
}

// This function calculates and returns the length of the vector.
template <typename T>
__host__ __device__ T Vec3<T>::length() const
{
    return std::sqrt( std::abs(x)*std::abs(x) + std::abs(y)*std::abs(y) + std::abs(z)*std::abs(z) );
}

/* This function performs element-wise addition on two vectors and 
 * stores the result in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator+(const Vec3<T> &b) const
{
    return Vec3<T>(x+b[0], y+b[1], z+b[2]);
}

/* This function performs element-wise addition on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator+=(const Vec3<T> &b)
{
    x += b[0];
    y += b[1];
    z += b[2];
}

/* This function performs element-wise substraction on two vectors
 * and stores the result in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator-(const Vec3<T> &b) const
{
    return Vec3<T>(x-b[0], y-b[1], z-b[2]);
}

/* This function negates each component of the vector and
 * stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator-() const
{
    return Vec3<T>(-x, -y, -z);
}

/* This function performs element-wise subtraction on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator-=(const Vec3<T> &b) const
{
    x -= b[0];
    y -= b[1];
    z -= b[2];
}

/* This function performs a scalar product on the vector and
 * stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator*(const T n) const
{
    return Vec3<T>(n*x, n*y, n*z);
}

/* This function performs a vector (cross) product on two vectors
 * and stores the resulting vector in a new Vec3 object, which is returned.
 */
template <typename T>
__host__ __device__ Vec3<T> Vec3<T>::operator*(const Vec3<T> &b) const
{
    return Vec3<T>(y*b[2] - z*b[1],
                   z*b[0] - x*b[2],
                   x*b[1] - y*b[0]);
}

/* This function performs a scalar product on the vector and
 * stores the resulting vector in the vector used for the operation.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator*=(const T n)
{
    x *= n;
    y *= n;
    z *= n;
}

/* This function performs a vector (cross) product on two vectors
 * and stores the result in the Vec3 object on the left of the operator.
 */
template <typename T>
__host__ __device__ void Vec3<T>::operator*=(const Vec3<T> &b)
{
    x = y*b[2] - z*b[1];
    y = z*b[0] - x*b[2];
    z = x*b[1] - y*b[0];
}

/* This function calculates the dot product of two vectors and
 * returns the result.
 */
template <typename T>
__host__ __device__ T Vec3<T>::operator|(const Vec3<T> &b) const
{
    return x*b[0] + y*b[1] + z*b[2];
}

/* This function is a basic comparison operator that returns true
 * if the two vectors are equal (components are equal) and false
 * otherwise.
 */
template <typename T>
__host__ __device__ bool Vec3<T>::operator==(const Vec3<T> &b) const
{
    return (x == b[0]) && (y == b[1]) && (z == b[2]);
}

/* This function is a basic comparison operator that returns true
 * if the two vectors are not equal (components are not equal) and
 * false otherwise.
 */
template <typename T>
__host__ __device__ bool Vec3<T>::operator!=(const Vec3<T> &b) const
{
    return (x != b[0]) || (y != b[1]) || (z != b[2]);
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
    if (i < 3) { return m_data[i]; }
    else
    {
        fprintf(stderr, "Vec3: Index %i out of bounds.\n", i);
/* If the error occurs on the GPU, the PTX assembly command "trap"
 * will be injected, causing the GPU code to immediately end.
 * This will also cause the kernel to return a cudaErrorUnknown
 * (errcode 30).
 */
#if defined(__CUDA_ARCH__)
        __threadfence();
        asm("trap;");
// If the error occurs on the CPU, an out_of_range exception is thrown.
#else
        throw std::out_of_range("out of bounds");
#endif
    }
}

/* This function is the same as the array lookup overload above,
 * except that it returns a NON-CONST reference to the data accessed.
 */
template <typename T>
__host__ __device__ T & Vec3<T>::operator[](int i)
{
    if (i < 3) { return m_data[i]; }
    else
    {
        fprintf(stderr, "Vec3: Index %i out of bounds.\n", i);
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::out_of_range("out of bounds");
#endif
    }
}

#endif
