#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

/* A basic struct for a Ray.
 * This struct is used to represent the neutrons on the
 * host (CPU) side.
 */
struct Ray
{
    /* Constructor that explicitly sets the initial position,
     * but initializes the velocity with default values.
     */
    Ray(float a, float b, float c)
    {
        origin[0] = a;
        origin[1] = b;
        origin[2] = c;
        t = 0;
        prob = 1;
    }

    // Constructor that explicitly sets all data.
    Ray(float a, float b, float c,
        float va, float vb, float vc)
    {
        origin[0] = a; origin[1] = b; origin[2] = c;
        vel[0] = va; vel[1] = vb; vel[2] = vc;
        t = 0;
        prob = 1;
    }

    /* Same as the 6-float constructor, except this one also
     * sets the time and probability.
     */
    Ray(float a, float b, float c,
        float va, float vb, float vc,
        float time, float p)
        : Ray(a, b, c, va, vb, vc)    
    { 
        t = time; 
        prob = p;
    }

    Ray(Vec3<float> &a, Vec3<float> &b, float time, float p)
    : Ray(a[0], a[1], a[2], b[0], b[1], b[2], time, p) { ; }

    // This function provides an easy mechanism to change the velocity values.
    void setVelocities(float a, float b, float c)
    {
        vel[0] = a; vel[1] = b; vel[2] = c;
    }

    // This function provides an easy mechanism to change the time value.
    void setTime(float time) { t = time; }

    // This function provides an easy mechanism to change the probability value.
    void setProbability(float p) { prob = p; }

    // This function updates position and time data.
    void update(float a, float b, float c, float time)
    {
        origin[0] = a;
        origin[1] = b;
        origin[2] = c;
        t = time;
    }

    // This function is a wrapper of the setVelocities function.
    void update(float va, float vb, float vc)
    {
       setVelocities(va, vb, vc);
    }

    // This function is a wrapper of the setProbability function.
    void update(float p) { setProbability(p); }

    // This function updates all the ray's data.
    void update(float a, float b, float c,
                float va, float vb, float vc,
                float time, float p)
    {
        update(a, b, c, time);
        update(va, vb, vc);
        update(p);
    }

    // This function updates all the ray's data using the Vec3 class.
    void update(Vec3<float> &orig, Vec3<float> &v, float time, float p)
    {
        update(orig[0], orig[1], orig[2], time);
        update(v[0], v[1], v[2]);
        update(p);
    }

    // This data stores the neutron's position and velocity.
    Vec3<float> origin, vel;

    // This data stores the neutron's time from its initial position
    // and its probability.
    float t, prob;
};

#endif
