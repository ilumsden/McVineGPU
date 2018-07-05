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
    Ray(double a, double b, double c)
    {
        origin[0] = a;
        origin[1] = b;
        origin[2] = c;
        t = 0;
        prob = 1;
    }

    // Constructor that explicitly sets all data.
    Ray(double a, double b, double c,
        double va, double vb, double vc)
    {
        origin[0] = a; origin[1] = b; origin[2] = c;
        vel[0] = va; vel[1] = vb; vel[2] = vc;
        t = 0;
        prob = 1;
    }

    /* Same as the 6-double constructor, except this one also
     * sets the time and probability.
     */
    Ray(double a, double b, double c,
        double va, double vb, double vc,
        double time, double p)
        : Ray(a, b, c, va, vb, vc)    
    { 
        t = time; 
        prob = p;
    }

    // This function provides an easy mechanism to change the velocity values.
    void setVelocities(double a, double b, double c)
    {
        vel[0] = a; vel[1] = b; vel[2] = c;
    }

    // This function provides an easy mechanism to change the time value.
    void setTime(double time) { t = time; }

    // This function provides an easy mechanism to change the probability value.
    void setProbability(double p) { prob = p; }

    // This function updates position and time data.
    void update(double a, double b, double c, double time)
    {
        origin[0] = a;
        origin[1] = b;
        origin[2] = c;
        t = time;
    }

    // This function is a wrapper of the setVelocities function.
    void update(double va, double vb, double vc)
    {
       setVelocities(va, vb, vc);
    }

    // This function is a wrapper of the setProbability function.
    void update(double p) { setProbability(p); }

    // This function updates all the ray's data.
    void update(double a, double b, double c,
                double va, double vb, double vc,
                double time, double p)
    {
        update(a, b, c, time);
        update(va, vb, vc);
        update(p);
    }

    // This data stores the neutron's position and velocity.
    Vec3<float> origin, vel;

    // This data stores the neutron's time from its initial position
    // and its probability.
    float t, prob;
};

#endif
