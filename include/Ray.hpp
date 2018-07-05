#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"

/* A basic struct for a Ray.
 * This struct is used to represent the neutrons on the
 * host (CPU) side.
 */
struct Ray
{
    // Default Constructor
    Ray() { ; }

    /* Alternate constructor that explicitly sets the initial position,
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

    // Alternate constructor that explicitly sets all data.
    Ray(double a, double b, double c,
        double va, double vb, double vc)
    {
        origin[0] = a; origin[1] = b; origin[2] = c;
        vel[0] = va; vel[1] = vb; vel[2] = vc;
        t = 0;
        prob = 1;
    }

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

    void setTime(double time) { t = time; }

    void setProbability(double p) { prob = p; }

    void update(double a, double b, double c, double time)
    {
        origin[0] = a;
        origin[1] = b;
        origin[2] = c;
        t = time;
    }

    void update(double va, double vb, double vc)
    {
        vel[0] = va;
        vel[1] = vb;
        vel[2] = vc;
    }

    void update(double p) { prob = p; }

    void update(double a, double b, double c,
                double va, double vb, double vc,
                double time, double p)
    {
        update(a, b, c, time);
        update(va, vb, vc);
        update(p);
    }

    Vec3<float> origin, vel;

    float t, prob;
};

#endif
