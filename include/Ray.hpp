#ifndef RAY_HPP
#define RAY_HPP

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
    Ray(double a, double b, double c) { x=a; y=b; z=c; }

    // Alternate constructor that explicitly sets all data.
    Ray(double a, double b, double c,
        double va, double vb, double vc) { x=a; y=b; z=c; vx=va; vy=vb; vz=vc; } 

    // This function provides an easy mechanism to change the velocity values.
    void setVelocities(double a, double b, double c) { vx=a; vy=b; vz=c; }

    // These members store the initial position (origin) of the ray.
    double x, y, z;
    // These members store the velocity data for the ray.
    double vx, vy, vz;
};

#endif
