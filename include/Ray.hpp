#ifndef RAY_HPP
#define RAY_HPP

struct Ray
{
    Ray() { ; }

    Ray(double a, double b, double c) { x=a; y=b; z=c; }

    Ray(double a, double b, double c,
        double va, double vb, double vc) { x=a; y=b; z=c; vx=va; vy=vb; vz=vc; } 

    void setVelocities(double a, double b, double c) { vx=a; vy=b; vz=c; }

    double x, y, z;
    double vx, vy, vz;
};

#endif
