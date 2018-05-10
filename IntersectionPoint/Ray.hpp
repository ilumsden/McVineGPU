struct Ray
{
    Ray() { ; }
    //Ray(double a, double b, double c) { x=a; y=b; z=c; }
    // Ray(double va, double vb, double vc) { vx=va; vy=vb; vz=vc; }
    Ray(double a, double b, double c,
        double va, double vb, double vc) { x=a; y=b; z=c; vx=va; vy=vb; vz=vc; } 
    double x, y, z;
    double vx, vy, vz;
};
