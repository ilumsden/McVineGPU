#include "Structs.hpp"

box::box(double a, double b, double c)
{
    x = a;
    y = b;
    z = c;
    xmin = -(a/2);
    xmax = a/2;
    ymin = -(b/2);
    ymax = b/2;
    zmin = -(c/2);
    zmax = c/2;
}

box::box(double amin, double bmin, double cmin,
         double amax, double bmax, double cmax)
{
    xmin = amin;
    ymin = bmin;
    zmin = cmin;
    xmax = amax;
    ymax = bmax;
    zmax = cmax;
    x = amax - amin;
    y = bmax - bmin;
    z = cmax - cmin;
}

cylinder::cylinder(double height, double radius)
{
    h = height;
    r = radius;
    bx = 0;
    by = 0;
    bz = -(height/2);
}

cylinder::cylinder(double height, double radius,
                   double bottomx, double bottomy, double bottomz)
{
    h = height;
    r = radius;
    bx = bottomx;
    by = bottomy;
    bz = bottomz;
}

cone::cone(double topradius, double bottomradius, double height)
{
    h = height;
    tr = topradius;
    br = bottomradius;
    bx = 0;
    by = 0;
    bz = -(height/2);
}

cone::cone(double topradius, double bottomradius, double height,
           double bottomx, double bottomy, double bottomz)
{
    h = height;
    tr = topradius;
    br = bottomradius;
    bx = bottomx;
    by = bottomy;
    bz = bottomz;
}

pyramid::pyramid(double height, double X, double Y)
{
    h = height;
    xmin = -(X/2);
    xmax = X/2;
    ymin = -(Y/2);
    ymax = Y/2;
    zmin = -height;
    zmax = 0;
}

pyramid::pyramid(double height,
                 double amin, double amax, double bmin, double bmax)
{
    h = height;
    xmin = amin;
    xmax = amax;
    ymin = bmin;
    ymax = bmax;
    zmin = -height;
    zmax = 0;
}

pyramid::pyramid(double height, 
                 double amin, double amax, double bmin, double bmax,
                 double cmin, double cmax)
{
    h = height; 
    xmin = amin;
    xmax = amax;
    ymin = bmin;
    ymax = bmax;
    zmin = cmin;
    zmax = cmax;
}

sphere::sphere(double radius)
{
    r = radius;
    x = 0;
    y = 0;
    z = 0;
}

sphere::sphere(double radius, double centerx, double centery, double centerz)
{
    r = radius;
    x = centerx;
    y = centery;
    z = centerz;
}
