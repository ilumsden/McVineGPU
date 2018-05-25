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
