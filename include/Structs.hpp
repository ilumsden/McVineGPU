struct box
{
    box() { ; }
    box(double a, double b, double c);
    box(double a1, double b1, double c1,
        double a2, double b2, double c2);
    double x, y, z;
    double xmin, xmax, ymin, ymax, zmin, zmax;
};

struct cylinder
{
    cylinder() { ; }
    cylinder(double height, double radius);
    cylinder(double height, double radius,
             double bottomx, double bottomy, double bottomz);
    double h, r;
    double bx, by, bz;
};

struct cone
{
    cone() { ; }
    cone(double topradius, double bottomradius, double height);
    cone(double topradius, double bottomradius, double height,
         double bottomx, double bottomy, double bottomz);
    double h, tr, br;
    double bx, by, bz;
};

struct pyramid
{
    pyramid() { ; }
    pyramid(double height, double X, double Y);
    pyramid(double height, 
            double amin, double amax, double bmin, double bmax);
    pyramid(double height,
            double amin, double amax, double bmin, double bmax
            double cmin, double cmax);
    double h;
    double xmin, xmax, ymin, ymax;
    double zmin, zmax;
};

struct sphere
{
    sphere() { ; }
    sphere(double radius);
    sphere(double radius, double centerx, double centery, double centerz);
    double r;
    double x, y, z;
};
