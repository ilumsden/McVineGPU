#ifndef PYRAMID_HPP
#define PYRAMID_HPP

#include "AbstractShape.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace composite
        {

            /* The struct defining the Pyramid primitive.
             * It inherits from AbstractShape to ensure it contains the
             * correct signature for the intersect function.
             */
            struct Pyramid : public AbstractShape
            {
                // Default Constructor
                Pyramid() 
                { 
                    type = "Pyramid"; 
                    data = new float[3];
                    data[0] = 0; data[1] = 0; data[2] = 0;
                }

                /* "Explicit" Constructor
                 * This function takes three floats and sets the base dimensions
                 * and height with them.
                 */
                Pyramid(float X, float Y, float h)
                {
                    type = "Pyramid";
                    data = new float[3];
                    data[0] = X; data[1] = Y; data[2] = h;
                }

                ~Pyramid() { delete [] data; }

                /* The function that handles the calculation of the intersection points
                 * and times between the Pyramid object and the neutrons represented
                 * by d_origins and d_vel when the neutrons start outside the Pyramid.
                 */
                virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                               const int N, const int blockSize, const int numBlocks,
                                               std::vector<float> &int_times, 
                                               std::vector< Vec3<float> > &int_coords) override;

                /* The function that handles the calculation of the intersection points
                 * and times between the Pyramid object and the neutrons represented
                 * by d_origins and d_vel when the neutrons start inside the Pyramid.
                 */
                virtual void interiorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                               const int N, const int blockSize, const int numBlocks,
                                               std::vector<float> &int_times, 
                                               std::vector< Vec3<float> > &int_coords) override;
            };

        }

    }

}

#endif
