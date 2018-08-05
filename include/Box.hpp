#ifndef BOX_HPP
#define BOX_HPP

#include "AbstractShape.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace composite
        {

            /* The struct defining the Box primitive.
             * It inherits from AbstractShape to ensure it contains the
             * correct signature for the intersect function.
             */
            struct Box : public AbstractShape
            {
                // Default Constructor
                Box() 
                { 
                    type = "Box"; 
                    data = new float[3];
                    data[0] = 0; data[1] = 0; data[2] = 0;
                }

                /* "Explicit" Constructor.
                 * This function takes three floats and sets the side lengths
                 * with them.
                 */
                Box(const float a, const float b, const float c)
                {
                    type = "Box";
                    data = new float[3];
                    data[0] = a; data[1] = b; data[2] = c;
                }

                ~Box() { delete [] data; }
                
                /* The function that handles the calculation of the intersection
                 * points and times between the Box object and the neutrons represented
                 * by d_origins and d_vel when the neutrons start outside the Box.
                 */
                virtual void exteriorIntersect(Vec3<float> *d_origins, Vec3<float> *d_vel,
                                               const int N, const int blockSize, const int numBlocks,
                                               std::vector<float> &int_times,
                                               std::vector< Vec3<float> > &int_coords) override;

                /* The function that handles the calculation of the intersection
                 * points and times between the Box object and the neutrons represented
                 * by d_origins and d_vel when the neutrons start inside the Box.
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
