#ifndef BEAM_HPP
#define BEAM_HPP

#include <cstdlib>
#include <memory>
#include <vector>

#include "Ray.hpp"
#include "UtilKernels.hpp"

namespace mcvine
{

    namespace gpu
    {

        struct Beam
        {
            Beam(std::vector< std::shared_ptr<Ray> > &rays, int size=-1, int bS=512);

            ~Beam();

            int N;
    
            Vec3<float> *origins, *vel;
            float *times, *probs;

            Vec3<float> *d_origins, *d_vel;
            float *d_times, *d_probs;

            int numBlocks, blockSize;
        };
        
    }

}

#endif
