#ifndef BEAM_HPP
#define BEAM_HPP

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "Ray.hpp"
#include "UtilKernels.hpp"

namespace mcvine
{

    namespace gpu
    {

        struct Beam
        {
            Beam(std::vector< ::std::shared_ptr<Ray> > &rays, int size=-1, int bS=512);

            ~Beam();

            void printAllData(const std::string &fname=::std::string());

            void updateRays();

            friend std::ostream& operator<<(::std::ostream &fout, const Beam &beam);

            int N;
    
            Vec3<float> *origins, *vel;
            float *times, *probs;

            Vec3<float> *d_origins, *d_vel;
            float *d_times, *d_probs;

            int numBlocks, blockSize;

            std::vector< ::std::shared_ptr<Ray> > *rayptr;
        };
        
    }

}

#endif
