#include "ConstantQEScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace scatter
        {

            void ConstantQEScatterer::scatter()
            {
                std::vector<float> int_times;
                std::vector< Vec3<float> > int_coords;
                // Starts the intersection calculation
                handleExteriorIntersect(int_times, int_coords);
                // Starts the scattering site calculation
                findScatteringSites(int_times, int_coords);
                // Starts the elastic scattering calculation
                findScatteringVels();
                handleInteriorIntersect();
            }

            void ConstantQEScatterer::findScatteringVels()
            {
#if defined(DEBUG) || defined(PRINT3)
                std::vector< Vec3<float> > tmp;
                tmp.resize(beam->N);
                Vec3<float> *ta = tmp.data();
                memcpy(ta, beam->vel, beam->N*sizeof(Vec3<float>));
#endif
#if defined(DEBUG) || defined(RANDTEST)
                std::vector<float> thetas, phis;
#endif
                curandGenerator_t gen;
                float *d_randnums;
                CudaErrchk( cudaMalloc(&d_randnums, beam->N*sizeof(float)) );
                CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
                CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
                CuRandErrchk( curandGenerateUniform(gen, d_randnums, beam->N) );
                const float *QE;
                CudaErrchk( cudaMalloc(&QE, 2) );
                CudaErrchk( cudaMemcpy(&QE[0], &Q, sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(&QE[1], &E, sizeof(float), cudaMemcpyHostToDevice) );
                /* Calls the elasticScatteringKernel function to update the neutron
                 * velocities post-elastic scattering.
                 */
                mcvine::gpu::kernels::scatter<<<beam->numBlocks, beam->blockSize>>>(type, beam->d_times, beam->d_vel, d_randnums, QE, beam->N);
                CudaErrchkNoCode();
                /* Copies the new neutron velocities into the host-side neutron
                 * velocity array.
                 */
                CudaErrchk( cudaMemcpy(beam->vel, beam->d_vel, beam->N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                // Opens a file stream and prints the 
                // relevant data to scatteringVels.txt
#if defined(DEBUG) || defined(PRINT3)
                std::fstream fout;
                fout.open("scatteringVels.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "scatteringVels.txt could not be opened.\n";
                    exit(-2);
                }
                for (int i = 0; i < beam->N; i++)
                {
                    fout << "\n";
                    fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
                         << beam->vel[i][0] << " " << beam->vel[i][1] << " " << beam->vel[i][2] << "\n";
                }
                fout.close();
#endif
#if defined(DEBUG) || defined(RANDTEST)
                for (int i = 0; i < beam->N; i++)
                {
                    thetas.push_back(acos(beam->vel[i][2] / beam->vel[i].length()));
                    phis.push_back(atan2(beam->vel[i][1], beam->vel[i][0]));
                }
                std::sort(thetas.begin(), thetas.end());
                std::sort(phis.begin(), phis.end());
                std::fstream f1, f2;
                f1.open("thetas.txt", std::ios::out);
                if (!f1.is_open())
                {
                    std::cerr << "thetas.txt could not be openned.\n";
                    exit(-2);
                }
                f2.open("phis.txt", std::ios::out);
                if (!f2.is_open())
                {
                    std::cerr << "phis.txt could not be openned.\n";
                    exit(-2);
                }
                f1 << "Theta Values (Radians): Should range from 0 to Pi\n";
                f2 << "Phi Values (Radians): Should range from 0 to 2*Pi\n";
                for (int i = 0; i < (int)(thetas.size()); i++)
                {
                    f1 << thetas[i] << "\n";
                    f2 << phis[i] << "\n";
                }
                f1.close();
                f2.close();
#endif
                CuRandErrchk( curandDestroyGenerator(gen) );
                CudaErrchk( cudaFree(d_randnums) );
            }

        }

    }

}
