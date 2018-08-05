#include "AbstractScatterer.hpp"

namespace mcvine
{

    namespace gpu
    {

        namespace scatter
        {

            void AbstractScatterer::handleExteriorIntersect(std::vector<float> &int_times,
                                                            std::vector< Vec3<float> > &int_coords)
            {
                shape->exteriorIntersect(beam->d_origins, beam->d_vel, beam->N, beam->blockSize, beam->numBlocks, int_times, int_coords);
                // Opens a file stream and prints the relevant data to time.txt
                // NOTE: this is for debugging purposes only. This will be removed later.
#if defined(DEBUG) || defined(PRINT1)
                std::fstream fout;
                fout.open("time.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "time.txt could not be opened.\n";
                    exit(-1);
                }
                for (int i = 0; i < (int)(int_coords.size()); i++)
                {
                    std::string buf = "        ";
                    if (i % 2 == 0)
                    {
                        int ind = i/2;
                        fout << "\n";
                        fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                             << beam->origins[ind][0] << " " << beam->origins[ind][1] << " " << beam->origins[ind][2] << " || "
                             << beam->vel[ind][0] << " " << beam->vel[ind][1] << " " << beam->vel[ind][2] << " | "
                             << int_times[i] << " / " << int_coords[i][0] << "\n";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][1] << "\n";
                        std::string buf = "        ";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][2] << "\n";
                    }
                    else
                    {
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << int_times[i] << " / " << int_coords[i][0] << "\n";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][1] << "\n";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][2] << "\n";
                    }
                }
                // Closes the file stream
                fout.close();
#endif
                return;
            }

            void AbstractScatterer::findScatteringSites(const std::vector<float> &int_times,
                                                        const std::vector< Vec3<float> > &int_coords)
            {
                namespace kernels = mcvine::gpu::kernels;
#if defined(DEBUG) || defined(PRINT2)
                std::vector< Vec3<float> > tmp;
                tmp.resize(beam->N);
                Vec3<float> *ta = tmp.data();
                memcpy(ta, beam->origins, beam->N*sizeof(Vec3<float>));
#endif
                // Stores the size of the `int_times` for later
                int tsize = (int)(int_times.size());
                /* Allocates memory for a device-side array that stores the
                 * data passed in from `int_times`.
                 */
                float *ts;
                CudaErrchk( cudaMalloc(&ts, 2*beam->N*sizeof(float)) );
                CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*beam->N*sizeof(float), cudaMemcpyHostToDevice) );
                /* `pos` is a device-side array that stores the coordinates of the
                 * scattering sites for the neutrons.
                 * The default value of its data is FLT_MAX.
                 */
                Vec3<float> *pos;
                CudaErrchk( cudaMalloc(&pos, beam->N*sizeof(Vec3<float>)) );
                kernels::initArray< Vec3<float> ><<<beam->numBlocks, beam->blockSize>>>(pos, beam->N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
                CudaErrchkNoCode();
                /* `scat_times` is a device_side array that stores the times at
                 * which the neutrons reach their scattering sites.
                 * The default value of its data is -5.
                 */
                float *scat_times;
                CudaErrchk( cudaMalloc(&scat_times, beam->N*sizeof(float)) );
                kernels::initArray<float><<<beam->numBlocks, beam->blockSize>>>(scat_times, beam->N, -5);
                curandGenerator_t gen;
                float *d_randnums;
                CudaErrchk( cudaMalloc(&d_randnums, beam->N*sizeof(float)) );
                CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
                CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
                CuRandErrchk( curandGenerateUniform(gen, d_randnums, beam->N) );
                // Calls the kernel for determining the scattering sites for the neutrons
                mcvine::gpu::kernels::calcScatteringSites<<<beam->numBlocks, beam->blockSize>>>(ts, beam->d_origins, beam->d_vel, pos, scat_times, d_randnums, beam->N);
                /* Propagates the neutrons to their scattering sites.
                 * In other words, the scattering coordinates and times are copied
                 * into the device arrays that store the neutrons' origins and times
                 * (d_origins and d_times respectively).
                 */
                mcvine::gpu::kernels::propagate<<<beam->numBlocks, beam->blockSize>>>(beam->d_origins, beam->d_times, pos, scat_times, beam->N);
                CudaErrchkNoCode();
                /* `ic` is a device-side array that stores the intersection
                 * coordinates between the neutron and scattering body, as calculated
                 * in the handleIntersect function.
                 */
                Vec3<float> *ic;
                CudaErrchk( cudaMalloc(&ic, 2*beam->N*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(ic, int_coords.data(), 2*beam->N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                /* Updates the probability attribute of the neutrons to account
                 * for the absorption that occurs as a neutron travels through the
                 * scattering body to the scattering site.
                 */
                mcvine::gpu::kernels::updateProbability<<<beam->numBlocks, beam->blockSize>>>(beam->d_probs, beam->d_origins, ic, 1, 2, atten, beam->N);
                CudaErrchkNoCode();
                // Updates the host-side arrays for the edited neutron data.
                CudaErrchk( cudaMemcpy(beam->origins, beam->d_origins, beam->N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(beam->times, beam->d_times, beam->N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(beam->probs, beam->d_probs, beam->N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT2)
                std::fstream fout;
                fout.open("scatteringSites.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "scatteringSites.txt could not be opened.\n";
                    exit(-2);
                }
                for (int i = 0; i < beam->N; i++)
                {
                    int ind = 2*i;
                    fout << "\n";
                    fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
                         << beam->vel[i][0] << " " << beam->vel[i][1] << " " << beam->vel[i][2] << " || "
                         << int_times[ind] << " " << int_times[ind+1] << " | "
                         << beam->origins[i][0] << "\n";
                    std::string buf = "        ";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->origins[i][1] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->origins[i][2] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->times[i] << "\n";
                }
                fout.close();
#endif
                CuRandErrchk( curandDestroyGenerator(gen) );
                // Frees the device memory allocated above.
                CudaErrchk( cudaFree(ts) );
                CudaErrchk( cudaFree(ic) );
                CudaErrchk( cudaFree(pos) );
                CudaErrchk( cudaFree(d_randnums) );
                return;
            }

            void AbstractScatterer::handleInteriorIntersect()
            {
#if defined(DEBUG) || defined(PRINT4)
                std::vector< Vec3<float> > tmp;
                tmp.resize(beam->N);
                Vec3<float> *ta = tmp.data();
                memcpy(ta, beam->origins, beam->N*sizeof(Vec3<float>));
#endif
                std::vector<float> int_times;
                std::vector< Vec3<float> > int_coords;
                shape->interiorIntersect(beam->d_origins, beam->d_vel, beam->N, beam->blockSize, beam->numBlocks, int_times, int_coords); 
                float *exit_times;
                CudaErrchk( cudaMalloc(&exit_times, beam->N*sizeof(float)) );
                CudaErrchk( cudaMemcpy(exit_times, int_times.data(), beam->N*sizeof(float), cudaMemcpyHostToDevice) );
                Vec3<float> *exit_coords;
                CudaErrchk( cudaMalloc(&exit_coords, beam->N*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(exit_coords, int_coords.data(), beam->N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                mcvine::gpu::kernels::updateProbability<<<beam->numBlocks, beam->blockSize>>>(beam->d_probs, exit_coords, beam->d_origins, 1, 1, atten, beam->N);
                mcvine::gpu::kernels::propagate<<<beam->numBlocks, beam->blockSize>>>(beam->d_origins, beam->d_times, exit_coords, exit_times, beam->N);
                CudaErrchk( cudaMemcpy(beam->origins, beam->d_origins, beam->N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(beam->times, beam->d_times, beam->N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(beam->probs, beam->d_probs, beam->N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT4)
                std::fstream fout;
                fout.open("exit.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "exit.txt could not be openned.\n";
                    exit(-2);
                }
                for (int i = 0; i < beam->N; i++)
                {
                    std::string buf = "        ";
                    fout << "\n";
                    fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
                         << beam->vel[i][0] << " " << beam->vel[i][1] << " " << beam->vel[i][2] << " | "
                         << beam->origins[i][0] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->origins[i][1] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->origins[i][2] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->times[i] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << beam->probs[i] << "\n";
                }
                fout.close();
#endif
                CudaErrchk( cudaFree(exit_times) );
                CudaErrchk( cudaFree(exit_coords) );
            }

        }

    }

}
