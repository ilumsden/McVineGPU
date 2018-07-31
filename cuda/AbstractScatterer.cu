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
                b->exteriorIntersect(d_origins, d_vel, N, blockSize, numBlocks, host_time, int_coords);
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
                             << origins[ind][0] << " " << origins[ind][1] << " " << origins[ind][2] << " || "
                             << vel[ind][0] << " " << vel[ind][1] << " " << vel[ind][2] << " | "
                             << host_time[i] << " / " << int_coords[i][0] << "\n";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][1] << "\n";
                        std::string buf = "        ";
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << buf << " / " << int_coords[i][2] << "\n";
                    }
                    else
                    {
                        fout << buf << " " << buf << " " << buf << "  " << buf << " " << buf << " " << buf << " | "
                             << std::fixed << std::setprecision(5) << std::setw(8) << std::right << host_time[i] << " / " << int_coords[i][0] << "\n";
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
#if defined(DEBUG) || defined(PRINT2)
                std::vector< Vec3<float> > tmp;
                tmp.resize(N);
                Vec3<float> *ta = tmp.data();
                memcpy(ta, origins, N*sizeof(Vec3<float>));
#endif
                // Stores the size of the `int_times` for later
                int tsize = (int)(int_times.size());
                /* Allocates memory for a device-side array that stores the
                 * data passed in from `int_times`.
                 */
                float *ts;
                CudaErrchk( cudaMalloc(&ts, 2*N*sizeof(float)) );
                CudaErrchk( cudaMemcpy(ts, int_times.data(), 2*N*sizeof(float), cudaMemcpyHostToDevice) );
                /* `pos` is a device-side array that stores the coordinates of the
                 * scattering sites for the neutrons.
                 * The default value of its data is FLT_MAX.
                 */
                Vec3<float> *pos;
                CudaErrchk( cudaMalloc(&pos, N*sizeof(Vec3<float>)) );
                initArray< Vec3<float> ><<<numBlocks, blockSize>>>(pos, N, Vec3<float>(FLT_MAX, FLT_MAX, FLT_MAX));
                CudaErrchkNoCode();
                /* `scat_times` is a device_side array that stores the times at
                 * which the neutrons reach their scattering sites.
                 * The default value of its data is -5.
                 */
                float *scat_times;
                CudaErrchk( cudaMalloc(&scat_times, N*sizeof(float)) );
                initArray<float><<<numBlocks, blockSize>>>(scat_times, N, -5);
                curandGenerator_t gen;
                float *d_randnums;
                CudaErrchk( cudaMalloc(&d_randnums, N*sizeof(float)) );
                CuRandErrchk( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
                CuRandErrchk( curandSetPseudoRandomGeneratorSeed(gen, time(NULL)) );
                CuRandErrchk( curandGenerateUniform(gen, d_randnums, N) );
                // Calls the kernel for determining the scattering sites for the neutrons
                calcScatteringSites<<<numBlocks, blockSize>>>(ts, d_origins, d_vel, pos, scat_times, d_randnums, N);
                /* Propagates the neutrons to their scattering sites.
                 * In other words, the scattering coordinates and times are copied
                 * into the device arrays that store the neutrons' origins and times
                 * (d_origins and d_times respectively).
                 */
                propagate<<<numBlocks, blockSize>>>(d_origins, d_times, pos, scat_times, N);
                CudaErrchkNoCode();
                /* `ic` is a device-side array that stores the intersection
                 * coordinates between the neutron and scattering body, as calculated
                 * in the handleIntersect function.
                 */
                Vec3<float> *ic;
                CudaErrchk( cudaMalloc(&ic, 2*N*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(ic, int_coords.data(), 2*N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                /* Updates the probability attribute of the neutrons to account
                 * for the absorption that occurs as a neutron travels through the
                 * scattering body to the scattering site.
                 */
                updateProbability<<<numBlocks, blockSize>>>(d_probs, d_origins, ic, 1, 2, atten, N);
                CudaErrchkNoCode();
                // Updates the host-side arrays for the edited neutron data.
                CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT2)
                std::fstream fout;
                fout.open("scatteringSites.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "scatteringSites.txt could not be opened.\n";
                    exit(-2);
                }
                for (int i = 0; i < N; i++)
                {
                    int ind = 2*i;
                    fout << "\n";
                    fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
                         << vel[i][0] << " " << vel[i][1] << " " << vel[i][2] << " || "
                         << int_times[ind] << " " << int_times[ind+1] << " | "
                         << origins[i][0] << "\n";
                    std::string buf = "        ";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << origins[i][1] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << origins[i][2] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << times[i] << "\n";
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
                tmp.resize(N);
                Vec3<float> *ta = tmp.data();
                memcpy(ta, origins, N*sizeof(Vec3<float>));
#endif
                std::vector<float> int_times;
                std::vector< Vec3<float> > int_coords;
                b->interiorIntersect(d_origins, d_vel, N, blockSize, numBlocks, int_times, int_coords); 
                float *exit_times;
                CudaErrchk( cudaMalloc(&exit_times, N*sizeof(float)) );
                CudaErrchk( cudaMemcpy(exit_times, int_times.data(), N*sizeof(float), cudaMemcpyHostToDevice) );
                Vec3<float> *exit_coords;
                CudaErrchk( cudaMalloc(&exit_coords, N*sizeof(Vec3<float>)) );
                CudaErrchk( cudaMemcpy(exit_coords, int_coords.data(), N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                updateProbability<<<numBlocks, blockSize>>>(d_probs, exit_coords, d_origins, 1, 1, atten, N);
                propagate<<<numBlocks, blockSize>>>(d_origins, d_times, exit_coords, exit_times, N);
                CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
#if defined(DEBUG) || defined(PRINT4)
                std::fstream fout;
                fout.open("exit.txt", std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << "exit.txt could not be openned.\n";
                    exit(-2);
                }
                for (int i = 0; i < N; i++)
                {
                    std::string buf = "        ";
                    fout << "\n";
                    fout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << tmp[i][0] << " " << tmp[i][1] << " " << tmp[i][2] << " || "
                         << vel[i][0] << " " << vel[i][1] << " " << vel[i][2] << " | "
                         << origins[i][0] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << origins[i][1] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << origins[i][2] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << times[i] << "\n";
                    fout << buf << " " << buf << " " << buf << "    "
                         << buf << " " << buf << " " << buf << "   "
                         << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                         << probs[i] << "\n";
                }
                fout.close();
#endif
                CudaErrchk( cudaFree(exit_times) );
                CudaErrchk( cudaFree(exit_coords) );
            }

        }

    }

}
