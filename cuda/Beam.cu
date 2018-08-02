#include "Beam.hpp"
#include "Error.hpp"

#include <chrono>

namespace mcvine
{

    namespace gpu
    {

        Beam::Beam(std::vector< std::shared_ptr<Ray> > &rays, int size, int bS)
        {
            blockSize = bS;
            rayptr = &rays;
            if (rays.size() == 1)
            {
                N = size;
                if (size == -1)
                {
                    printf("Warning: Current number of neutrons is set to 1. This will run very inefficiently on GPU. It is recommended that you use the original MCViNE for this task.\n\nWould you like to continue? (y/n)\n");
                    char neg = 'n';
                    char pos = 'y';
                    while (1)
                    {
                        int response = getchar();
                        if (response == (int)(neg))
                        {
                            printf("Exiting program\n");
                            exit(-3);
                        }
                        else if (response == (int)(pos))
                        {
                            N = 1;
                            break;
                        }
                        else
                        {
                            printf("Unknown command. Would you like to continue? (y/n)\n");
                        }
                    }
                }
                numBlocks = (N + blockSize - 1) / blockSize;
                origins = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
                CudaErrchk( cudaMalloc(&d_origins, N*sizeof(Vec3<float>)) );
                vel = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
                CudaErrchk( cudaMalloc(&d_vel, N*sizeof(Vec3<float>)) );
                times = (float*)malloc(N*sizeof(float));
                CudaErrchk( cudaMalloc(&d_times, N*sizeof(float)) );
                probs = (float*)malloc(N*sizeof(float));
                CudaErrchk( cudaMalloc(&d_probs, N*sizeof(float)) );
                auto start = std::chrono::steady_clock::now();

                namespace kernels = mcvine::gpu::kernels;

                kernels::initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_origins, N, rays[0]->origin);
                kernels::initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_vel, N, rays[0]->vel);
                kernels::initArray<float><<<numBlocks, blockSize>>>(d_times, N, rays[0]->t);
                kernels::initArray<float><<<numBlocks, blockSize>>>(d_probs, N, rays[0]->prob);
                CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(vel, d_vel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
                auto stop = std::chrono::steady_clock::now();
                double time = std::chrono::duration<double>(stop-start).count();
                printf("Data Gen Time = %f s\n", time);
            }
            else
            {
                N = (int)(rays.size());
                numBlocks = (N + blockSize - 1) / blockSize;
                origins = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
                CudaErrchk( cudaMalloc(&d_origins, N*sizeof(Vec3<float>)) );
                vel = (Vec3<float>*)malloc(N*sizeof(Vec3<float>));
                CudaErrchk( cudaMalloc(&d_vel, N*sizeof(Vec3<float>)) );
                times = (float*)malloc(N*sizeof(float));
                CudaErrchk( cudaMalloc(&d_times, N*sizeof(float)) );
                probs = (float*)malloc(N*sizeof(float));
                CudaErrchk( cudaMalloc(&d_probs, N*sizeof(float)) );
                // Copies the data from the rays to the host arrays.
                int c = 0;
                for (auto ray : rays)
                {
                    origins[c] = ray->origin;
                    vel[c] = ray->vel;
                    times[c] = ray->t;
                    probs[c] = ray->prob;
                    c++;
                }
                // Copies the data from the host arrays to the device arrays.
                CudaErrchk( cudaMemcpy(d_origins, origins, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_vel, vel, N*sizeof(Vec3<float>), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_times, times, N*sizeof(float), cudaMemcpyHostToDevice) );
                CudaErrchk( cudaMemcpy(d_probs, probs, N*sizeof(float), cudaMemcpyHostToDevice) );
            }
        }

        Beam::~Beam()
        {
            free(origins); 
            free(vel);
            free(times);
            free(probs);
            CudaErrchk( cudaFree(d_origins) );
            CudaErrchk( cudaFree(d_vel) );
            CudaErrchk( cudaFree(d_times) );
            CudaErrchk( cudaFree(d_probs) );
        }

        void Beam::printAllData(const std::string &fname)
        {
            /* If there is a file name provided (i.e. fname != std::string()),
             * the C++ stdout stream (cout) is redirected to print to the
             * desired file. Otherwise, all data is printed to stdout.
             */
            std::streambuf *coutbuf = std::cout.rdbuf();
            std::fstream fout;
            if (fname != std::string())
            {
                fout.open(fname.c_str(), std::ios::out);
                if (!fout.is_open())
                {
                    std::cerr << fname << " cannot be openned.\n";
                    exit(-2);
                }
                std::cout.rdbuf(fout.rdbuf());
            }
            // A generic buffer for separation purposes
            std::string buf = "        ";
            // Prints header info
            std::cout << "Position" << " " << buf << " " << buf << " || "
                      << "Velocity" << " " << buf << " " << buf << " || "
                      << "  Time  " << " || " << "Probability" << "\n\n";
            // Prints the data for each neutron
            for (int i = 0; i < N; i++)
            {
                std::cout << std::fixed << std::setprecision(5) << std::setw(8) << std::right
                     << origins[i][0]
                 << " " << origins[i][1]
                 << " " << origins[i][2]
                 << " || "
                         << vel[i][0]
                         << " " << vel[i][1]
                         << " " << vel[i][2]
                         << " || "
                         << times[i]
                         << " || "
                         << probs[i]
                         << "\n\n";
            }
            /* If cout was redirected, this "fixes" it so that it prints to
             * stdout in the future. Otherwise, this does nothing.
             */
            std::cout.rdbuf(coutbuf);
            // Closes the file stream if it was ever openned.
            if (fname != std::string())
            {
                fout.close();
            }
        }

        void Beam::updateRays()
        {
            if (rayptr->size() > 1)
            {
                if (rayptr->size() != N)
                {
                    printf("Warning: The vector of Rays used to create this Beam object has been altered.\n\nWould you like to empty the vector to allow its data to be updated? (y/n)\n");
                    char neg = 'n';
                    char pos = 'y';
                    while (1)
                    {
                        int response = getchar();
                        if (response == (int)(neg))
                        {
                            printf("Aborting updateRays\n");
                            return;
                        }
                        else if (response == (int)(pos))
                        {
                            rayptr->clear();
                            rayptr->resize(N);
                            break;
                        }
                        else
                        {
                            printf("Unknown command. Would you like to continue? (y/n)\n");
                        }
                    }
                }
                for (int i = 0; i < N; i++)
                {
                    (*rayptr)[i]->update(origins[i], vel[i], times[i], probs[i]);
                }
            }
            else
            {
                rayptr->clear();
                for (int i = 0; i < N; i++)
                {
                    printf("This process might take a while.\n");
                    rayptr->push_back(std::make_shared<Ray>(origins[i], vel[i], times[i], probs[i]));
                }
            }
        }

        std::ostream& operator<<(std::ostream &fout, const Beam &beam)
        {
            std::vector<float> data;
            for (int i = 0; i < beam.N; i++)
            {
                data.push_back(beam.vel[i][0]);
                data.push_back(beam.vel[i][1]);
                data.push_back(beam.vel[i][2]);
                data.push_back(beam.probs[i]);
            }
            char *bytes = (char*)(data.data());
            fout.write(bytes, ((int)(data.size()))*sizeof(float));
            return fout;
        }
 
    }

}
