#include "Beam.hpp"
#include "Error.hpp"

namespace mcvine
{

    namespace gpu
    {

        Beam::Beam(std::vector< std::shared_ptr<Ray> > &rays, int size, int bS)
        {
            blockSize = bS;
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
                initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_origins, N, rays[0].origin);
                initArray< Vec3<float> ><<<numBlocks, blockSize>>>(d_vel, N, rays[0].vel);
                initArray<float><<<numBlocks, blockSize>>>(d_times, N, rays[0].t);
                initArray<float><<<numBlocks, blockSize>>>(d_probs, N, rays[0].prob);
                CudaErrchk( cudaMemcpy(origins, d_origins, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(vel, d_vel, N*sizeof(Vec3<float>), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(times, d_times, N*sizeof(float), cudaMemcpyDeviceToHost) );
                CudaErrchk( cudaMemcpy(probs, d_probs, N*sizeof(float), cudaMemcpyDeviceToHost) );
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
 
    }

}
