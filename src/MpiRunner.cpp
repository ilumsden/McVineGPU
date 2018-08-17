#include "MpiRunner.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace mcvine
{

    namespace gpu
    {

        void runMPI(int *argc, char ***argv,
                    std::function<void(std::shared_ptr<AbstractShape>,
                                       const int)> simFunc,
                    std::shared_ptr<AbstractShape> &shape,
                    const int numNeutrons, const int blockSize)
        {
            MPI_Init(argc, argv);
            int numProcesses;
            MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::vector<int> neutronsByProc;
            int neutrons;
            float *s_data;
            int dataSize;
            std::string type;
            if (rank == 0)
            {
                int neutronsPerProcess = numNeutrons / numProcesses;
                int modNeutrons = numNeutrons % numProcesses;
                neutronsByProc.resize(numProcesses)
                for (int i = 0; i < numProcesses; i++) 
                {
                    neutronsByProc[i] = neutronsPerProcess;
                }
                if (modNeutrons != 0)
                {
                    int index = 0;
                    for (int i = 0; i < modNeutrons; i++) 
                    {
                        neutronsByProc[index]++;
                        index++;
                        if (index >= (int)(neutronsByProc.size()))
                        {
                            index = 0;
                        }
                    }
                }
                s_data = shape->data;
                s_type = shape->type;
                switch (s_type)
                {
                    case "Box": dataSize = 3; break;
                    case "Cylinder": dataSize = 2; break;
                    case "Pyramid": dataSize = 3; break;
                    case "Sphere": dataSize = 1; break;
                    default: throw std::invalid_argument("Invalid Shape Type.");
                }
            }
            MPI_Bcast(s_data, dataSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(type.c_str(), (int)(type.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
        }

    }

}
