import numpy as np
from matplotlib import pyplot as plt

blockSizes = np.array([128, 128, 128, 256, 256, 256, 384, 384, 384, 512, 512, 512, 1024, 1024, 1024])#640, 768, 896, 1024])
times = np.array([25.682942, 21.127134, 22.044509, 21.605955, 27.066542, 20.994522, 22.135867, 21.918512, 23.196579, 21.662236, 21.587916, 21.316156, 26.799930, 22.524807, 21.185329])

avg = times[:]
avgBlock = np.unique(blockSizes)
avg = np.reshape(avg, (5, 3))
avg = np.average(avg, axis=1)

plt.plot(blockSizes, times, ".", color="#74B71B")
plt.plot(avgBlock, avg, "--", color="#74B71B")

plt.xticks(np.unique(blockSizes))

plt.xlabel("Number of CUDA Threads per Block")
plt.ylabel("Execution Times (s)")

plt.title("Execution Time vs Number of CUDA Threads per Block\nfor 100 Million Neutron Simulation")

plt.grid()

plt.show()
