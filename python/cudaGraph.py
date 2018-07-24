import numpy as np
from matplotlib import pyplot as plt

blockSizes = np.array([128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 384, 384, 384, 384, 384, 384, 384, 384, 384, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024])
times = np.array([25.682942, 21.127134, 22.044509, 27.301100, 22.045217, 22.322862, 22.314951, 23.473169, 22.351696, 21.605955, 27.066542, 20.994522, 24.329438, 27.507785, 21.830767, 22.520015, 24.122933, 22.702655, 22.135867, 21.918512, 23.196579, 22.882496, 22.182173, 22.935859, 26.967589, 23.110419, 22.348543, 21.662236, 21.587916, 21.316156, 22.878511, 22.581577, 21.903921, 21.920379, 22.158511, 24.552118, 26.799930, 22.524807, 21.185329, 21.548105, 23.021155, 22.197328, 21.317003, 21.533016, 22.405058])

avg = times[:]
tmp = times[:]
tmp = np.reshape(tmp, (5, 9))
avgBlock = np.unique(blockSizes)
avg = np.reshape(avg, (5, 9))
avg = np.average(avg, axis=1)
sdev = np.std(tmp, axis=1)

plt.errorbar(avgBlock, avg, yerr=sdev, fmt="-", color="#FF0000", linewidth=2.5, elinewidth=2.5, capsize=5, capthick=2.5)

plt.xticks(np.unique(blockSizes), fontsize=18)
plt.yticks(fontsize=18)

plt.xlabel("Number of CUDA Threads per Block", fontsize=22)
plt.ylabel("Execution Times (s)", fontsize=22)

plt.title("Execution Time vs Number of CUDA Threads per Block\nfor 100 Million Neutron Simulation", fontsize=24)

plt.grid()

plt.show()
