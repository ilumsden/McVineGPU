import numpy as np
from matplotlib import pyplot as plt

x = np.array([1e6, 3e6, 6e6, 1e7, 3e7, 6e7, 1e8])
cputimes = np.array([50., 147., 293., 491., 1473., 3005., 4977.])
gputimes = np.array([3.72244, 4.24126, 4.71751, 5.69286, 8.8244, 14.7054, 21.627])

cpuspeed = x / cputimes
gpuspeed = x / gputimes

ratio = gpuspeed / cpuspeed

plt.plot(x, ratio, ".-", color="#74B71B")
plt.title("Ratio of McVineGPU's Speed to MCViNE's Speed vs Number of Neutrons")
plt.xlabel("Number of Neutrons")
plt.ylabel("Ratio of McVineGPU's Speed to MCViNE's Speed")

plt.xscale("log")

plt.grid()

plt.show()
