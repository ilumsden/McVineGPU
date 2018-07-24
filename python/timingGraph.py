import numpy as np
from matplotlib import pyplot as plt

x = np.array([1e6, 3e6, 6e6, 1e7, 3e7, 6e7, 1e8])
cputimes = np.array([50.0, 147.0, 293.0, 491.0, 1473.0, 3005.0, 4977.0])
gputimes = np.array([3.72244, 4.24126, 4.71751, 5.69286, 8.8244, 14.7054, 21.627])

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(x, gputimes, ".-", color="#FF0000", linewidth=2.5, markersize=7.5)
ax1.set_xlabel("Number of Neutrons", fontsize=22)
ax1.set_ylabel("McVineGPU Execution Time (s)", color="#FF0000", fontsize=22)
ax1.set_ylim(0, 25)
ax1.tick_params("x", labelsize=18)
ax1.tick_params("y", colors="#FF0000", labelsize=18)

ax2.plot(x, cputimes, ".-", color="#0000FF", linewidth=2.5, markersize=7.5)
ax2.set_ylabel("MCViNE Execution Time (s)", color="#0000FF", fontsize=22)
ax2.tick_params("y", colors="#0000FF", labelsize=18)

fig.suptitle("Timing Comparison of MCViNE vs McVineGPU", y=0.95, fontsize=24)

plt.show()
