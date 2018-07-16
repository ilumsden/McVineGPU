from __future__ import print_function, division, unicode_literals

import ast
from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def generateHistogram(fname, v):
    f = open(fname, "rb")
    data = np.fromfile(f, dtype=np.float32)
    rows = data.size // 4
    data = data.reshape(rows, 4)
    vel = data[:,:3]
    prob = data[:,-1]
    initVel = np.array(v, dtype=np.float32)
    angles = np.empty(shape=(rows))
    normi = np.linalg.norm(initVel)
    normf = np.linalg.norm(vel, axis=1)
    denom = normi * normf
    cosvals = np.dot(vel, initVel)
    cosvals /= denom
    angles = np.degrees( np.arccos(cosvals) )
    hist, binEdges = np.histogram(angles, bins=180, range=(0,180), weights=prob)
    hist = hist.astype(np.float64)
    sbins = np.sin( np.radians(binEdges) )
    hist[1:] /= sbins[1:-1]
    return hist, binEdges

def plotHistogram(hist, binEdges):
    center = (binEdges[:-1] + binEdges[1:])/2
    plt.plot(center, hist)
    ymax = np.amax(hist) + 1000
    plt.ylim(0, ymax)
    plt.xlim(0, 180)
    plt.xlabel("Scattering Angle ({})".format(chr(176)))
    plt.ylabel("Normalized Intensity")
    plt.title("Normalized Intensity vs Scattering Angle")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    fname = os.path.abspath("../build/finalData.dat")
    v = [1.0, 0.0, 0.0]
    plot = False
    if len(sys.argv) > 1:
        for arg in sys.argv:
            if arg == os.path.basename(__file__):
                continue
            elif arg == "-h" or arg == "--help":
                print("Usage: python datahist.py [Options]")
                print("    -h | --help: Prints the usage info to stdout")
                print("    --fname=_: Specifies the data file to read")
                print("               Replace \"_\" with the absolute or")
                print("               relative path to the file")
                print("    --vel=(#,#,#): Specifies the initial velocity")
                print("                   Replace \"#\" with the desired velocity components")
                print("    -plot: Use this flag to plot the histogram")
                sys.exit()
            elif arg[:8] == "--fname=":
                fname = os.path.abspath(arg[8:])
            elif arg[:6] == "--vel=":
                v = list(ast.literal_eval(arg[6:]))
            elif arg == "-plot":
                plot = True
            else:
                print("Usage: python datahist.py [Options]")
                print("    -h | --help: Prints the usage info to stdout")
                print("    --fname=_: Specifies the data file to read")
                print("               Replace \"_\" with the absolute or")
                print("               relative path to the file")
                print("    --vel=(#,#,#): Specifies the initial velocity")
                print("                   Replace \"#\" with the desired velocity components")
                print("    -plot: Use this flag to plot the histogram")
                sys.exit()
    hist, binEdges = generateHistogram(fname, v)
    if plot:
        plotHistogram(hist, binEdges)
