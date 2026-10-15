import numpy as np

import glob as glob

direc = "sync_runs2/"

filenames = glob.glob(direc + "*")

nprocs = np.array([8, 16, 32, 64], dtype=np.int32)

timedata = [[] for i in range(len(nprocs))]
nbranchesdata = [[] for i in range(len(nprocs))]


for filename in filenames:
    with open(filename, 'r') as f:
        nproc = int(filename.split("procs")[0].split("_")[-1])
        for index, line in enumerate(f):
            if "Total elapsed time" in line:
                elapsed_time = float(line.split("Total elapsed time: ")[1])
            if "number of branches in bnb algorithm" in line:
                numbranches = float(line.split("number of branches in bnb algorithm = ")[1])
        arg = np.argwhere(nprocs == nproc).flatten()[0]
        timedata[arg].append(elapsed_time)
        nbranchesdata[arg].append(numbranches)
        #print("elapsed time = ", elapsed_time)
        #print("nprocs = ", nproc)
        #print("number of branches = ", numbranches)
        #print("time / branch = ", elapsed_time / numbranches)

for i in range(len(nprocs)):
    timedata[i] = np.array(timedata[i])
    nbranchesdata[i] = np.array(nbranchesdata[i])
    print("nprocs = ", nprocs[i])
    print("elapsed time = {0:1.2e} +- {1:1.2e}".format(np.mean(timedata[i]), np.std(timedata[i])))
    print("num branches = {0:1.2e} +- {1:1.2e}".format(np.mean(nbranchesdata[i]), np.std(nbranchesdata[i])))
    print("elapsed time / branch = {0:1.2e} +- {1:1.2e}".format(np.mean(timedata[i] / nbranchesdata[i]), 
        np.std(timedata[i] / nbranchesdata[i])))
    #print("std dev elapsed time / branch = {0:1.2e}".format(np.std(timedata[i] / nbranchesdata[i])))

