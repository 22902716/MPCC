import numpy as np
from matplotlib import pyplot as plt


def projected_trajectory_plot(mapname,N,wpts,pausetime):
    data = np.loadtxt("./csv/"+mapname+"/"+mapname+'.csv',delimiter=",",skiprows=1)
    print(data)
    plt.figure()
    

    for i in range(len(data[0])):

        for j in range(N):
            plt.plot(data[4*j+0][i],data[4*j+1][i],'wx')
            plt.plot(data[4*j+2][i],data[4*j+3][i],'wo')
            plt.plot(data[4*j+0][i],data[4*j+1][i],'bx')
            plt.plot(data[4*j+2][i],data[4*j+3][i],'bo')
            
        plt.pause(pausetime)
        
        plt.clf()
        plt.plot(wpts[:,0],wpts[:,1],"bx",markersize=1)
    # plt.show()


if __name__ == "__main__":
    N = 5
    pausetime = 0.01
    mapname = "gbr"

    track_lu_table = np.loadtxt('./new_maps/'+ mapname+'_'+'lutab'+'.csv', delimiter=",")
    wpts = np.vstack((track_lu_table[:,2],track_lu_table[:,3])).T

    projected_trajectory_plot(mapname, N, wpts,pausetime)