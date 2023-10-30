import numpy as np
import yaml
from scipy.interpolate import CubicSpline


def getwaypoints(map_name):
    """
    loads waypoints
    """
    # self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    linetype = "centerline"
    # linetype = "raceline"
    # if self.map_name == "example":
    #     self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)
    # else:
    #     self.waypoints = np.loadtxt('./maps/'+ map_name+'_'+linetype+'.csv', delimiter=",")

    full_csv = np.loadtxt('./new_maps/'+ map_name+'_'+linetype+'.csv', delimiter=",")
    waypoints = np.vstack((full_csv[:, 0], full_csv[:, 1])).T

    return waypoints

def generatelookuptable(track):
    #load track
    waypoints = getwaypoints(track)
    #plt.scatter(waypoints[:,0], waypoints[:,1])
    #trackwidth
    r = 1.5
    #abez,bbez coeffs
    a, b = interpolate(waypoints)
    ts_inverse, smax = fit_st(waypoints, a, b)

    lutable_density = 100 #[p/m]

    npoints = np.int(np.floor(2 * smax * lutable_density))
    print("table generated with npoints = ", npoints)
    svals = np.linspace(0, 2*smax, npoints)
    tvals = ts_inverse(svals)

    #  entries :
    names_table = ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']
    table = []
    for idx in range(npoints):
        track_point = eval_raw(waypoints, a, b, tvals[idx])
        phi = getangle_raw(waypoints, a, b, tvals[idx])
        n = [-np.sin(phi), np.cos(phi)]
        g_upper = r + track_point[0]*n[0] + track_point[1]*n[1]
        g_lower = -r + track_point[0]*n[0] + track_point[1]*n[1]
        table.append([svals[idx], tvals[idx], track_point[0], track_point[1], phi, np.cos(phi), np.sin(phi), g_upper, g_lower])

    table = np.array(table)
    #plot_track(table)
    print("Variables stored in following order = ", names_table)
    np.savetxt("./new_maps/"+str(track) + '_lutab.csv', table, delimiter = ', ')

    dict = {'smax': float(smax), 'ppm' : lutable_density}
    with open(r'./new_maps/'+track+'_params.yaml', 'w') as file:
        documents = yaml.dump(dict, file)
    return table, smax

def interpolate(waypoints):
    #interpolates with cubic bezier curves with cyclic boundary condition
    n = len(waypoints)
    M = np.zeros([n,n])

    #build M
    tridiagel = np.matrix([[1, 4, 1]])
    for idx in range(n-2):
        M[idx+1:idx+2, idx:idx+3] = tridiagel

    M[0,0:2]= tridiagel[:,1:3]
    M[-1,-2:]= tridiagel[:,0:2]
    M[0:2,-1] = tridiagel[:,0].reshape(1,-1)
    M[-1,0] = tridiagel[:,0].reshape(1,-1)


    #build sol vector
    s =np.zeros([n,2])
    for idx in range(n-1):
        s[idx,:] = 2*(2*waypoints[idx,:] + waypoints[idx+1,:])
    s[-1:] = 2*(2*waypoints[-1,:] + waypoints[0,:])

    #solve for a & b
    Ax = np.linalg.solve(M,s[:,0])
    Ay = np.linalg.solve(M,s[:,1])

    a = np.vstack([Ax,Ay])
    b = np.zeros([2,n])

    b[:,:-1] = 2*waypoints.T[:,1:] - a[:,1:]
    b[:,-1] = 2*waypoints.T[:,0] - a[:,0]

    return a, b

def fit_st(waypoints, a, b):
    #using two revolutions to account for horizon overshooting end of lap

    #fit  the s-t rel.
    nwp = len(waypoints)
    npoints = 20 * nwp
    #compute approx max distance
    tvals = np.linspace(0, nwp, npoints+1)
    coords =[]
    for t in tvals:
        coords.append(eval_raw(waypoints, a, b, t))
    coords = np.array(coords)
    dists = []
    dists.append(0)
    for idx in range(npoints):
        dists.append(np.sqrt(np.sum(np.square(coords[idx,:]-coords[np.mod(idx+1,npoints-1),:]))))
    dists = np.cumsum(np.array(dists))
    smax = dists[-1]

    #--------fit  the s-t rel. to two track revolutions------
    npoints = 2 * 20 * nwp

    #compute approx distance to arc param
    tvals = np.linspace(0, 2*nwp, npoints+1)

    coords =[]
    for t in tvals:
        coords.append(eval_raw(waypoints, a, b, np.mod(t, nwp)))
    coords = np.array(coords)

    distsr = []
    distsr.append(0)
    for idx in range(npoints):
        distsr.append(np.sqrt(np.sum(np.square(coords[idx,:]-coords[np.mod(idx+1,npoints-1),:]))))
    dists = np.cumsum(np.array(distsr))

    ts_inverse = CubicSpline(dists, tvals)
    svals = np.linspace(0, 2*smax, npoints)
    t_corr = ts_inverse(svals)
    #t_corr = compute_t(coeffs,order,svals)

    #plt.figure()
    #plt.plot(tvals, dists, linestyle = '--')
    #plt.plot(t_corr, svals)
    #plt.xlabel("t (Bezier param) [-]")
    #plt.ylabel("s (approx. distance traveled) [m] ")

    return ts_inverse, smax

def eval_raw(waypoints, a, b, t):
    n = len(waypoints)
    t = np.mod(t, n)
    segment = np.floor(t)
    segment = np.int(segment)

    if segment>=n:
        t =n-0.0001
        segment = n-1
    elif t<0:
        t = 0
    t_val = t-segment
    coords = np.power(1 - t_val, 3) * waypoints.T[:,segment] + 3 * np.power(1 - t_val, 2) * t_val * a[:,segment]\
    + 3 * (1 - t_val) * np.power(t_val, 2) * b[:,segment] + np.power(t_val, 3) * waypoints.T[:,np.int(np.mod(segment+1,n))]

    return coords


def getangle_raw(waypoints, a, b, t):
    der = eval_raw(waypoints, a, b, t+0.1) - eval_raw(waypoints, a, b, t)
    phi = np.arctan2(der[1],der[0])
    return phi