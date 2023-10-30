import time
from f110_gym.envs.base_classes import Integrator
import gym
import yaml
import numpy as np
from argparse import Namespace
import casadi as ca
import Bezier
import math, cmath
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
from my_MPCC_util import my_MPCC_util
# np.set_printoptions(threshold=np.inf)


mass=2.5                                       #vehicle mass
height = 0.09
width = 0.14
length = 0.33
speed  = 4
center_to_trackbound = 0.5

# VERBOSE = True
VERBOSE = False

# SAVEDATA = True
SAVEDATA = False

REALTIME_VERBOSE = True
# REALTIME_VERBOSE = False

# REALTIME_VERBOSE_temp = True
REALTIME_VERBOSE_temp = False

class MPCC:
    def __init__(self, conf, map_name):
        self.speedgain = 1.

        self.nx = 3 #number of input [x,y,theta]
        self.nu = 1 #number of output [steering_command]
        self.N = 5  #prediction horizon
        self.dt = 0.1
        self.map_name = map_name
        self.conf = conf
        self.drawn_waypoints = []
        self.L = 0.324
        self.load_waypoints()

        self.u_min = [-0.4]
        self.u_max = [0.4]

        completion = self.track_lu_table[1100,1]
        xt0 = self.track_lu_table[1100,2]
        yt0 = self.track_lu_table[1100,3]
        phit0 = self.track_lu_table[1100,4]
        
        self.x0 = np.array([xt0,yt0,phit0])
        # print(self.x0)

        self.mu = my_MPCC_util(map_name, 0.1)


    def load_waypoints(self):
        """
        loads waypoints
        """
        # ***if its a new map uncomment this and generate the new trajectory file***
        # self.track_lu_table, smax = Bezier.generatelookuptable(self.map_name) 
        # exit()

        # self.track_lu_table = np.loadtxt('./new_maps/'+ self.map_name+'_'+'lutab_partial'+'.csv', delimiter=",")
        self.track_lu_table = np.loadtxt('./new_maps/'+ self.map_name+'_'+'lutab'+'.csv', delimiter=",")
        self.waypoints = np.loadtxt('./new_maps/'+ self.map_name+'_'+'lutab'+'.csv', delimiter=",")
        self.wpts = np.vstack((self.track_lu_table[:,2],self.track_lu_table[:,3])).T
        self.centerline_width = np.loadtxt('./new_maps/'+self.map_name+'_centerline.csv', delimiter=',') 
        self.centerline = np.vstack((self.centerline_width[:,0],self.centerline_width[:,1])).T
        self.boundary_inner_outer_distance = np.vstack((self.centerline_width[:,2],self.centerline_width[:,3])).T

        #track_lu_table_heading = ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']


    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """


        
        scaled_points = 50.*self.wpts

        for i in range(scaled_points.shape[0]):
            if len(self.drawn_waypoints) < scaled_points.shape[0]:
                b = e.batch.add(1, 0, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

        self.diffs = self.wpts[1:,:] - self.wpts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        seg_lengths = np.linalg.norm(np.diff(self.wpts, axis=0), axis=1)
        self.ss = np.insert(np.cumsum(seg_lengths), 0, 0)
        self.trueSpeedProfile = []
        for i in range(0,len(self.wpts)):
            self.trueSpeedProfile.append(speed)

        self.trueSpeedProfile = np.array(self.trueSpeedProfile)
        self.vs = self.trueSpeedProfile*self.speedgain   #speed profile

        self.total_s = self.ss[-1]
        self.tN = len(self.wpts)


    def get_timed_trajectory_segment(self, position, dt, n_pts=10):
        pose = np.array([position[0], position[1], speed])
        trajectory, distances = [pose], [0]
        for i in range(n_pts-1):
            # distance = dt * pose[2]
            distance = dt * speed
            
            current_distance = self.calculate_progress(pose[0:2])
            next_distance = current_distance + distance
            distances.append(next_distance)
            
            interpolated_x = np.interp(next_distance, self.ss, self.wpts[:, 0])
            interpolated_y = np.interp(next_distance, self.ss, self.wpts[:, 1])

            interpolated_v = np.interp(next_distance, self.ss, self.vs)
            
            pose = np.array([interpolated_x, interpolated_y, interpolated_v])
            

            trajectory.append(pose)

        interpolated_waypoints = np.array(trajectory)
        return interpolated_waypoints

    def print_temp_graph(self, temp, idx1,idx2):
            plt.plot(self.wpts[:,0],self.wpts[:,1],markersize = 1)
            plt.scatter(temp[:,idx1],temp[:,idx2])
            plt.pause(0.001)
            plt.scatter(temp[:,idx1],temp[:,idx2])

    def interp_pts(self, idx, dists):
        """
        Returns the distance along the trackline and the height above the trackline
        Finds the reflected distance along the line joining wpt1 and wpt2
        Uses Herons formula for the area of a triangle
        
        """
        d_ss = self.ss[idx+1] - self.ss[idx]

        d1, d2 = dists[idx], dists[idx+1]

        if d1 < 0.01: # at the first point
            x = 0   
            h = 0
        elif d2 < 0.01: # at the second point
            x = dists[idx] # the distance to the previous point
            h = 0 # there is no distance
        else: 
            # if the point is somewhere along the line
            s = (d_ss + d1 + d2)/2
            Area_square = (s*(s-d1)*(s-d2)*(s-d_ss))
            if Area_square < 0:
                # negative due to floating point precision
                # if the point is very close to the trackline, then the trianlge area is increadibly small
                h = 0
                x = d_ss + d1
                # print(f"Area square is negative: {Area_square}")
            else:
                Area = Area_square**0.5
                h = Area * 2/d_ss
                x = (d1**2 - h**2)**0.5

        return x, h
    
    def get_trackline_segment(self, point):
        """
        Returns the first index representing the line segment that is closest to the point.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)

        if min_dist_segment == len(self.wpts)-1:
            min_dist_segment = 0

        return min_dist_segment,dists
        
    def calculate_progress(self, point):
        idx, dists = self.get_trackline_segment(point)
        x, h = self.interp_pts(idx, dists)
        s = self.ss[idx] + x
        
        return s
    

    def estimate_u0(self, reference_path, x0):

        # print("ref path: ",reference_path)
        reference_theta = np.arctan2(reference_path[1:, 1] - reference_path[:-1, 1], reference_path[1:, 0] - reference_path[:-1, 0])
        # print(reference_path[1:, 1] - reference_path[:-1, 1])
        # print(reference_path[1:, 0] - reference_path[:-1, 0])
        # print("reference theta",reference_theta)
        

#----------------------------------------------------------------------------------------
        # th_dot = self.calculate_angle_diff(reference_theta) 
        # th_dot[0] += (reference_theta[0]- x0[2]) 
        # speeds = reference_path[:, 2]
        # print((np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt)
        # print((np.arctan(th_dot) * self.L / speeds[:-2]) )
        # steering_angles = (np.arctan(th_dot) * self.L / speeds[:-2]) / self.dt
        # speeds[0] += (speed - reference_path[0, 2] )
#----------------------------------------------------------------------------------------

        speeds = reference_path[:, 2]
        accelerations = np.diff(speeds) / self.dt

        steering_angles = np.zeros(self.N)

        #correct the steering angle initial guess (try 0) first
        u0_estimated = np.vstack((steering_angles, accelerations[:-1])).T
        # print("estimates: ",u0_estimated)
        return u0_estimated
    
    def calculate_angle_diff(self,angle_vec):
        angle_diff = np.zeros(len(angle_vec)-1)

        for i in range(len(angle_vec)-1):
            angle_diff[i] = self.sub_angles_complex(angle_vec[i], angle_vec[i+1])
        
        return angle_diff
        
    def sub_angles_complex(self, a1, a2): 
        real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
        im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

        cpx = complex(real, im)
        phase = cmath.phase(cpx)

        return phase
    
    def Problem_setup(self,x0_init, x_ref, u_init):

        Qc = 1000.  #contour weighting 
        Ql = 10. #lag weighting
        J = 0
        x_predict = x0_init
        self.inner_bound = [[0.0]*2]*self.N
        self.outer_bound = [[0.0]*2]*self.N
        self.inner_bound = np.array(self.inner_bound)
        self.outer_bound = np.array(self.outer_bound)

        print("x0 init",x0_init)
        print("x_ref: ",x_ref)
        # print("u_init",u_init)
        
        x = ca.SX.sym('x', self.nx, self.N+1)
        # print(x)
        u = ca.SX.sym('u', self.nu, self.N)
        # print(u)

        # lbg = [0, 0, 0, 0]* (self.N) 

        # ubg = [0, 0, 0, 0]* (self.N)

        lbg = [0] * ((self.nx+1) * self.N) + [0] * self.nx 

        ubg = [0] * ((self.nx+1) * self.N) + [0] * self.nx 
        # print("ubg",ubg)

        
        g = []
        initial_constraint = x[:,0] - x0_init
        g.append(initial_constraint)
        # g.append(x[0]*delta_x_path + x[1]*delta_y_path)

        for i in range(self.N):
            xt =  x[0+i*self.nx]
            yt =  x[1+i*self.nx]
            # print([xt,yt])
            posx = x_ref[i][0]
            posy = x_ref[i][1]
            point = [posx, posy]
            # print(point)

            dists1 = np.linalg.norm(point - self.wpts, axis=1)   #closest distance from car to reference point
            min_dist_segment1 = np.argmin(dists1)

            temp = self.dynamic_literal(x_predict,u_init[i])
            x_predict = np.add(x_predict,temp)
            print("x_predict", x_predict)
            dists2 = np.linalg.norm(x_predict[:2] - self.wpts, axis=1)      #cloeset distance from prediction of car to reference point
            min_dist_segment2 = np.argmin(dists2)

            self.inner_bound[i],self.outer_bound[i] = self.boundary_point_estimation(x_ref[i][0:2])


            theta = self.track_lu_table[min_dist_segment2][0]
            
            sin_phit = self.track_lu_table[min_dist_segment1][6]
            cos_phit = self.track_lu_table[min_dist_segment1][5]
            theta_hat = self.track_lu_table[min_dist_segment1][0]
            
            xt_hat = xt + cos_phit * ( theta - theta_hat)
            yt_hat = yt + sin_phit * ( theta - theta_hat)

            e_cont = sin_phit * (xt_hat - posx) - cos_phit *(yt_hat - posy)
            e_lag = cos_phit * (xt_hat - posx) + sin_phit *(yt_hat - posy)

            J += e_cont*Qc*e_cont + e_lag*Ql*e_lag


            #---------------------------------------------------------
            #printing things that might matter
            J_print = 0

            xt_hat_print = x_predict[0] + cos_phit * ( theta - theta_hat)
            yt_hat_print = x_predict[1] + sin_phit * ( theta - theta_hat)

            e_cont_print = sin_phit * (xt_hat_print - posx) - cos_phit *(yt_hat_print - posy)
            e_lag_print = cos_phit * (xt_hat_print - posx) + sin_phit *(yt_hat_print - posy)
            
            J_print += e_cont_print*Qc*e_cont_print + e_lag_print*Ql*e_lag_print
            # print("xt_hat: ",xt_hat_print,"yt_hat: ",yt_hat_print)
            # print("e_cont: ",e_cont_print,"e_lag: ",e_lag_print)
            # print("J: ",J_print)
            #---------------------------------------------------------


            s = theta_hat
            if s > self.mu.track_length:
                s = s - self.mu.track_length
            # right_point = [self.mu.right_lut_x(s).full()[0, 0], self.mu.right_lut_y(s).full()[0, 0]]
            # left_point = [self.mu.left_lut_x(s).full()[0, 0], self.mu.left_lut_y(s).full()[0, 0]]

            # self.outer_bound[i] = [self.mu.right_lut_x(s).full()[0, 0], self.mu.right_lut_y(s).full()[0, 0]]
            # self.inner_bound[i] = [self.mu.left_lut_x(s).full()[0, 0], self.mu.left_lut_y(s).full()[0, 0]]
            
            # print("right_point", right_point)
            # print("left_point", left_point)
            
            # print("est_right point",self.inner_bound[i])
            # print("est_left point",self.outer_bound[i])
            # exit()


            delta_x_path = self.inner_bound[i,0] - self.outer_bound[i,0] 
            delta_y_path = self.inner_bound[i,1] - self.outer_bound[i,1] 
            print("delta X: ",delta_x_path)
            print("delta Y: ",delta_y_path)
            up_bound = max(delta_x_path * self.outer_bound[i, 0] + delta_y_path * self.outer_bound[i, 1],
                           delta_x_path * self.inner_bound[i, 0] + delta_y_path * self.inner_bound[i, 1])
            low_bound = min(delta_x_path * self.outer_bound[i, 0] + delta_y_path * self.outer_bound[i, 1],
                            delta_x_path * self.inner_bound[i, 0] + delta_y_path * self.inner_bound[i, 1])
            # print("upper bound = ",up_bound)
            # print("Lower bound = ",low_bound)
            print("Bound: ",delta_x_path * self.outer_bound[i, 0] + delta_y_path * self.outer_bound[i, 1],
                           delta_x_path * self.inner_bound[i, 0] + delta_y_path * self.inner_bound[i, 1])



            # delta_x_path = right_point[0] - left_point[0]
            # delta_y_path = right_point[1] - left_point[1]

            # up_bound = max(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
            #                -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            # low_bound = min(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
            #                 -delta_x_path * left_point[0] - delta_y_path * left_point[1])

            ubg[4*i+3] = up_bound
            lbg[4*i+3] = low_bound

            x_next = x[:,i] + self.dynamic(x[:,i], u[:,i])*self.dt
            # print(k,":",x_next)
            g.append(x[0+(i*self.nx)]*delta_x_path + x[1+(i*self.nx)]*delta_y_path)
            g.append(x_next - x[:,i+1])

            #-----
            # up_bound = max(self.outer_bound[k, 0] + self.outer_bound[k, 1],
            #                self.inner_bound[k, 0] + self.inner_bound[k, 1])
            # low_bound = min(self.outer_bound[k, 0] + self.outer_bound[k, 1],
            #                 self.inner_bound[k, 0] + self.inner_bound[k, 1])
            # # print("upper bound = ",up_bound)
            # # print("Lower bound = ",low_bound)
            # print("Bound: ", self.outer_bound[k, 0] +  self.outer_bound[k, 1],
            #                 self.inner_bound[k, 0] +  self.inner_bound[k, 1])

            # ubg[4*k+3] = up_bound
            # lbg[4*k+3] = low_bound

            # x_next = x[:,k] + self.dynamic(x[:,k], u[:,k])*self.dt
            # # print(k,":",x_next)
            # g.append(x[0+(k*self.nx)] + x[1+(k*self.nx)])
            # g.append(x_next - x[:,k+1])
            #------
            
            # print ("G:",k,": ",g)
        # print("lbg: ",lbg)
        # print("ubg: ",ubg)
        
        # print("x", x)
        # print("x0 init",x0_init)

        
        x_init = [x0_init]
        # print(u_init)
        for i in range(0, self.N):
            x_init.append(x_init[i] + self.dynamic(x_init[i], u_init[i])*self.dt)
        for i in range(len(u_init)):
            x_init.append(u_init[i][0])

        # print("x_init: ",x_init)
        

        x_init = ca.vertcat(*x_init)
        # print("x_init.shape : ",x_init.shape)

        lbx = [-ca.inf, -ca.inf, -ca.inf] * (self.N+1) + self.u_min*self.N
        # print("lbx: " , lbx)

        ubx = [ca.inf, ca.inf, ca.inf] * (self.N+1) + self.u_max*self.N
        # print("ubx: " , ubx)


        


        x_nlp = ca.vertcat(x.reshape((-1,1)), u.reshape ((-1,1)))
        # print("x_nlp.shape", x_nlp)

        g_nlp = ca.vertcat(*g)
        # print("g_nlp.shape", g_nlp)


        nlp = {'x': x_nlp,
               'f' : J,
               'g' : g_nlp}
        
        opts = {'ipopt' : {'print_level': 2},
                'print_time' : False}
        solver = ca.nlpsol('solver', 'ipopt' , nlp, opts)
        sol = solver(x0 = x_init, lbx = lbx, ubx = ubx, lbg = lbg, ubg = ubg)
        # print(sol['x'])

        x_bar = np.array(sol['x'][:self.nx*(self.N+1)].reshape((self.nx, self.N+1)))
        print(x_bar)

        u_bar = sol['x'][self.nx*(self.N+1):]
        # print("u_bar: ",u_bar)

        Steering = u_bar[0][0]
        # print("steering : ", Steering)
        # exit()
        # Speed = 0
        return x_bar,Steering

    def dynamic_literal(self,x,u):
        return [np.cos(x[2])*speed*self.dt, np.sin(x[2])*speed*self.dt, speed*self.dt/self.L * np.tan(u[0])]


    def dynamic(self, x, u):
        # define the dynamics as a casadi array
        xdot = ca.vertcat(
            ca.cos(x[2])*speed,
            ca.sin(x[2])*speed,
            speed/self.L * ca.tan(u[0])
        )
        return xdot
    
    def plan(self,x0):
        x_ref = self.get_timed_trajectory_segment(x0, self.dt, self.N+2)

        u_init = self.estimate_u0(x_ref, x0)

        x_bar, steering_angle = self.Problem_setup(x0, x_ref,u_init)

        return x_bar, steering_angle, x_ref
    
    def boundary_point_estimation(self, point):
        # dists3 = np.linalg.norm(point - self.centerline,axis=1)
        dists3 = np.linalg.norm(point - self.wpts,axis=1)
        min_dist_segment3 = np.argmin(dists3)
        # if min_dist_segment3+2 >= len(self.centerline)-1:
        if min_dist_segment3+2 >= len(self.wpts)-1:
            min_dist_segment3 -= 1

        # x1 = [self.centerline[min_dist_segment3][0],self.centerline[min_dist_segment3][1]]
        # x2 = [self.centerline[min_dist_segment3+2][0],self.centerline[min_dist_segment3+2][1]]
        x1 = [self.wpts[min_dist_segment3][0],self.wpts[min_dist_segment3][1]]
        x2 = [self.wpts[min_dist_segment3+2][0],self.wpts[min_dist_segment3+2][1]]
        
        gradient= self.angle_between(x1,x2)

        outer_bound = [x1[0]+center_to_trackbound*np.cos(gradient+(np.pi/2)),
                       x1[1]+center_to_trackbound*np.sin(gradient+(np.pi/2))]
        inner_bound = [x1[0]-center_to_trackbound*np.cos(gradient+(np.pi/2)),
                       x1[1]-center_to_trackbound*np.sin(gradient+(np.pi/2))]
        

        # print("x0 = ", point)
        # print("angle = ",gradient)
        # print("inner bound = ",self.inner_bound)
        # print("outer bound = ",self.outer_bound)


        return inner_bound,outer_bound

    def angle_between(self, p1, p2):
        theta = np.arctan2((p1[1]-p2[1]),p1[0]-p2[0])
        return theta


def main():

    speedgain = 1.
    map_name_list = ["gbr"]
    map_name = map_name_list[0]
    xbar_x = []
    xbar_y = []
    xref_x = []
    xref_y = []        
    counter = 0

    with open('maps/config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    
    planner = MPCC(conf,map_name_list[0])

    def render_callback(env_renderer):
        e = env_renderer

        # update camera to follow car

        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    if map_name == "example":
        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
        obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    else:
        env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
        obs, step_reward, done, info = env.reset(np.array([[planner.x0[0], planner.x0[1], planner.x0[2]]]))

    env.add_render_callback(render_callback)
    env.render()


    if REALTIME_VERBOSE:
        plt.figure()
    if REALTIME_VERBOSE_temp:
        plt.figure()

    while not done:
        print("new iter.................")
        x0 = [obs['poses_x'][0], obs['poses_y'][0],obs['poses_theta'][0]]
        x_bar,steering_angle,x_ref = planner.plan(x0)
        print("steering : ",steering_angle)
        # print(x_ref)
        z = 2
        while z > 0:
            # obs, _, done, _ = env.step(np.array([[0., speed*speedgain]]))
            obs, _, done, _ = env.step(np.array([[steering_angle, speed*speedgain]]))
            z-=1
        env.render(mode='human_fast')

        xbar_x.append(x_bar[0,:])
        xbar_y.append(x_bar[1,:])
        xref_x.append(x_ref[0:planner.N+1,0])    
        xref_y.append(x_ref[0:planner.N+1,1])
        
        # print(xbar_x)
        # print(xbar_y)
        # print(xref_x)
        # print(xref_y)

        counter += 1
    

        if REALTIME_VERBOSE:
            # plt.plot(planner.wpts[:,0],planner.wpts[:,1],"bx",markersize=1)
            # plt.plot(planner.centerline[:,0],planner.centerline[:,1],"bx",markersize=1)
            plt.plot(x_bar[0, :], x_bar[1, :], 'bo', markersize=4, label="Solution States (x_bar)") 
            plt.plot(x_ref[:,0],x_ref[:,1],'bx',label = "x_ref")
            # print(planner.inner_bound)
            plt.plot(planner.inner_bound[:,0],planner.inner_bound[:,1],'bx',markersize = 3)
            plt.plot(planner.outer_bound[:,0],planner.outer_bound[:,1],'rx',markersize = 3)
            plt.pause(0.01)
            plt.clf()


    print(counter)
    if VERBOSE:
        
        xbar_x = np.array(xbar_x)
        xbar_y = np.array(xbar_y)
        xref_x = np.array(xref_x)
        xref_y = np.array(xref_y)
        # print(xbar_x)
        # print(xbar_y)
        # print(xref_x)
        # print(xref_y)
        save_arr = np.zeros(counter)
        # exit()
        # print (xbar_x.shape)
        for i in range(planner.N):
            save_arr = np.vstack((save_arr, xbar_x[:,i]))
            save_arr = np.vstack((save_arr, xbar_y[:,i]))
            save_arr = np.vstack((save_arr, xref_x[:,i]))
            save_arr = np.vstack((save_arr, xref_y[:,i]))
        save_arr = np.delete(save_arr, 0,0)


        if SAVEDATA:
            np.savetxt('csv/'+map_name+'/'+map_name+'.csv',save_arr,delimiter=',',header="xbar_x,xbar_y,xref_x,xref_y,N = "+ str(planner.N),fmt="%-10f")
            print("file saved successfully")
            

        
        
if __name__ == '__main__':
    main()