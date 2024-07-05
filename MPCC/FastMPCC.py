
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from ReferencePath import ReferencePath as rp
from DataSave import dataSave

mu = 0.
sigma = 0.2

class MPCC:
    def __init__(self, map_name,TESTMODE):
        print("This is Fast MPCC TEST")
        self.nx = 4 #number of input [x, y, psi, s]
        self.nu = 3 #number of output [delta, v, p],steering(change in yaw angle), change in reference path progress and acceleration

        self.map_name = map_name
        self.TESTMODE = TESTMODE
        self.wheelbase = 0.324
        self.load_waypoints()
        self.scale = 0.

        if self.TESTMODE == "Benchmark" or self.TESTMODE == " ": 
            self.Max_iter= 5           
        elif self.TESTMODE == "perception_noise" or self.TESTMODE == "Outputnoise_speed" or self.TESTMODE == "Outputnoise_steering":
            self.Max_iter = 300
        elif self.TESTMODE == "control_delay_speed" or self.TESTMODE == "control_Delay_steering" or self.TESTMODE == "perception_delay":
            self.Max_iter = 100

        self.ds = dataSave(TESTMODE, map_name, self.Max_iter)

        #adjustable params
        #----------------------
        if map_name == "mco":
            self.dt = 0.17
        else:
            self.dt = 0.16
        self.N = 5  #prediction horizon
        self.mass = 3.8
        self.L = 0.324

        self.delta_min = -0.4
        self.delta_max = 0.4
        self.p_init = 7
        self.p_min = 1
        self.p_max = 10
        self.fmax = 9.81*1.0*self.mass

        self.psi_min = -10
        self.psi_max = 10

        self.weight_progress = 100
        self.weight_lag = 1000
        self.weight_contour = 1
        self.weight_steering = 1.5

        self.v_min = 2 
        self.v_max = 8
        #------------------------

        #initial position
        # position = 1000
        # completion = self.track_lu_table[position,1]
        # xt0 = self.track_lu_table[position,2]
        # yt0 = self.track_lu_table[position,3]
        # phit0 = self.track_lu_table[position,4]
        # self.x0 = np.array([xt0,yt0,phit0])

        self.rp = rp(map_name,w=0.55)
        self.u0 = np.zeros((self.N, self.nu))
        self.X0 = np.zeros((self.N + 1, self.nx))
        self.warm_start = True
        self.x_bar = np.zeros((2, 2))
        self.prev_x0 = [0.0,0.0,0.0]

        self.drawn_waypoints = []

        self.problem_setup()


    def load_waypoints(self):
        """
        loads waypoints
        """
        # ***if its a new map uncomment this and generate the new trajectory file***
        # self.track_lu_table, smax = Bezier.generatelookuptable(self.map_name) 
        # exit()
        self.centerline = np.loadtxt('./maps/'+ self.map_name +'_centerline.csv', delimiter=",")
        self.wpts = np.vstack((self.centerline[:,0],self.centerline[:,1])).T
        self.waypoints = np.loadtxt('./maps/'+ self.map_name +'_raceline.csv', delimiter=",")

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

    def get_trackline_segment(self, point):
        """
        Returns the first index representing the line segment that is closest to the point.
        """
        dists = np.linalg.norm(point - self.wpts, axis=1)

        min_dist_segment = np.argmin(dists)

        if min_dist_segment == len(self.wpts)-1:
            min_dist_segment = 0

        return min_dist_segment,dists
        

    def plan(self, obs,laptime):
        # lower case x0: [x,y,psi,states]
        # upper case X0: [x,y,psi,speed]

        x0, self.X0_slip = self.inputStateAdust(obs)
        x0 = self.build_initial_state(x0)
        self.construct_warm_start_soln(x0) 

        p = self.generate_parameters(x0,self.X0_slip[3])
        controls,self.x_bar = self.solve(p)

        action = np.array([controls[0, 0], controls[0,1]])
        speed,steering = self.outputActionAdjust(action[1],action[0])

        ego_index,min_dists = self.get_trackline_segment(x0[0:2])
        self.completion = 100 if ego_index/len(self.wpts) == 0 else round(ego_index/len(self.wpts)*100,2)
        slip_angle = self.slipAngleCalc(obs)
        self.ds.saveStates(laptime, self.X0_slip, 0.0, 0.0, self.scaledRand, self.completion, steering, slip_angle)

        return speed,steering
    
    def slipAngleCalc(self,obs):
        x = [self.X0_slip[0] - self.prev_x0[0]]
        y = [self.X0_slip[1] - self.prev_x0[1]]
        
        velocity_dir = np.arctan2(y,x)
        slip = np.abs(velocity_dir[0] - obs['poses_theta'][0]) *360 / (2*np.pi)
        if slip > 180:
            slip = slip-360

        self.prev_x0 = self.X0_slip

        return slip

    def problem_setup(self):
        states = ca.MX.sym('states', self.nx) #[x, y, psi, s]
        controls = ca.MX.sym('controls', self.nu) # [delta, v, p]

        #set up dynamic states of the vehichle
        rhs = ca.vertcat(controls[1] * ca.cos(states[2]), controls[1] * ca.sin(states[2]), (controls[1] / self.wheelbase) * ca.tan(controls[0]), controls[2])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', self.nu, self.N)
        self.X = ca.MX.sym('X', self.nx, (self.N + 1))
        self.P = ca.MX.sym('P', self.nx + 2 * self.N + 1) # Parameters: init state and boundaries of the reference path

        '''Initialize upper and lower bounds for state and control variables'''
        self.lbg = np.zeros((self.nx * (self.N + 1) + self.N*2, 1))
        self.ubg = np.zeros((self.nx * (self.N + 1) + self.N*2, 1))
        self.lbx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
        self.ubx = np.zeros((self.nx + (self.nx + self.nu) * self.N, 1))
                
        x_min, y_min = np.min(self.rp.path, axis=0) - 2
        x_max, y_max = np.max(self.rp.path, axis=0) + 2
        s_max = self.rp.s_track[-1] *1.5
        lbx = np.array([[x_min, y_min, self.psi_min, 0]])
        ubx = np.array([[x_max, y_max, self.psi_max, s_max]])
        for k in range(self.N + 1):
            self.lbx[self.nx * k:self.nx * (k + 1), 0] = lbx
            self.ubx[self.nx * k:self.nx * (k + 1), 0] = ubx

        state_count = self.nx * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + self.nu, 0] = np.array([[-self.delta_max, self.v_min, self.p_min]]) 
            self.ubx[state_count:state_count + self.nu, 0] = np.array([[self.delta_max, self.v_max, self.p_max]])  
            state_count += self.nu

        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:self.nx]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            self.g = ca.vertcat(self.g, self.P[self.nx + 2 * k] * st_next[0] - self.P[self.nx + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  :represents path boundary constraints
        for k in range(self.N):
            force_lateral = self.U[1,k] **2 / self.L * ca.tan(ca.fabs(self.U[0,k])) *  self.mass
            self.g = ca.vertcat(self.g, force_lateral)


        self.J = 0  # Objective function
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.J = self.J + countour_error **2 * self.weight_contour  
            self.J = self.J + lag_error **2 * self.weight_lag
            self.J = self.J - self.U[2, k] * self.weight_progress 
            self.J = self.J - self.U[0, k] * self.weight_steering
            
        optimisation_variables = ca.vertcat(ca.reshape(self.X, self.nx * (self.N + 1), 1),
                                ca.reshape(self.U, self.nu * self.N, 1))

        nlp_prob = {'f': self.J,
                     'x': optimisation_variables,
                       'g': self.g,
                         'p': self.P}
        opts = {"ipopt": {"max_iter": 1500, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def build_initial_state(self, current_x):
        x0 = current_x
        x0[2] = self.normalise_psi(x0[2]) 
        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))

        return x0

    def generate_parameters(self, x0_in, x0_speed):
        p = np.zeros(self.nx + 2 * self.N + 1)
        p[:self.nx] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            
            right_x = self.rp.right_lut_x(s_progress).full()[0, 0]
            right_y = self.rp.right_lut_y(s_progress).full()[0, 0]
            left_x = self.rp.left_lut_x(s_progress).full()[0, 0]
            left_y = self.rp.left_lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y

            self.lbg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = min(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)
            self.ubg[self.nx - 1 + (self.nx + 1) * (k + 1), 0] = max(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)
            
            

            p[self.nx + 2 * k:self.nx + 2 * k + 2] = [-delta_x, delta_y]
            p[-1] = max(x0_speed, 1) # prevent constraint violation
            # p[-1] = x0_speed
        self.lbg[self.nx *2, 0] = - ca.inf
        self.ubg[self.nx *2, 0] = ca.inf
        for k in range(self.N):
            self.lbg[self.nx * (self.N + 1) + self.N + k] = -self.fmax
            self.ubg[self.nx * (self.N + 1) + self.N + k] = self.fmax


        return p

    def solve(self, p):

        x_init = ca.vertcat(ca.reshape(self.X0.T, self.nx * (self.N + 1), 1),
                         ca.reshape(self.u0.T, self.nu * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.X0 = ca.reshape(sol['x'][0:self.nx * (self.N + 1)], self.nx, self.N + 1).T
        controls = ca.reshape(sol['x'][self.nx * (self.N + 1):], self.nu, self.N).T

        # print(controls[0, 0], controls[0,1])


        # if self.solver.stats()['return_status'] != 'Solve_Succeeded':
        #     print("Solve failed!!!!!")

        return controls.full(), self.X0
        
    def construct_warm_start_soln(self, initial_state):
        if not self.warm_start: return
        # self.warm_start = False

        self.X0 = np.zeros((self.N + 1, self.nx))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_init * self.dt

            psi_next = self.rp.angle_lut_t(s_next).full()[0, 0]
            x_next, y_next = self.rp.center_lut_x(s_next), self.rp.center_lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2
            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

        # self.realTimePlot(self.x_bar, self.X0)

    def inputStateAdust(self,obs):

        if self.reset:
            x0 = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]
            self.prev_x0 = [0.0,0.0,0.0]
            self.reset = 0
        else:
            x0 = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]]


        if self.TESTMODE == "perception_noise":
            rand = np.random.normal(mu,sigma,1)
            self.scaledRand = rand*self.scale
            x0 = [obs['poses_x'][0]+self.scaledRand[0], obs['poses_y'][0]+self.scaledRand[0], obs['poses_theta'][0]]

        X0_slip = [obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], obs['linear_vels_x'][0]]

        return x0, X0_slip
    
    def outputActionAdjust(self,speed,steering):
        rand = np.random.normal(mu,sigma,1)
        self.scaledRand = rand*self.scale

        speed_mod = speed
        steering_mod = steering

        if self.TESTMODE == "Outputnoise_speed":
            speed_mod = speed + self.scaledRand[0]
        elif self.TESTMODE == "Outputnoise_steering":
            steering_mod = steering + self.scaledRand[0]

        return speed_mod, steering_mod
    
    def realTimePlot(self,x_bar,x_guess):
        plt.plot(self.wpts[:,0],self.wpts[:,1],"bx",markersize=1)
        plt.plot(x_bar[:,0], x_bar[:,1], 'bo', markersize=4, label="Solution States (x_bar)") 
        plt.plot(x_guess[:,0], x_guess[:,1], 'ro', markersize = 4, label = "inital guess")
        plt.legend()
        plt.pause(0.01)
        plt.clf()

    def normalise_psi(self,psi):
        while psi > np.pi:
            psi -= 2*np.pi
        while psi < -np.pi:
            psi += 2*np.pi
        return psi


