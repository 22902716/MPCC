from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
from my_MPCC import MPCC
import matplotlib.pyplot as plt
import numpy as np

REALTIME_VERBOSE = True
# REALTIME_VERBOSE = False

# REALTIME_VERBOSE_temp = True
REALTIME_VERBOSE_temp = False

def main():
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
        # obs, step_reward, done, info = env.reset(np.array([[planner.X0[0], planner.X0[1], planner.X0[2]]]))
        obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))

    env.add_render_callback(render_callback)
    env.render()


    if REALTIME_VERBOSE:
        plt.figure()
    if REALTIME_VERBOSE_temp:
        plt.figure()

    while not done:
        x0 = [obs['poses_x'][0], obs['poses_y'][0],obs['poses_theta'][0]]
        steering_angle,speed,x_bar = planner.plan(x0)
        # print("steering : ",steering_angle)
        # print(x_ref)
        z = 2
        while z > 0:
            # obs, _, done, _ = env.step(np.array([[0., speed*speedgain]]))
            obs, _, done, _ = env.step(np.array([[steering_angle, speed]]))
            z-=1
        env.render(mode='human_fast')

        # xbar_x.append(x_bar[0,:])
        # xbar_y.append(x_bar[1,:])
        # xref_x.append(x_ref[0:planner.N+1,0])    
        # xref_y.append(x_ref[0:planner.N+1,1])
        
        # print(xbar_x)
        # print(xbar_y)
        # print(xref_x)
        # print(xref_y)

        counter += 1
    

        if REALTIME_VERBOSE:
            plt.plot(planner.wpts[:,0],planner.wpts[:,1],"bx",markersize=1)
            # plt.plot(planner.centerline[:,0],planner.centerline[:,1],"bx",markersize=1)
            plt.plot(x_bar[:,0], x_bar[:,1], 'bo', markersize=4, label="Solution States (x_bar)") 
            # plt.plot(x_ref[:,0],x_ref[:,1],'bx',label = "x_ref")
            # print(planner.inner_bound)
            # plt.plot(planner.inner_bound[:,0],planner.inner_bound[:,1],'bx',markersize = 3)
            # plt.plot(planner.outer_bound[:,0],planner.outer_bound[:,1],'rx',markersize = 3)
            plt.pause(0.01)
            plt.clf()


        
        
if __name__ == '__main__':
    main()