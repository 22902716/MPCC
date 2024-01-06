from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
from ConstantMPCC import MPCC as MPCC
from FastMPCC import MPCC as fastMPCC
import matplotlib.pyplot as plt
import numpy as np

MPCCMODE = "fast"           #Mode: "Fast"; "constant";

REALTIME_VERBOSE = True
# REALTIME_VERBOSE = False

# REALTIME_VERBOSE_temp = True
REALTIME_VERBOSE_temp = False

def main():
    map_name_list = ["gbr"]
    map_name = map_name_list[0]       
    counter = 0
    
    if MPCCMODE == "constant":
        planner = MPCC(map_name_list[0])

    if MPCCMODE == "fast":
        planner = fastMPCC(map_name_list[0])


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

    env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    # obs, step_reward, done, info = env.reset(np.array([[planner.X0[0], planner.X0[1], planner.X0[2]]]))
    obs, step_reward, done, info = env.reset(np.array([[0, 0, 0]]))

    env.add_render_callback(render_callback)
    env.render()


    if REALTIME_VERBOSE:
        plt.figure()

    while not done:
        
        steering_angle,speed,x_bar = planner.plan(obs)

        z = 2
        while z > 0:
            obs, _, done, _ = env.step(np.array([[steering_angle, speed]]))
            z-=1
        env.render(mode='human_fast')

        counter += 1


        
        
if __name__ == '__main__':
    main()