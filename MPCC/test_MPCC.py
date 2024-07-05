from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
from ConstantMPCC import MPCC
from FastMPCC import MPCC as fastMPCC
from TuningFastMPCC import MPCCTuning as MPCCT
import matplotlib.pyplot as plt
import numpy as np
import time 
import collections as co

MPCCMODE = "fast"           #Mode: "fast"; "constant"; "fastTuning"
UPDATE_PERIOD = 4
SAVELAPDATA = True
np.random.seed(0)

def main():
    map_name_list = ["gbr","esp","mco"]
    # map_name_list = ["mco"]

    '''Tuning'''
    # testmode_list = ["Tuning"]

    '''Experiments'''
    # testmode_list = ["Outputnoise_steering"]

    # testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering","control_delay_speed","control_Delay_steering","perception_delay"]
    # testmode_list = ["Benchmark","perception_noise","Outputnoise_speed","Outputnoise_steering"]
    testmode_list = ["control_delay_speed","control_Delay_steering","perception_delay"]     

    for map_name in map_name_list:
        print("new map, " + map_name)
        for TESTMODE in testmode_list:
            print("new mode, " + TESTMODE)
    
            if MPCCMODE == "constant":
                planner = MPCC(map_name)

            if MPCCMODE == "fast":
                planner = fastMPCC(map_name,TESTMODE)

            if MPCCMODE == "fastTuning":
                planner = MPCCT(map_name,TESTMODE)


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

            time_delay = 0
            iter_count = 0
            reset_count = 0
            lapCount = 0
            collision_count = 0
            laptime = 0.0
            init_pos = 1

            env = gym.make('f110_gym:f110-v0', map='./maps/'+map_name, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
            obs, step_reward, done, info = env.reset(np.array([[planner.waypoints[init_pos][1], planner.waypoints[init_pos][2],planner.waypoints[init_pos][4]]]))
            new_obs = obs
            planner.reset = 1
            

            env.add_render_callback(render_callback)
            env.render()
            computation_time_start = time.time()


            new_speed,new_steering_angle = planner.plan(obs,laptime)
            control = [new_speed, new_steering_angle]
            control_queue, obs_queue = initqueue(obs,control,time_delay)
            planner.reset = 1


            while iter_count < planner.Max_iter:
                if (lapCount+obs['lap_counts']+collision_count+reset_count) != iter_count or  obs['collisions'] or new_obs['collisions']:
                    # print((lapCount+obs['lap_counts']+collision_count))
                    # print(iter_count)
                    computation_time = time.time() - computation_time_start
                    lap_success = 1
                    planner.scale = iter_count // 10 * 0.02
                    iter_count += 1

                    if obs['collisions'] or new_obs['collisions']:
                        print("Iter_count = ", iter_count, "I crashed, completion Percentage is", int(planner.completion),"%")
                        lap_success = 0
                        collision_count += 1
                        lapCount += obs['lap_counts'][0]  
                        if TESTMODE == "control_delay_speed" or TESTMODE == "control_Delay_steering" or TESTMODE == "perception_delay":
                            rand_start_x = np.random.normal(0,0.1,1)
                            rand_start_y = np.random.normal(0,0.1,1)
                            obs, _, _, _ = env.reset(np.array([[planner.waypoints[init_pos][1]+rand_start_x, planner.waypoints[init_pos][2]+rand_start_y,planner.waypoints[init_pos][4]]]))
                        else:
                            obs, _, _, _ = env.reset(np.array([[planner.waypoints[init_pos][1], planner.waypoints[init_pos][2],planner.waypoints[init_pos][4]]]))
                        reset_count-=1
                        planner.reset = 1
                    else:
                        print("Iter_count = ", iter_count, "laptime = ", laptime)                          

                    if TESTMODE == "Benchmark":
                        var1 = 0
                        var2 = 0
                    if TESTMODE == "perception_noise":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,6])
                    if TESTMODE == "Outputnoise_speed":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,6])
                    if TESTMODE == "Outputnoise_steering":
                        var1 = planner.scale
                        var2 = max(planner.ds.txt_x0[:,6])
                    if TESTMODE == "control_delay_speed" or TESTMODE == "control_Delay_steering" or TESTMODE == "perception_delay":
                        var1 = time_delay*10
                        var2 = 0
                    if TESTMODE == "Tuning":
                        parameter_list = ["dt", "N", "weight_progress", "weight_lag", "weight_contour", "weight_steering"]
                        var1 = planner.dt
                        var2 = 0
                        planner.dt +=0.02
                    aveTrackErr = np.mean(planner.ds.txt_x0[:,5])

                    if SAVELAPDATA:
                        planner.ds.savefile(iter_count)

                    planner.ds.lapInfo(iter_count,lap_success,laptime,planner.completion,var1,var2,aveTrackErr,computation_time)
                    laptime = 0.0
                    computation_time_start = time.time()
                    if TESTMODE == "control_delay_speed" or TESTMODE == "control_Delay_steering" or TESTMODE == "perception_delay":
                        time_delay = iter_count // 10
                        new_speed,new_steering_angle = planner.plan(obs,laptime)
                        control = [new_speed, new_steering_angle]
                        control_queue, obs_queue = initqueue(obs,control,time_delay)
                    obs, _, _, _ = env.reset(np.array([[planner.waypoints[init_pos][1], planner.waypoints[init_pos][2],planner.waypoints[init_pos][4]]]))
                    reset_count+=1


                if TESTMODE == "perception_delay":
                    speed,steering_angle = planner.plan(obs_queue[0],laptime)
                else:
                    speed,steering_angle = planner.plan(obs,laptime)
                    control = [speed, steering_angle]

                z = UPDATE_PERIOD
                while z > 0:
                    
                    if TESTMODE == "control_delay_speed":
                        control_queue.append(control)
                        obs, _, _, _ = env.step(np.array([[steering_angle, control_queue[0][0]]]))
                    elif TESTMODE == "control_Delay_steering":
                        control_queue.append(control)
                        obs, _, _, _ = env.step(np.array([[control_queue[0][1], speed]]))
                    else:
                        if TESTMODE == "perception_delay":
                            new_obs, _, _, _ = env.step(np.array([[steering_angle,speed]]))
                            obs_queue.append(new_obs)
                        else:
                            obs, _, _, _ = env.step(np.array([[steering_angle, speed]]))
                    z -= 1
                    laptime += 0.01
                # env.render(mode='human_fast') #'human_fast'(without delay) or 'human' (with 0.005 delay)
            planner.ds.saveLapInfo()



def initqueue(obs, control, time_delay):
    control_queue = co.deque(maxlen = time_delay+1)
    for i in range(0, time_delay+1):
        control_queue.append(control)

    obs_queue = co.deque(maxlen = time_delay+1)
    for i in range(0, time_delay+1):
        obs_queue.append(obs)

    return control_queue, obs_queue
        
if __name__ == '__main__':
    main()