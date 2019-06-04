import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import sqrt

def mass_center(model):
    mass = model.body_mass[:14]
    xpos = model.data.xipos[:14,:]
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    #parameters for collecting data, delayed multiplier flipper, radial constraint check and multiplier and rough ground hfield
    collected_data = np.array([])
    m = 1
    flipper = False
    circlecheck = False
    hfield = False

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:24], 
                               data.qvel.flat[:23], 
                               data.cinert[:14].flat,
                               data.cvel[:14].flat,
                               data.qfrc_actuator.flat[:23], 
                               data.cfrc_ext[:14].flat])
	


    def _step(self, a):
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
	
        #radial constraint
        leftcircle = False
        # get the two feet
        leftfootpos = self.model.data.geom_xpos[11]
        rightfootpos = self.model.data.geom_xpos[8]
        # *****THIS IS NOT ACCURATE IN THE PAPER (many apologies) 
        # THIS IS NOT THE CENTRE OF MASS BUT THE LOCATION OF THE ROOT JOINT (a com function is above)
        cmass = self.model.data.qpos[:2]
        # work out midpoint between feet 
        midpoint = [(leftfootpos[0] + rightfootpos[0])/2, (leftfootpos[1] + rightfootpos[1])/2]
        # work out radius via half the distance between feet
        radius = sqrt( ((midpoint[0] - leftfootpos[0])**2) + ((midpoint[1]-leftfootpos[1])**2) )
        # work out distance from com to midpoint
        comdist = sqrt( ((cmass[0] - midpoint[0])**2) + ((cmass[1] - midpoint[1])**2) )
        # is it outside the circle?
        if(comdist > (self.m*radius)):
            leftcircle = True
               
        pos_after = mass_center(self.model)       
        alive_bonus = 5.0
        data = self.model.data

        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext[:14]).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

	#reward function code (if the flipper has been activated the multiplier is applied)
        reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus
        if(self.flipper == True):
            reward = lin_vel_cost - (0.25*quad_ctrl_cost) - quad_impact_cost + alive_bonus
      
        qpos = self.model.data.qpos
        vertdone = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
	#applying height correction if using a hfield (we cannot rely on the position of the ground)
        if (self.hfield == True):
            averagefoot = (leftfootpos[2] + rightfootpos[2])/2
            vertdone = bool((qpos[2]-averagefoot) < 1.0)
        done = vertdone
	
        distrecord = sqrt(cmass[0]**2 + cmass[1]**2)   

        self.collected_data = np.append(self.collected_data,[lin_vel_cost, quad_ctrl_cost, data.actuator_force[0], data.actuator_force[1], data.actuator_force[2], data.actuator_force[3], data.actuator_force[4], data.actuator_force[5], data.actuator_force[6], data.actuator_force[7], data.actuator_force[8], data.actuator_force[9], data.actuator_force[10], data.actuator_force[11], data.actuator_force[12], data.actuator_force[13], data.actuator_force[14], data.actuator_force[15], data.actuator_force[16], distrecord], axis=0)
        #confirming the radial constraint
	if(self.circlecheck == True):
            done = (leftcircle or vertdone) 
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20
