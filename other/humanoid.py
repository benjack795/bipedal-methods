import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import sqrt

def mass_center(model):
    mass = model.body_mass[:14]
    xpos = model.data.xipos[:14,:]
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    
    collected_data = np.array([])
    m = 1
    flipper = False
    circlecheck = False
    hfield = False
    turner = False
    comeback = False
    counters = np.zeros(17)
    knees = False
    hips = False
    shoulders = False
    footy = False

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoid.xml', 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.model.data
        #does the network need the inertia, velocity and forces acting on the ball? no
        #slicing the q-types should not affect the system as they describe the root joint
        #gait appears good without
        return np.concatenate([data.qpos.flat[2:24], #22 without, 29 with
                               data.qvel.flat[:23], #23 without 29 with
                               data.cinert[:14].flat,
                               data.cvel[:14].flat,
                               data.qfrc_actuator.flat[:23], #23 without 29 with
                               data.cfrc_ext[:14].flat])
	


    def _step(self, a):
        pos_before = mass_center(self.model)
        self.do_simulation(a, self.frame_skip)
	
        #foot circle code
        leftcircle = False
        # get the two feet
        leftfootpos = self.model.data.geom_xpos[11]
        rightfootpos = self.model.data.geom_xpos[8]
        # get the com
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
        
        #turning/coming back code
        #start with reward being distance to x,y
        distpos = [8, 8]
        distpoint = sqrt( ((distpos[0] - cmass[0])**2) + ((distpos[1] - cmass[1])**2) )
        distreward = -distpoint/2
        #if position is near 8,8 flip a boolean to true
        if(self.turner == True):
            if(distpoint < 0.5):
                self.comeback = True
                self.turner = False
        #as long as this boolean is true, reward is distance to 0,0
        distorigin = sqrt( ((0 - cmass[0])**2) + ((0 - cmass[1])**2) )
        distoriginreward = -distorigin/2
        #check these work and run a test
        
        pos_after = mass_center(self.model)       
        alive_bonus = 5.0
        data = self.model.data

        #joint bonuses code
	#0.4s seem to be the default max, use these to apply bonuses
       
        #hipys
        hipybonus = 0
        if(np.absolute(data.actuator_force[5]) > 0.3):
            hipybonus = hipybonus + 2
        if(np.absolute(data.actuator_force[9]) > 0.3):
            hipybonus = hipybonus + 2
       
        #knees
        kneebonus = 0
        if(np.absolute(data.actuator_force[6]) > 0.3):
            kneebonus = kneebonus + 2
        if(np.absolute(data.actuator_force[10]) > 0.3):
            kneebonus = kneebonus + 2
        
        #shoulder joints
        shoulbonus = 0
        if(np.absolute(data.actuator_force[11]) > 0.3):
            shoulbonus = shoulbonus + 1
        if(np.absolute(data.actuator_force[12]) > 0.3):
            shoulbonus = shoulbonus + 1
        if(np.absolute(data.actuator_force[14]) > 0.3):
            shoulbonus = shoulbonus + 1
        if(np.absolute(data.actuator_force[15]) > 0.3):
            shoulbonus = shoulbonus + 1

        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext[:14]).sum()
        quad_impact_cost = min(quad_impact_cost, 10)

        # typical values
        # vel ~2 ctrl ~0.3 impact ~0.1-0.9 alive 5 dist would be -4 tops
        reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus
        if(self.flipper == True):
            reward = lin_vel_cost - (0.25*quad_ctrl_cost) - quad_impact_cost + alive_bonus
        if(self.turner == True):
            reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + distreward + alive_bonus
        if(self.comeback == True):
            reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + distoriginreward + alive_bonus
        if(self.hips == True):
             reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus + hipybonus
        if(self.knees == True):
             reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus + kneebonus
        if(self.shoulders == True):
             reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus + shoulbonus
        if(self.footy == True):
             ballbonus = 0
             ##if walker is 1 from ball, bonus is 4
             ballpos = self.model.data.geom_xpos[18]
             balldist = sqrt( ((ballpos[0] - cmass[0])**2) + ((ballpos[1] - cmass[1])**2) )
             if(balldist < 1):
                 ballbonus = ballbonus + 4
             reward = lin_vel_cost - (quad_ctrl_cost) - quad_impact_cost + alive_bonus + ballbonus
        qpos = self.model.data.qpos
        vertdone = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        if (self.hfield == True):
            averagefoot = (leftfootpos[2] + rightfootpos[2])/2
            vertdone = bool((qpos[2]-averagefoot) < 1.0)
        done = vertdone
        #done = False
        #distance
        distrecord = sqrt(cmass[0]**2 + cmass[1]**2)   

        #print("foot {}".format(leftfootpos[2]))

        self.collected_data = np.append(self.collected_data,[lin_vel_cost, quad_ctrl_cost, data.actuator_force[0], data.actuator_force[1], data.actuator_force[2], data.actuator_force[3], data.actuator_force[4], data.actuator_force[5], data.actuator_force[6], data.actuator_force[7], data.actuator_force[8], data.actuator_force[9], data.actuator_force[10], data.actuator_force[11], data.actuator_force[12], data.actuator_force[13], data.actuator_force[14], data.actuator_force[15], data.actuator_force[16], distrecord], axis=0)
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
