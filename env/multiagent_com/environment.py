import gym
from gym import spaces
import numpy as np
from env.multiagent_com.multi_discrete import MultiDiscrete
D_max = 20
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.agents
        # set required vectorized gym env property
        self.n_a = len(world.agents)
        self.n_l = len(world.landmarks)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True #True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = True#world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            if agent.movable:
                total_action_space.append(u_action_space)

            a_action_space = spaces.Discrete(world.dim_a)
            total_action_space.append(a_action_space)

            # total action space
            if len(total_action_space) > 1:
                act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space

            self.obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim ,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
            agent.action.a = int(0)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n_a
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        #info_n = []
        #info_n = {'n': []}
        self.agents = self.world.agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            #self._set_access_action(action_n[i], agent, self.action_space[i])
            self._set_move_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record obserstevation for each agent
        reward_n.append(self._get_reward(self.agents[0]))
        reward_n = [reward_n] * self.n_a
        info_n = self._get_evl_data()
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))

            done_n.append(self._get_done(agent))

            #info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        #reward = np.sum(reward_n)
        #if self.shared_reward:
        #    reward_n = [reward] * self.n_a

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        #if self.info_callback is None:
        return {}
        #return self.info_callback(agent, self.world)

    def _get_evl_data(self):
        if self.info_callback is None:
            return {}
        return self.info_callback(self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)


    def _set_access_action(self, action, agent, action_space, ):

        # access action at 1
        action = action[0]

        agent.action.a = action
        # set the id from outside
        agent.state.a_id = agent.action.a


    # set env action for a particular agent
    def _set_move_action(self, action, agent, action_space, time=None):
        # set the speed
        action_speed = action[0]
        speed_ratio = 0.0
        #if action_speed == 1: speed_ratio = 0.25
        if action_speed == 1: speed_ratio = 0.5
        #if action_speed == 3: speed_ratio = 0.75
        if action_speed == 2: speed_ratio = 1

        # access action at 1
        action = action[1]
        action += 1
        agent.action.u = np.zeros(self.world.dim_p)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            # process discrete action
            if action == 1: agent.action.u[0] = -1.0
            if action == 2: agent.action.u[0] = +1.0
            if action == 3: agent.action.u[1] = -1.0
            if action == 4: agent.action.u[1] = +1.0
            if action == 5:
                agent.action.u[0] = -1.0*np.sqrt(0.5)
                agent.action.u[1] = -1.0*np.sqrt(0.5)
            if action == 6:
                agent.action.u[0] = +1.0*np.sqrt(0.5)
                agent.action.u[1] = -1.0*np.sqrt(0.5)
            if action == 7:
                agent.action.u[0] = -1.0*np.sqrt(0.5)
                agent.action.u[1] = +1.0*np.sqrt(0.5)
            if action == 8:
                agent.action.u[0] = +1.0*np.sqrt(0.5)
                agent.action.u[1] = +1.0*np.sqrt(0.5)
            sensitivity = 1.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity*speed_ratio



    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from env.multiagent_com import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from env.multiagent_com import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.landmarks:
                geom = rendering.make_circle(entity.size)
                #geom = rendering.Image(r'C:\Users\Alex_sheep\Desktop\user.png',25*2,25*2)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color, alpha=0.7)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)


        # Add the agents and their movement
        self.time += 1
        self.time = self.time%20
        from env.multiagent_com import rendering
        self.render_geoms_a = []
        self.render_geoms_xform_a = []
        for entity in self.world.agents:
            if self.time !=39:
                geom = rendering.make_circle(entity.size)
                if 'agent' in entity.name:
                    #geom.set_color(*entity.color, alpha=0.5)
                    geom.set_color(*entity.color, alpha=self.time/25.0*0.6 + 0.2 )
                else:
                    geom.set_color(*entity.color)
            else:
                geom = rendering.Image(r'C:\Users\Alex_sheep\Desktop\uav_1.png', 8 * 2, 8 * 2)
            xform = rendering.Transform()
            geom.add_attr(xform)
            self.render_geoms_a.append(geom)
            self.render_geoms_xform_a.append(xform)

        # add geoms to viewer
        for viewer in self.viewers:
            #viewer.geoms = []
            for geom in self.render_geoms_a:
                viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from env.multiagent_com import rendering
            # update bounds to center around agent
            #######################################################
            ########################here change the cam_range###############################
            #######################################################
            # ************************** #
            #cam_range = 1
            cam_range = 60
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.landmarks):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            for e, entity in enumerate(self.world.agents):
                self.render_geoms_xform_a[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results
