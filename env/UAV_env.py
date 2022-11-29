
import time



from gym import spaces
import numpy as np
from types import SimpleNamespace
from smac.env.multiagentenv import MultiAgentEnv



class UAVEnv():
    def __init__(self):
        from env.multiagent_com.environment import MultiAgentEnv
        import env.multiagent_com.scenarios as scenarios
        scenario = scenarios.load('uav_com_u_centric' + ".py").Scenario()
        world = scenario.make_world()
        self.env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                        scenario.observation, scenario.eval_data)
        self.obs_dim = self.env.obs_dim
        self.n_agents = len(self.env.agents)
        # number of
        self.a_dim = 24
        self.episode_limit = 50

    def dec_to_base(self, num, base):  # Maximum base - 36
        base_num = []
        if num == 0:
            base_num = [0,0]
        while num > 0:
            dig = int(num % base)
            if dig < 10:
                base_num.append(dig)
            else:
                base_num.append(ord('A') + dig - 10)   # Using uppercase letters
            num //= base
        base_num = base_num[::-1]  # To reverse the string
        if len(base_num)==1:
            base_num =[0] + base_num
        return base_num

    def trans_aciton(self, actions):
        #
        return [self.dec_to_base(action, 8) for action in actions]

    def step(self, actions):
        """Returns reward, terminated, info."""
        ation_n = self.trans_aciton(actions)
        _, reward_n, done_n, info_n = self.env.step(ation_n)
        return np.array(reward_n).mean(), done_n[0], info_n

    def get_obs(self):
        """Returns all agent observations in a list."""
        obs_n = []
        for agent in self.env.agents:
            obs_n.append(self.env._get_obs(agent))
        return obs_n

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        return self.env._get_obs(self.env.agents[agent_id])

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_dim

    def get_state(self):
        """Returns the global state."""
        state = []
        for i in range(self.n_agents):
            state += self.get_obs_agent(i).tolist()
        state = np.array(state)
        return state

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.obs_dim*self.n_agents

    def get_avail_actions(self):
        """Returns the available actions of all agents in a list."""
        actions_n = []
        for i in range(self.n_agents):
            actions_n.append(self.get_avail_agent_actions(i))
        raise actions_n

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id."""
        return [1 for i in range(self.a_dim)]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.a_dim

    def reset(self):
        """Returns initial observations and states."""
        self.env.reset()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        """Save a replay."""
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
if __name__=='__main__':
    env = UAVEnv()
    reward_c = 0
    for i in range(50):
        ation_n =[np.random.randint(0,24,1) for j in range(4)]
        reward_n, done_n, info_n = env.step(ation_n)
        reward_c += reward_n
        env.render()
        time.sleep(0.5)
        #print([a.state.n_serv for a in env.env.world.landmarks])
        #print(reward_n)
    print(reward_c)