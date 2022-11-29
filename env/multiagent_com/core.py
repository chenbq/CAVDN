import numpy as np

# customed for the UAV communicaitons scenario
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # number of serve/served count
        self.n_serv = None


class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        self.a_id = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # 用户选择
        self.a = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        #self.max_speed = None
        #★★★★★★#
        # 02-16 changed to 10
        self.max_speed = 20
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # position dimensionality
        self.dim_p = 2
        # 接入用户数的纬度，在最近的3个用户中选择
        self.dim_a = 3
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        #self.dt = 0.1
        #★★★★★★★★#
        self.dt = 1
        # ★★★★★★★★#
        # physical damping
        self.damping = 0.4
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    #@property
    # def agents(self):
    #     return [agent for agent in self.agents]

    # update the association for
    def step_access(self):
        for i,agent in enumerate(self.agents):
            self.update_agent_access_state(agent)

    def step_move(self):
        # update agent state
        for i, agent in enumerate(self.agents):
            self.integrate_position_state(agent)

    # update state of the world
    def step(self):
        # update agent state
        for i,agent in enumerate(self.agents):
            self.update_agent_access_state(agent)
            self.integrate_position_state(agent)


    # integrate physical state
    def integrate_position_state(self,agent):
        #v_r = agent.action.u[0]*agent.max_speed
        #theta = agent.action.u[1]*3.14
        agent.state.p_vel[0] = agent.action.u[0]*agent.max_speed
        agent.state.p_vel[1] =  agent.action.u[1]*agent.max_speed
        agent.state.p_pos += agent.state.p_vel * self.dt

        # 2021-1-13 clip into the region
        # agent.state.p_pos = np.clip(agent.state.p_pos, -50.0, 50.0) # 1-11 clip the active area

    def integrate_position_state_phy(self, agent):
        agent.state.p_vel = agent.state.p_vel * (1 - self.damping)
        agent.state.p_vel += (agent.action.u / agent.mass) * self.dt
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
            if speed > agent.max_speed:
                agent.state.p_vel = agent.state.p_vel / np.sqrt(np.square(agent.state.p_vel[0]) +
                                                              np.square(agent.state.p_vel[1])) * agent.max_speed
        agent.state.p_pos += agent.state.p_vel * self.dt
        # agent.state.p_pos = np.clip(agent.state.p_pos, -50.0, 50.0)  # 1-11 clip the active area

    def update_agent_access_state(self, agent):
        #agent.state.power = (agent.action.p + 1)/4 + 0.5
        # 均为int类型
        agent.state.a_id = agent.action.a


