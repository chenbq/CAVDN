import numpy as np
from env.multiagent_com.core import World, Agent, Landmark
from types import SimpleNamespace

N_neighbor = 20

config = {
    'UAV_n':4,
    'UE_n':50,
    'range':50,
    'd_max':20,
    'noise': np.power(10, -100/10),
    'P_ref': 500*np.power(10, -37.6/10),# -37.6 reference at 1m loss,
    'alpha': -2,
    'height': 50, #m,
    'p_los_B': 0.35,
    'p_los_C': 5,
    'data_rate_requirement': 0.05,  # np.array([0.1, 0.2, 0.3, 0.4]),
    'lambda_ratio': 5,
    'P_blada_power': 0.012/8*1.225*0.05*0.79*np.power(400*0.5,3),# delta/8*rho*s*A*omega^3*R^3,
    'U_tip_speed_2': 200*200
    }
config = SimpleNamespace(**config)

class Scenario(object):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.collaborative = True
        # add agents
        world.agents = [Agent() for i in range(config.UAV_n)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 20
        # add landmarks
        world.landmarks = [Landmark() for i in range(config.UE_n)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 1.6
        # make initial conditions
        self.reset_world(world)
        return world

    # initialize the state of all agents and landmark
    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            #agent.color = np.array([0.35, 0.35, 0.85])
            agent.color = np.array([2/255.0, 162/255.0, 241/255.0])
            #agent.color = np.array([0.70, 0.83, 0.55])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            #landmark.color = np.array([248/255.0, 198/255.0, 149/255.0])

        # set random initial states
        pos_list = [np.array([-40,-40],dtype=np.float), np.array([40,-40],dtype=np.float),
                    np.array([-40,40],dtype=np.float) , np.array([40,40],dtype=np.float)]
        for i, agent in enumerate(world.agents):
            # agent.state.p_pos = np.array([i//2*config.range-config.range/2, i%2*config.range-config.range/2 ],dtype=np.float)
            agent.state.p_pos = pos_list[i]
            #agent.state.p_pos = np.random.uniform(-500, +500, world.dim_p)
            #np.random.uniform(-100, +100, world.dim_p)
            #agent.state.p_pos = np.array([-500, 500], dtype=np.float)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.n_serv = 0
            agent.state.a_id = int(0)

        for i, landmark in enumerate(world.landmarks):
           landmark.state.p_pos = np.random.uniform(-config.range, +config.range, world.dim_p)
           landmark.state.p_vel = np.zeros(world.dim_p)
           landmark.state.n_serv = 0

    # 用于最终统计实际表现的结果
    # def benchmark_data(self, agent, world):
    #     rew = 0
    #     collisions = 0
    #     occupied_landmarks = 0
    #     min_dists = 0
    #     for l in world.landmarks:
    #         dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
    #         min_dists += min(dists)
    #         rew -= min(dists)
    #         if min(dists) < 0.1:
    #             occupied_landmarks += 1
    #     if agent.collide:
    #         for a in world.agents:
    #             if self.is_collision(a, agent):
    #                 rew -= 1
    #                 collisions += 1
    #     return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        #print(agent1.state.p_pos)
        #print(agent2.state.p_pos)
        # 两者距离太近
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_out_range(self, agent):
        return True if np.max(np.abs(agent.state.p_pos))>config.range else False

    def reward(self, agent, world):
        # First part
        # average and minimal data rate
        rew = 0

        acc_mat = np.zeros([config.UE_n, config.UAV_n])
        dis_mat = np.array([[np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents] for l in world.landmarks])
        dis_mat_idx_descend = dis_mat.argsort(0) # for given UAV the order of users.
        for idx_a, a in enumerate(world.agents):
            #check whether distance smaller than the threshold
            if dis_mat[dis_mat_idx_descend[a.state.a_id, idx_a], idx_a]<= config.d_max:
                acc_mat[dis_mat_idx_descend[a.state.a_id, idx_a] ,idx_a] = 1.0

        # check the overlap of service of UAVs to the same user.
        n_access_UAV = acc_mat.sum(1)
        for idx_l, l in enumerate(world.landmarks):
            if n_access_UAV[idx_l] > 1:
                UAV_lst = np.where(acc_mat[idx_l,:]==1.0)
                inner_idx = np.argmin(dis_mat[idx_l][UAV_lst]) # UAV_lst内部最近的idx
                acc_mat[idx_l, :] = 0
                # big bug here inner_idx should be transferred into UAV_lst[inner_idx]
                acc_mat[idx_l, UAV_lst[0][inner_idx]] = 1.0


        sum_data_rate = 0
        # compute the data rate of different user

        # parameters
        noise = config.noise
        P_ref = config.P_ref
        alpha = config.alpha
        height = config.height
        p_los_B = config.p_los_B
        p_los_C = config.p_los_C
        n_access_UAV = acc_mat.sum(1)
        # sum_data_rate_lst = []
        for idx_l, l in enumerate(world.landmarks):
            if n_access_UAV[idx_l] > 0:
                # find the serving UAV index
                idx_a = np.where(acc_mat[idx_l, :] == 1.0)[0]
                dist = np.sqrt(np.square(dis_mat[idx_l, idx_a]) + np.square(height))
                # compute the channel with given distanc
                P_los_pro = 1.0 / (  1.0 + p_los_C* np.exp(-p_los_B*( 180/np.pi*np.arctan(height/dist) - p_los_C) )  )
                channel_loss = P_los_pro* np.power(10, -3/10) + (1-P_los_pro)* np.power(10, -23/10)
                signal = P_ref*np.power(dist,alpha)*channel_loss
                # sum_data_rate_lst.append(np.log2(1 + signal / noise))
                sum_data_rate += np.log2(1 + signal / noise)
        # print(sum_data_rate)


        # compute the fairness of UEs and BSs
        n_serv_UE = np.array([l.state.n_serv for l in world.landmarks])
        UE_fair_mole = np.square(n_serv_UE.sum())
        UE_fair_deno = np.square(n_serv_UE).sum()*config.UE_n
        if UE_fair_deno>0:
            UE_fair = UE_fair_mole/UE_fair_deno
        else:
            UE_fair = 0
        n_serv_UAV = np.array([a.state.n_serv for a in world.agents])
        UAV_fair_mole = np.square(n_serv_UAV.sum())
        UAV_fair_deno = np.square(n_serv_UAV).sum() * config.UAV_n
        if UAV_fair_deno > 0:
            UAV_fair = UAV_fair_mole / UAV_fair_deno
        else:
            UAV_fair = 0

        # print(UAV_fair)
        # print(UE_fair)
        rew += sum_data_rate*UE_fair*UAV_fair

        # considering the collision and out of range
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 1/2
        for a in world.agents:
            if self.is_out_range(a):
                rew -= 1 / 2
        return rew

    def observation(self, agent, world):
        #获取当前的输出
        # get positions of all entities in this agent's reference frame
        range_limt = config.range*2
        entity_pos = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.landmarks]
        idx_nearby = np.argsort(dists)
        #idx_nearby = idx_nearby[:3]
        idx_nearby = idx_nearby[:N_neighbor]
        landmarks_nearby = [world.landmarks[i] for i in idx_nearby]
        for entity in landmarks_nearby:  # world.entities:
            #entity_pos.append( entity.state.p_pos - agent.state.p_pos)
            entity_pos.append( (entity.state.p_pos - agent.state.p_pos)/range_limt)
        UE_serv_features = [world.landmarks[i].state.n_serv for i in idx_nearby]
        UE_serv_features = np.array(UE_serv_features)/(np.max(UE_serv_features)+0.000001)
        UE_serv_features = UE_serv_features.tolist()

        # adding the other UAV position
        other_pos = []
        dists = [np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos))) for l in world.agents]
        idx_nearby = np.argsort(dists)
        agents_nearby = [world.agents[i] for i in idx_nearby]
        for other in agents_nearby:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.state.p_pos - agent.state.p_pos)/range_limt)
        phy_feature = np.concatenate([agent.state.p_vel/agent.max_speed] + [agent.state.p_pos/range_limt] +  other_pos + entity_pos).tolist()
        #phy_feature = np.concatenate([agent.state.p_pos/range_limt] +  other_pos).tolist()

        # adding the serv_number as features
        UAV_serv_features = [a.state.n_serv for a in world.agents]
        UAV_serv_features = np.array(UAV_serv_features) / (np.max(UAV_serv_features)+0.000001)
        UAV_serv_features = UAV_serv_features.tolist()


        return np.array(phy_feature + UAV_serv_features + UE_serv_features)


if __name__ == '__main__':
    from env.multiagent_com.environment import MultiAgentEnv
    import env.multiagent_com.scenarios as scenarios
    import time
    scenario = scenarios.load('uav_com' + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation)
    env.reset()
    for i in range(25):
        ation_n =[[0,np.random.randint(0,9,1)] for j in range(config.UAV_n)]
        obs_n, reward_n, done_n, info_n = env.step(ation_n)
        env.render()
        time.sleep(0.5)
        print([a.state.n_serv for a in world.landmarks])
        print(reward_n)
