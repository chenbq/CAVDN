import numpy as np
from env.multiagent_com.core import World, Agent, Landmark
from types import SimpleNamespace

N_neighbor = 6
A_neighbor = 3
config = {
    'UAV_n':4,
    'UE_n':12,
    'range':50,
    'd_max':10,
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
            agent.size = 3
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
        color_list = [np.array([1., 0., 0.]),
                      np.array([0., 1., 0.]),
                      np.array([0., 0., 1.]),
                      np.array([1., 1., 0.]),
                      np.array([0., 1., 1.]),
                      np.array([1., 0., 1.]),
                      np.array([0.75, 0.75, 0.75])
                      ]
        world.reward_calc_n = 0
        for i, agent in enumerate(world.agents):
            #agent.color = np.array([0.35, 0.35, 0.85])
            # agent.color = np.array([2/255.0, 162/255.0, 241/255.0])
            agent.color = color_list[i]
            #agent.color = np.array([0.70, 0.83, 0.55])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            #landmark.color = np.array([248/255.0, 198/255.0, 149/255.0])

        # set random initial states
        #np.array([0, 0], dtype=np.float),
        pos_list = [np.array([-40,-40],dtype=np.float), np.array([40,-40],dtype=np.float),
                    np.array([-40,40],dtype=np.float) , np.array([40,40],dtype=np.float)]
        for i, agent in enumerate(world.agents):
            # agent.state.p_pos = np.array([i//2*config.range-config.range/2, i%2*config.range-config.range/2 ],dtype=np.float)
            agent.state.p_pos = pos_list[i]
            #agent.state.p_pos = np.random.uniform(-500, +500, world.dim_p)
            #np.random.uniform(-100, +100, world.dim_p)
            #agent.state.p_pos = np.array([-500, 500], dtype=np.float)
            # 改为随机的初始位置
            #agent.state.p_pos = np.random.uniform(-config.range, +config.range, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.n_serv = 0
            agent.state.a_id = int(0)


        landmark_pos_lst = []
        for i, landmark in enumerate(world.landmarks):
           # landmark.state.p_pos = np.random.uniform(-config.range, +config.range, world.dim_p)

           new_rand = np.random.uniform(-config.range, config.range, world.dim_p)
           if len(landmark_pos_lst) == 0:
               landmark.state.p_pos = new_rand
               landmark_pos_lst.append(new_rand)
           else:
               dists = [np.sqrt(np.sum(np.square(new_rand - a))) for a in landmark_pos_lst]
               min_dist = min(dists)
               while min_dist < 10:
                   new_rand = np.random.uniform(-config.range, config.range, world.dim_p)
                   dists = [np.sqrt(np.sum(np.square(new_rand - a))) for a in landmark_pos_lst]
                   min_dist = min(dists)
               landmark.state.p_pos = new_rand
               landmark_pos_lst.append(new_rand)
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
        #用户接入最近的UAV
        dis_mat_idx_descend = dis_mat.argsort(1) # for given UE the order of UAVs.
        for idx_l, l in enumerate(world.landmarks):
            if dis_mat[idx_l, dis_mat_idx_descend[idx_l, 0]] <=config.d_max:
                acc_mat[idx_l, dis_mat_idx_descend[idx_l, 0]] = 1.0
        # 最近的用户，接入UAV
        # dis_mat_idx_descend = dis_mat.argsort(0) # for given UAV the order of UE.
        # for idx_a, a in enumerate(world.agents):
        #     if dis_mat[dis_mat_idx_descend[0, idx_a],idx_a] <= config.d_max:
        #         acc_mat[ dis_mat_idx_descend[0, idx_a], idx_a ] = 1.0




        # update the serv_n of UAVs
        #if world.reward_calc_n == 0:
        n_access_UAV = acc_mat.sum(1)
        for idx_l, l in enumerate(world.landmarks):
            if n_access_UAV[idx_l] > 0:
                UAV_id = np.where(acc_mat[idx_l, :] == 1.0)[0][0]
                world.agents[UAV_id].state.n_serv += 1
                l.state.n_serv += 1
        # world.reward_calc_n += 1
        # if world.reward_calc_n == config.UAV_n:
        #     world.reward_calc_n = 0

        # sum_data_rate = 0
        # # compute the data rate of different user
        #
        # # parameters
        # noise = config.noise
        # P_ref = config.P_ref
        # alpha = config.alpha
        # height = config.height
        # p_los_B = config.p_los_B
        # p_los_C = config.p_los_C
        # n_access_UAV = acc_mat.sum(1)
        # n_access_UE= acc_mat.sum(0) #number of users access to a given UAV
        # # sum_data_rate_lst = []
        # for idx_l, l in enumerate(world.landmarks):
        #     if n_access_UAV[idx_l] > 0:
        #         # find the serving UAV index
        #         idx_a = np.where(acc_mat[idx_l, :] == 1.0)[0]
        #         if len(idx_a)>1:
        #             idx_a = idx_a[np.argmin(dis_mat[idx_l, idx_a])]
        #         dist = np.sqrt(np.square(dis_mat[idx_l, idx_a]) + np.square(height))
        #         # compute the channel with given distanc
        #         P_los_pro = 1.0 / (  1.0 + p_los_C* np.exp(-p_los_B*( 180/np.pi*np.arctan(height/dist) - p_los_C) )  )
        #         channel_loss = P_los_pro* np.power(10, -3/10) + (1-P_los_pro)* np.power(10, -23/10)
        #         signal = P_ref*np.power(dist,alpha)*channel_loss
        #         # sum_data_rate_lst.append(np.log2(1 + signal / noise))
        #         sum_data_rate += np.log2(1 + signal / noise)/float(n_access_UE[idx_a])
        # # print(sum_data_rate)
        # sum_data_rate = sum_data_rate/config.UAV_n


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
        sum_data_rate = 1
        rew += sum_data_rate*(10*UE_fair+5*UAV_fair) + 0.1*n_serv_UE.sum() #用户公平性、UAV公平性、服务用户数

        world.UE_fair = UE_fair
        world.UAV_fair = UAV_fair
        world.sum_UE = n_serv_UE.sum()

        #print(UE_fair,UAV_fair,sum_data_rate*(10*UE_fair+5*UAV_fair), 0.1*n_serv_UE.sum())
        # considering the collision and out of range
        if agent.collide:
            for a in world.agents:
                for b in world.agents:
                    if a is b: continue
                    if self.is_collision(a, b):
                        rew -= 1/2
        for a in world.agents:
            if self.is_out_range(a):
                rew -= 5
        return rew

    # get the evaluation results for given  policy
    def eval_data(self,world):
        #return (np.sum(world.data_rate_list), np.sum(world.true_data_list), world.trans_power,world.velocity_list)
        return (world.UE_fair, world.UAV_fair, world.sum_UE)

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
        idx_nearby = idx_nearby[:A_neighbor + 1]
        agents_nearby = [world.agents[i] for i in idx_nearby]
        for other in agents_nearby:
            if other is agent: continue
            # comm.append(other.state.c)
            other_pos.append((other.state.p_pos - agent.state.p_pos)/range_limt)
        # phy_feature = np.concatenate([agent.state.p_vel/agent.max_speed] + [agent.state.p_pos/range_limt] +  other_pos + entity_pos).tolist()
        phy_feature = np.concatenate(
            [agent.state.p_vel / agent.max_speed] + [agent.state.p_pos / range_limt] + other_pos).tolist()
        #phy_feature = np.concatenate([agent.state.p_pos/range_limt] +  other_pos).tolist()

        # adding the serv_number as features
        UAV_serv_features = [a.state.n_serv for a in world.agents]
        UAV_serv_features = np.array(UAV_serv_features) / (np.max(UAV_serv_features)+0.000001)
        UAV_serv_features = UAV_serv_features.tolist()

        onehot_dim = [1.0 if l is agent else 0.0 for l in world.agents]


        return np.array(phy_feature + UAV_serv_features + UE_serv_features + onehot_dim)


if __name__ == '__main__':
    from env.multiagent_com.environment import MultiAgentEnv
    import env.multiagent_com.scenarios as scenarios
    import time
    scenario = scenarios.load('uav_com_u_centric' + ".py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation)
    env.reset()
    for i in range(20):
        ation_n =[[0, np.random.randint(0,9,1)] for j in range(config.UAV_n)]
        obs_n, reward_n, done_n, info_n = env.step(ation_n)
        env.render()
        time.sleep(0.5)
        print([a.state.n_serv for a in world.landmarks])
        print(reward_n)
