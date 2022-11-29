from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import numpy as np
import random
import torch
# setting random seed for all random factors
from env.D2D_env import D2DEnv
from env.UAV_env import UAVEnv
from env.PP_env import PPEnv
from types import SimpleNamespace
import shutil
from pathlib import Path
import os
seeds_list = [8, 16, 32, 64]
Eval_episode = 500
import pylustrator
pylustrator.start()
marker_list = ['.',  # point
                   '+',  # plus
                   'x',  # x
                   'D',  # diamond
                   'v',  # triangle_down
                   '^',  # triangle_up
                   '<',  # triangle_left
                   '>'  # triangle_right
                   ]
line_list = ['-',  # solid line style
             '--',  # dashed line style
             '-.',  # dash-dot line style
             ':',  # dotted line style
             '-',  # solid line style
             '--',  # dashed line style
             '-.',  # dash-dot line style
             ':'  # dotted line style
             ]
color_list = ['b',  # blue
              'g',  # green
              'r',  # red
              'c',  # cyan
              'm',  # magenta
              'y',  # yellow
              'k',  # black
              'w'  # white
              ]


def get_alg_runner(alg, run_time):
    seed_value = seeds_list[0]
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)

    args = get_common_args()
    args.alg = alg
    args.learn = False
    args.load_model = True

    if args.alg.find('coma') > -1:
        args = get_coma_args(args)
    elif args.alg.find('central_v') > -1:
        args = get_centralv_args(args)
    elif args.alg.find('reinforce') > -1:
        args = get_reinforce_args(args)
    else:
        args = get_mixer_args(args)

    if args.alg.find('commnet') > -1:
        args = get_commnet_args(args)
    if args.alg.find('g2anet') > -1:
        args = get_g2anet_args(args)
    # create folder
    Eval_run = run_time
    args.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/' + Eval_run

    config = {
        'range': 500,
        'n_cue': 10,
        'n_dpair': 10,
        'n_rb': 10,
        'pb': np.power(10.0, 46 / 10),
        'pd': np.power(10.0, 23 / 10),
        'noise': np.power(10.0, -174 / 10) * 180,  # -dBm/Hz
        # 'ref_loss': np.power(10.0, -36.5 / 10), #36.5 #-128.1
        'ref_loss': np.power(10.0, -36.5 / 10),  # 36.5 #-128.1
        # 'd_ref_loss': np.power(10.0, -37.6 / 10), #37.6
        'd_ref_loss': np.power(10.0, 0 / 10),  # 37.6
        'alpha': -3.76,  # 37.6
        'd_alpha': -3.68,  # 36.8
        'band': 180,  # Khz
        'n_state': 4,  # position 2 rb block 1 inteference 1
        'n_action': 2,  # power control 1, resource block 1 # resource block 可以用0-1近似
        'sinr_t': np.power(10.0, 0),  # dB
        'neg_r': -1
    }
    config = SimpleNamespace(**config)
    # env = D2DEnv(config)
    # env = PPEnv()
    env = UAVEnv()

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    return env, Runner(env, args), args

def action_from_pos(pos_l, pos_a):
    action = pos_l - pos_a
    action_norm = np.linalg.norm(action)
    max_speed = 20
    return np.array(action/action_norm), min(action_norm/max_speed,1)

direction_list = np.array([[-1.0,.0], [1.0, 0 ], [.0,-1], [.0,1.0],
                          [-1.0*np.sqrt(0.5), -1.0*np.sqrt(0.5)],
                          [+1.0*np.sqrt(0.5), -1.0*np.sqrt(0.5)],
                          [-1.0*np.sqrt(0.5), +1.0*np.sqrt(0.5)],
                          [+1.0*np.sqrt(0.5), +1.0*np.sqrt(0.5)]]
                          )
speed_list = np.array([0.0, 0.5, 1.0])

def greedy_evaluate(env, args):

    UE_fair = np.zeros((Eval_episode, args.episode_limit))
    UAV_fair = np.zeros((Eval_episode, args.episode_limit))
    sum_UE = np.zeros((Eval_episode, args.episode_limit))
    for ep_i in range(Eval_episode):
        env.reset()
        # obtain the sequence for each agent and landmark position
        target_position_list = np.array([l.state.p_pos for l in env.env.world.landmarks])

        dis_mat = np.array([[np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in env.env.world.agents] for l in
                            env.env.world.landmarks])
        dis_mat_idx_descend = dis_mat.argsort(0) # target 的list
        current_idx = 0
        for et_i in range(args.episode_limit):

            current_idx += 1
            current_idx = current_idx % len(env.env.world.landmarks)
            agent_position_list = np.array([a.state.p_pos for a in env.env.world.agents])

            actions = []
            for a_i in range(len(agent_position_list)):

                #direction, speed = action_from_pos( target_position_list[dis_mat_idx_descend[current_idx][a_i]], agent_position_list[a_i] )

                direction, speed = action_from_pos( target_position_list[current_idx], agent_position_list[a_i] )


                max_direction_id =  np.argmax(np.matmul(direction_list,direction.reshape(2,1)).squeeze())
                speed_id = 2 #np.argmin(np.abs(speed_list-speed))

                actions.append(speed_id*8+max_direction_id)

            reward, terminated, info = env.step(actions)

            UE_fair[ep_i, et_i] = np.mean(info[0])
            UAV_fair[ep_i, et_i] = np.mean(info[1])
            sum_UE[ep_i, et_i] = np.mean(info[2])
    env.close()
    return np.mean(UE_fair, axis=0), np.mean(UAV_fair, axis=0), np.mean(sum_UE, axis=0)
    #return np.mean(AoI_vs_time, axis=0), 1.0*np.ones(config.episode_length), 20.0*np.ones(config.episode_length)



def random_evaluate(env, args):
    UE_fair = np.zeros((Eval_episode, args.episode_limit))
    UAV_fair = np.zeros((Eval_episode, args.episode_limit))
    sum_UE = np.zeros((Eval_episode, args.episode_limit))
    for ep_i in range(Eval_episode):
        env.reset()

        for et_i in range(args.episode_limit):
            # agent_position_list = np.array([a.state.p_pos for a in env.envs[0].world.agents])
            #actions = [np.random.uniform(-1, 1, (len(env.envs[0].world.agents), 2))]
            ation_n = [np.random.randint(0, args.n_actions, 1) for j in range(args.n_agents)]

            reward, terminated, info = env.step(ation_n)

            UE_fair[ep_i, et_i] = np.mean(info[0])
            UAV_fair[ep_i, et_i] = np.mean(info[1])
            sum_UE[ep_i, et_i] = np.mean(info[2])
    env.close()
    return np.mean(UE_fair, axis=0), np.mean(UAV_fair, axis=0), np.mean(sum_UE, axis=0)


def alg_eval(runner, env):
    UE_fair = np.zeros((Eval_episode, args.episode_limit))
    UAV_fair = np.zeros((Eval_episode, args.episode_limit))
    sum_UE = np.zeros((Eval_episode, args.episode_limit))
    for ep_i in range(Eval_episode):
        episode_reward, episode_info = runner.evaluateStep()
        UE_fair[ep_i, :] = np.array(episode_info)[:,0]
        UAV_fair[ep_i, :] = np.array(episode_info)[:,1]
        sum_UE[ep_i, :] = np.array(episode_info)[:,2]


    env.close()
    return np.mean(UE_fair, axis=0), np.mean(UAV_fair, axis=0), np.mean(sum_UE, axis=0)


if __name__ == '__main__':

    label_list = ['CAVDN', 'VDN', 'Double DQN', 'Greedy', 'Random'] #
    config_list = ['FC+QMIX', 'vdn', 'iql', 'iql', 'iql']
    model_idx = ['run2', 'run1', 'run3', 'run3','run3']
    y_label_list = ['UE Fairness', 'UAV Fairness', 'Offloaded UE No.']

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    ax_list = [ax1, ax2, ax3]
    # t = np.linspace(0, 49, 50) + 1
    t = np.linspace(0, 48, 25) + 1

    for i_curve, txt_file in enumerate(config_list):
        env, runner, args = get_alg_runner(txt_file,model_idx[i_curve])

        if label_list[i_curve] == 'Random':
            UE_fair, UAV_fair, sum_UE = random_evaluate(env, args)
        elif label_list[i_curve] == 'Greedy':
            UE_fair, UAV_fair, sum_UE = greedy_evaluate(env, args)
        else:
            UE_fair, UAV_fair, sum_UE = alg_eval(runner, env)

        ax1.plot(t, UE_fair[t.astype(int)], linestyle=line_list[i_curve], color=color_list[i_curve],
                 marker=marker_list[i_curve], label=label_list[i_curve])
        ax2.plot(t, UAV_fair[t.astype(int)], linestyle=line_list[i_curve], color=color_list[i_curve],
                 marker=marker_list[i_curve], label=label_list[i_curve])
        ax3.plot(t, sum_UE[t.astype(int)], linestyle=line_list[i_curve], color=color_list[i_curve],
                 marker=marker_list[i_curve], label=label_list[i_curve])
        for i_ax,ax in enumerate(ax_list):
            ax.legend(loc="best")
            ax.grid()
            ax.set_facecolor("white")
            ax.set_ylabel(y_label_list[i_ax], fontsize='large')
            ax.set_xlabel('Time steps', fontsize='large')

    #% start: automatic generated code from pylustrator
    plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
    import matplotlib as mpl
    plt.figure(1).axes[0].set_xlim(0.0, 51.4)
    plt.figure(1).axes[0].set_ylim(0.0, 1.0)
    plt.figure(1).axes[0].legend(borderpad=0.30000000000000004, labelspacing=0.19999999999999998, handletextpad=0.7999999999999999, fontsize=10.0, title_fontsize=10.0)
    plt.figure(1).axes[0].set_position([0.098438, 0.110000, 0.227941, 0.770000])
    plt.figure(1).axes[0].yaxis.labelpad = 1.040000
    plt.figure(1).axes[0].get_legend()._set_loc((0.026994, 0.738485))
    plt.figure(1).axes[0].get_legend()._set_loc((0.026994, 0.752014))
    plt.figure(1).axes[1].set_xlim(0.0, 51.4)
    plt.figure(1).axes[1].set_ylim(0.0, 1.0)
    plt.figure(1).axes[1].legend(borderpad=0.3, labelspacing=0.3, fontsize=10.0, title_fontsize=10.0)
    plt.figure(1).axes[1].set_position([0.417279, 0.110000, 0.227941, 0.770000])
    plt.figure(1).axes[1].yaxis.labelpad = -0.240000
    plt.figure(1).axes[1].get_legend()._set_loc((0.047559, 0.021495))
    plt.figure(1).axes[2].set_xlim(0.0, 51.4)
    plt.figure(1).axes[2].set_ylim(0.0, 100.0)
    plt.figure(1).axes[2].legend(borderpad=0.3, labelspacing=0.3, fontsize=10.0, title_fontsize=10.0)
    plt.figure(1).axes[2].set_position([0.729872, 0.110000, 0.227941, 0.770000])
    plt.figure(1).axes[2].yaxis.labelpad = -7.360000
    plt.figure(1).axes[2].get_legend()._set_loc((0.027039, 0.727182))
    #% end: automatic generated code from pylustrator
    plt.show()
    plt.savefig('learning_curve.pdf', dpi=400)

