from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args
import numpy as np
import random
import torch
# setting random seed for all random factors
from env.UAV_env import UAVEnv
from types import SimpleNamespace
import shutil
from pathlib import Path
import os
seeds_list = [8, 16, 32, 64]

def save_code(run_dir):
    code_dir = Path(run_dir) / 'codes'
    src_dir = Path('.')
    if not code_dir.exists():
        os.makedirs(code_dir)

        # file list
        file_list = ['main.py', 'runner.py']
        # dir list
        dir_list = ['agent', 'common', 'env', 'network', 'policy']
        for file_i in file_list:
            shutil.copy2(src_dir /file_i, code_dir)  # complete target filename given

        for dir_i in dir_list:
            shutil.copytree(src_dir / dir_i, code_dir / dir_i)

if __name__ == '__main__':

    for i in range(1):
        seed_value = seeds_list[i]
        np.random.seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)

        args = get_common_args()
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
        Eval_run = 'run7'
        # 用来保存plt和pkl
        result_dir = Path(args.result_dir + '/' + args.alg + '/' + args.map)

        # 判断是否存在，以便另建立文件夹
        if not result_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             result_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)

        args.result_dir = args.result_dir + '/' + args.alg + '/' + args.map + '/'+ curr_run
        if args.learn:
            # save code
            save_code(args.result_dir)
        else:
            curr_run = Eval_run

        model_dir = Path(args.model_dir + '/' + args.alg + '/' + args.map)
        # 判断是否存在，以便另建立文件夹
        if not model_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                             model_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        if not args.learn:
            curr_run = Eval_run

        args.model_dir = args.model_dir + '/' + args.alg + '/' + args.map + '/'+ curr_run



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
        #env = PPEnv()
        env = UAVEnv()

        # env = StarCraft2Env(map_name='2s3z',#'3s5z',2s3z #args.map, #3s_vs_4z ##1-11 Next 2s3z
        #                     step_mul=args.step_mul,
        #                     difficulty=args.difficulty,
        #                     game_version=args.game_version,
        #                     obs_all_health=args.obs_all_health,
        #                     obs_own_health=args.obs_own_health,
        #                     seed=args.seed,
        #                     replay_dir=args.replay_dir)
        env_info = env.get_env_info()
        args.n_actions = env_info["n_actions"]
        args.n_agents = env_info["n_agents"]
        args.state_shape = env_info["state_shape"]
        args.obs_shape = env_info["obs_shape"]
        args.episode_limit = env_info["episode_limit"]
        runner = Runner(env, args)
        if args.learn:
            runner.run(i)
        else:
            for i in range(100):
                win_rate, _ = runner.evaluate()
                print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
