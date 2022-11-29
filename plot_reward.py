#import seaborn as sns; sns.set(font_scale=1.2)
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import pylustrator
import pickle

np.random.seed(1)
pylustrator.start()
rc('pdf', fonttype=42) #keep the font type to be adapted to submit requirements.

def smooth_data(window_size, data):
    #np_array input
    y = np.ones(window_size)
    for idx in range(data.shape[0]):
        x = data[idx,:]
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        data[idx,:] = smoothed_x
    return data


def get_array(name, mode='np'):
    epoches = 400
    dtpc_r_all = np.zeros(epoches)
    if mode == 'np':
        x = np.load(name)
    else:
        #file_pk = open(r'F:\TVT_Offloading_DQN\models\uav_com_u_centric\double_dqn\run1\epsode_rewards.pkl', 'rb')
        # file_pk = open(name, 'rb')
        # x = pickle.load(file_pk)
        # #x = x[:20000]
        # x = np.array(x)
        # zz = np.linspace(0, 19994, 400) + 1
        # zz = zz.astype(int)
        # x = x[zz]

        from numpy import genfromtxt
        train = genfromtxt(name, delimiter=',')
        x = train[1:, 2]*50.0
        x = x[:400]


    dtpc_r_all[:len(x)] = x
    if len(x)<epoches:
        dtpc_r_all[len(x):epoches]= x[2*len(x)-epoches:len(x)]

    dtpc_r_all = np.array(dtpc_r_all).reshape(100, 4).T
    dtpc_r_all = smooth_data(2, dtpc_r_all)
    return dtpc_r_all




def get_data():
    '''获取数据
    '''
    label_list = ['FC+QMIX', 'VDN', 'IQL']
    config_list = ['FC+QMIX', 'vdn', 'iql']
    model_idx = ['run2', 'run1', 'run3']

    #label_list = ['ATOC', 'ACML', 'MADDPG', 'DDPG','Greedy','Random']
    #config_list = ['atoc','acml','maddpg','ddpg','ddpg','ddpg']
    #model_idx = ['run8','run5','run10','run11']
    top_path = r'F:\TVT_Offloading\result'
    # basecond = get_array(top_path+r'\FC+QMIX\UAV_offload\run2\episode_rewards_0.npy')
    cond0 = get_array(top_path+r'\FC\UAV_offload\run8\episode_rewards_0.npy')
    cond1 = get_array(top_path+r'\vdn\UAV_offload\run1\episode_rewards_0.npy')
    cond2 = get_array(top_path+r'\iql\UAV_offload\run3\episode_rewards_0.npy')
    cond3 = get_array('F:\TVT_Offloading\scalars_double_dqn.csv', 'pickle')
    cond4 = get_array('F:\TVT_Offloading\scalars_dueling_dqn.csv', 'pickle')
    cond5 = get_array('F:\TVT_Offloading\scalars_dqn.csv', 'pickle')
    # cond4 = get_array(top_path+r'\double_dqn\run1\logs\all_reward.csv')
    # cond5 = get_array(top_path+r'\dueling_dqn\run1\logs\all_reward.csv')

    return   cond0, cond1, cond3, cond4

data = get_data()
label = ['CAVDN', 'VDN', 'DDQN', 'Dueling_DQN' ]
#['CATCP-part', 'CATCP-all', 'MADDPG', 'DDPG']
df=[]
for i in range(len(data)):
    tmp = pd.DataFrame(data[i]).melt(var_name='Episode', value_name='Reward')
    tmp.Episode *= 200
    df.append(tmp)
    df[i]['Algos']= label[i]


fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
t = np.linspace(0, 99, 100)
linestyle_tuple = [
     ('line',        (0, (100, 1))),
     ('densely dotted',        (0, (1, 1))),
     ('dashed',                (0, (5, 5))),
     ('dashdotted',            (0, (3, 2, 2, 2))),
     ('dashdotted',            (0, (2, 2, 10, 2))),
     ('dashdotdotted',         (0, (3, 2, 2, 2, 2, 2)))]

color_list  =['b', #blue
                'g', #green
                'r', #red
                'c', #cyan
                'm', #magenta
                'y', #yellow
                'k', #black
                'w' #white
                ]
for i,label_txt in enumerate(label):
    line, = ax1.plot(t, data[i][0],  label=label_txt, color = color_list[i])
    line.set_dashes(linestyle_tuple[i][1][1])

ax1.legend(loc="best")
ax1.grid()
ax1.set_facecolor("white")
ax1.set_ylabel('Reward',fontsize='large')
ax1.set_xlabel('Episode*100', fontsize='large')
fig.patch.set_facecolor('white')

#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_xlim(-0.0, 100.0)
plt.figure(1).axes[0].set_ylim(-0.0, 700.0)
plt.figure(1).axes[0].get_legend()._set_loc((0.663195, 0.069415))
plt.figure(1).axes[0].get_xaxis().get_label().set_text("Episode*200")
#% end: automatic generated code from pylustrator
plt.show()
plt.savefig('reward_curve.pdf',dpi = 400)