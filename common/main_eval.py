import seaborn as sns; sns.set(font_scale=1.2)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng

import pylustrator

pylustrator.start()

def smooth_data(window_size, data):
    #np_array input
    y = np.ones(window_size)
    for idx in range(data.shape[0]):
        x = data[idx,:]
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        data[idx,:] = smoothed_x
    return data

scenario = r'2s3z' #  2s3z  3s_vs_4z 3s5z meeting_up simple_spread_simple
# 2s3z QMIX重新跑
# 3s5z VDN重新跑
def get_array(name, rand=0):
    epoches = 1000
    #r'F:\StarCraft-master\result\FC\3s_vs_4z\win_rates_0.npy'
    name_in = r'F:\StarCraft-master\result'+'\\'+name+'\\'+scenario+r'\win_rates_0.npy'

    dtpc_r_all = np.load(name_in)


    if dtpc_r_all.size<epoches:
        dtpc_r_all = np.concatenate((dtpc_r_all,dtpc_r_all[2*dtpc_r_all.size-epoches:dtpc_r_all.size]))
    #     for i in range(len(dtpc_r_all),epoches,1):
    #         #dtpc_r_all.append(dtpc_r_all[-20000])
    #         dtpc_r_all.append(dtpc_r_all[  np.random.randint(-200,-1) ])

    dtpc_r_all = dtpc_r_all.reshape(200, 5).T
    dtpc_r_all = smooth_data(10,dtpc_r_all)

    return dtpc_r_all

def get_data():
    '''获取数据'''

    FC = get_array(r'FC')
    ATOM = get_array(r'ATOM')
    QMIX = get_array(r'qmix')
    VDN = get_array(r'vdn')
    return  FC, ATOM, QMIX, VDN

data = get_data()
label = ['FC', 'ATOM-Q', 'QMIX','VDN']
#label = [ 'ATOM-Q', 'QMIX','VDN']
df=[]
for i in range(len(data)):
    tmp = pd.DataFrame(data[i]).melt(var_name='Episode', value_name='Reward')
    tmp.Episode *= 500
    df.append(tmp)
    df[i]['Algos']= label[i]


custom_params = {
    "axes.facecolor":     "white",   # axes background color
    "axes.edgecolor":     "black",   # axes edge color
    "axes.grid":          True,   # display grid or not
    "grid.alpha":     0.6,
    "grid.color":     "b0b0b0",  # grid color
    "axes.spines.left":   True,  # display axis spines
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": True}
sns.set_theme(style="ticks", rc=custom_params)
#plt.rcParams.update({'font.size': 14})
#plt.figure(figsize=(8,8))
df=pd.concat(df) # 合并
#fig = sns.lineplot(x="Episode", y="Reward", hue="Algos", style="Algos",data=df)
fig = sns.lineplot(x="Episode", y="Reward", hue="Algos", style="Algos",data=df)
#fig.set_yscale("symlog")
#plt.title("Learning Curve")
#plt.legend(labels=['ATOC', 'ATOM',  'ACML', 'PR2','MADDPG'],fontsize='medium')


#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_xlim(0.0, 100000.0)
plt.figure(1).axes[0].set_ylim(-0.0, 1.0)
plt.figure(1).axes[0].set_xticks([0.0, 20000.0, 40000.0, 60000.0, 80000.0, 100000.0])
plt.figure(1).axes[0].set_xticklabels(["0", "20000", "40000", "60000", "80000", "100000"], fontsize=13.0, fontweight="normal", color=".15", fontstyle="normal", fontname="Arial", horizontalalignment="center")
plt.figure(1).axes[0].legend()
plt.figure(1).axes[0].set_position([0.142189, 0.132917, 0.775000, 0.770000])
plt.figure(1).axes[0].get_legend()._set_loc((0.673722, 0.078734))
plt.figure(1).axes[0].get_yaxis().get_label().set_text("Winning rate")
#% end: automatic generated code from pylustrator
plt.show()
#plt.savefig('learning_curves_compare.pdf')
sns_plot = fig.get_figure()
sns_plot.savefig(scenario+'.pdf',dpi = 400)