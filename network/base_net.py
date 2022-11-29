import torch.nn as nn
import torch as th
import torch.nn.functional as f

# N_neigbor = 2
# dim_obs = 24 ##??
# dim_nearby = N_neigbor + 1  # 包含自己的agent的数量
# # 附近为3个predator
# dim_landmarks = 3 # 附近的landmark 或者prey的数量
# dim_joint_obs = 4 * 2  # 包含自己的agent的数量
# dim_entity_feature = 4
# dim_total = dim_nearby+dim_landmarks

# parameter 3s5z
# N_neigbor = 7
# #dim_obs = 14 ##??
# # 附近为3个predator
# dim_landmarks = 8 # 附近的landmark 或者prey的数量
# #dim_joint_obs = 2 * 2  # 包含自己的agent的数量
# dim_entity_feature = 8

# parameter 3s_vs_4z
# N_neigbor = 2
# #dim_obs = 14 ##??
# # 附近为3个predator
# dim_landmarks = 4 # 附近的landmark 或者prey的数量
# #dim_joint_obs = 2 * 2  # 包含自己的agent的数量
# dim_entity_feature = 6

# parameter 5m_vs_6m
N_neigbor = 4
#dim_obs = 14 ##??
# 附近为3个predator
dim_landmarks = 5 # 附近的landmark 或者prey的数量
dim_entity_feature = 8



dim_nearby = N_neigbor + 1  # 包含自己的agent的数量
#dim_entity_feature = 6 # without health and shield but with type 2 bit
dim_total = dim_nearby+dim_landmarks # 总的agent： Enemy+Alley+self
dim_soft = 2*dim_nearby # aggregation and pooling的时候，按照每个enemy对alley和self的attention来反向计算。

in_features = dim_entity_feature #64 # 6 to 64 dimension

# def data_preprocess(obs):
#     # first step: data processing
#     obs = obs[:,2:]
#     obs_expand = obs.reshape(-1, 1, 1, dim_obs).repeat(1, 1, dim_total, dim_total, 1)
#     obs_index = th.zeros_like(obs_expand, dtype=th.long)
#
#     for i in range(dim_total):
#         for j in range(dim_total):
#             if i == j:
#                 obs_index[:, :, i, j, 0:dim_joint_obs] = 1.0
#             else:
#                 obs_index[:, :, i, j, i * dim_entity_feature:(i + 1) * dim_entity_feature] = 1.0
#                 obs_index[:, :, i, j, j * dim_entity_feature:(j + 1) * dim_entity_feature] = 1.0
#
#     if obs_expand.device != 'cuda':
#         obs_index = obs_index.to('cuda')
#     # (n_agent, n_batch, dim_nearby,dim_landmarks, dim_joint_obs)
#     obs_in = obs_expand[obs_index == 1].reshape( -1, dim_total, dim_total, dim_joint_obs)
#     for i in range(dim_total):
#         for j in range(dim_total):
#             if i == j:
#                 obs_in[:, i, j, :] = 0.0
#     return obs_in

def data_preprocess(obs):
    # first step: data processing
    # last action and agents number 8+14?
    #last_ac_vs_agent_id = 8 + 14
    last_ac_vs_agent_id = dim_nearby
    dim_ori = obs.shape[-1]-last_ac_vs_agent_id
    obs = obs[:,:dim_ori]
    # inital number of obs for agent is 4.
    obs = th.cat((obs[:,4:],obs[:,:4]),dim=-1)

    obs_in = obs.reshape(-1, dim_total, dim_entity_feature)

    return obs_in

def data_postprocess(obs_in, obs_out):
    # 要先进行处理再乘以attention，否则输入有问题
    #thought_pre_i = obs_in[:,dim_nearby:,:]* obs_out.unsqueeze(-1).repeat(1, 1, dim_entity_feature) 12-29-change the input
    thought_pre_i = obs_in[:,dim_nearby:,:]* obs_out.unsqueeze(-1).repeat(1, 1, in_features)
    # thought_pre_i = torch.sum(thought_pre_i, 2)
    thought_sum = thought_pre_i.sum(dim=-2)
    #obs_adj = obs_in[:, :, 0, 0:dim_entity_feature].reshape(-1, dim_entity_feature * dim_nearby)
    # return thought_pre_i.reshape( -1, dim_landmarks*dim_joint_obs), obs_out
    return thought_sum



class ATOM_RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args, in_features= 64, out_features= 64,alpha=0.01):
        super(ATOM_RNN, self).__init__()

        ## original RNN part
        self.args = args
        #input_shape += dim_entity_feature
        input_shape += dim_landmarks # 0106 改为直接对softmax的结果进行变换
        # input_shape += in_features # 12-29 改为增加相同的纬度的输入，进行变换
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        ## ATOM appendix part
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.input_layer = nn.Linear(dim_entity_feature, in_features)

        # if starcraft
        self.input_layer_0 = nn.Linear(dim_entity_feature, in_features)
        self.input_layer_1 = nn.Linear(dim_entity_feature, in_features)
        self.input_layer_2 = nn.Linear(dim_entity_feature, in_features)

        self.W = nn.Parameter(th.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(th.empty(size=(2*out_features, 1)))

        self.output_layer1 = nn.Linear(dim_soft, 32)
        self.output_layer2 = nn.Linear(32, 16)
        self.output_layer3 = nn.Linear(16, 1)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.inital_all_weights()

    def inital_all_weights(self):
        nn.init.xavier_uniform_(self.W.data, gain=1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        ##nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.rnn.weight, gain=1.414)
        #nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        #nn.init.calculate_gain('relu')
        '''
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        nn.init.xavier_uniform_(self.input_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.input_layer_0.weight, gain=1.414)
        nn.init.xavier_uniform_(self.input_layer_1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.input_layer_2.weight, gain=1.414)


        

        nn.init.xavier_uniform_(self.output_layer1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.output_layer2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.output_layer3.weight, gain=1.414)'''

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = th.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = th.matmul(Wh, self.a[self.out_features:, :])
        Wh2 = Wh2.permute(0,2,1)
        # broadcast add
        e = Wh1 + Wh2
        return self.leakyrelu(e)

    def forward(self, obs, hidden_state):

        obs_in  = data_preprocess(obs)
        h_0 = self.leakyrelu(self.input_layer_0(obs_in))
        h_1 = self.leakyrelu(self.input_layer_1(obs_in))
        h_2 = self.leakyrelu(self.input_layer_2(obs_in))
        h_mix = th.cat((h_0[:, :dim_landmarks, :], h_1[:, dim_landmarks:dim_total-1, :], h_2[:, dim_total-1, :].unsqueeze(dim=1)), dim=-2)
        Wh = th.matmul(h_mix, self.W)
        #e = self.leakyrelu(th.matmul(Wh, self.a).squeeze(-1))

        # zero_vec = -9e15 * th.ones_like(e)
        # attention = th.where(adj > 0, e, zero_vec)
        # landmark to nearby
        e = self._prepare_attentional_mechanism_input(Wh)

        #attention_0 = f.softmax(e[:,:dim_nearby,dim_nearby:], dim=-1)
        #attention_0 = attention_0.permute(0, 2, 1)
        # revised 01-08 to keep consistent with h_mix above
        attention_0 = f.softmax(e[:, :dim_landmarks, dim_landmarks:], dim=-1)

        attention_1 = f.softmax(e[:, dim_landmarks:, :dim_landmarks], dim=-1)
        attention_1 = attention_1.permute(0, 2, 1)


        #attention = f.softmax(e, dim=-1)
        attention = th.cat((attention_0, attention_1), -1)

        h_prime = self.leakyrelu(self.output_layer1(attention))
        h_prime = self.leakyrelu(self.output_layer2(h_prime))
        h_prime = self.leakyrelu(self.output_layer3(h_prime)).squeeze(-1)
        obs_out = f.softmax(h_prime, dim=-1)
        #atom_plug = data_postprocess(obs_in, obs_out)
        #atom_plug = data_postprocess(h_mix, obs_out) #change to weights the same dimension
        # 01-04 直接将得到的分布，加入到网络之中
        atom_plug = obs_out

        obs_comb = th.cat((obs, atom_plug),dim=-1)

        ## orignal RNN part

        x = f.relu(self.fc1(obs_comb))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        return q, h

class RNN(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        #self.inital_all_weights()

    def inital_all_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.rnn.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight)

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

# Definition of Message Encoder
class Env_blender(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_shape):
        super(Env_blender, self).__init__()
        self.fc1 = nn.Linear(input_shape, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, output_shape)

    def forward(self, input_mac):
        hidden = f.elu(self.fc1(input_mac))
        output_mac = self.fc2(hidden)
        return output_mac


class RNN_FC(nn.Module):
    # Because all the agents share the same network, input_shape=obs_shape+n_actions+n_agents
    def __init__(self, input_shape, args,hidden=32):
        super(RNN_FC, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.ec1 = nn.Linear(args.rnn_hidden_dim, hidden)
        self.ec2 = nn.Linear(hidden, args.n_actions)
        #self.ec = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        #self.inital_all_weights()
        #self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.leakyrelu = nn.LeakyReLU()

    def inital_all_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.xavier_uniform_(self.rnn.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight)

        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


    def forward(self, obs, hidden_state):
        x = f.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        # message_encoder
        # use elu in original
        x = self.leakyrelu(self.ec1(h))
        #x = f.elu(self.ec1(h))
        #01-16- encoder to two layer
        x = self.ec2(x)

        #x = self.ec(h)
        return q, h, x

# Critic of Central-V
class Critic(nn.Module):
    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
