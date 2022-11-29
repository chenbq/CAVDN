import torch as th
import torch.nn as nn
import torch.nn.functional as F

# Atoc中的channel对应于ACML中的MessageCoordinatorNet
# thought dim 即为output_Q的dimension
class CoordNet(nn.Module):
    # max_agents为最大允许通信的agent数目，thought_dim为thought的维度
    def __init__(self, args):
        thought_dim = args.rnn_hidden_dim
        super(CoordNet, self).__init__()

        self.hidden_dim = thought_dim // 2
        self.FC1 = nn.Linear(thought_dim, self.hidden_dim)
        self.BN1 = nn.LayerNorm(self.hidden_dim)
        # 此处hidden_dim为thought的1/2，使得输出正好为thought的维度
        self.bicnet = nn.GRU(input_size=self.hidden_dim,
                             hidden_size=self.hidden_dim,
                             num_layers=1,
                             batch_first=True,
                             #nonlinearity='relu',
                             bidirectional=True)
        #self.FC2 = nn.Linear(thought_dim, thought_dim)
    '''
    thought_comb : n_agent, batch_size, thought_dim
    '''

    def forward(self, thought_comb):
        thought_comb = self.BN1(F.relu(self.FC1(thought_comb)))
        bi_output, _ = self.bicnet(thought_comb)
        #i_output = self.FC2(bi_output)
        return F.relu(bi_output)

# Definition of Q-output-layer
class Q_out(nn.Module):
    def __init__(self, args, hidden_shape=32):
        super(Q_out, self).__init__()
        self.fc1 = nn.Linear(args.rnn_hidden_dim*2, hidden_shape)
        self.fc2 = nn.Linear(hidden_shape, args.n_actions)

    def forward(self, input_mac):
        hidden = F.elu(self.fc1(input_mac))
        output_mac = self.fc2(hidden)
        return output_mac