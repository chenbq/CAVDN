## Pytorch implementation of "Communication-Assisted Multi-Agent Reinforcement Learning Improves Task-Offloading in UAV-Aided Edge-Computing Networks"

This is the github repo for the work "Communication-Assisted Multi-Agent Reinforcement Learning Improves Task-Offloading in UAV-Aided Edge-Computing Networks".



Detailed settings regarding the neural networks and the training process for the paper are provided as follows.

The FNN in encode module and intention module have three hidden layers with [128, 128, 64] neurons. The RNN has one hidden layer with 64 neurons. The FNN in combine module has two hidden layers having [128, 128] neurons. For all networks, \emph{ReLU} and \emph{linear} activation functions are used for the hidden layers and the output layer, respectively. The number of episodes is $T = 25$, and $\delta = 1$ s. To balancing different optimization objective, we set $\lambda=0.5$ and $\beta=0.5$. The replay buffer size is $10^5$. For network update, we set $K=32$, $\gamma = 0.95$, $\tau = 0.001$. We use the Adam optimizer with an initial learning rate of $0.001$. 