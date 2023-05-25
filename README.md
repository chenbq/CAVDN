## Pytorch implementation of "Communication-Assisted Multi-Agent Reinforcement Learning Improves Task-Offloading in UAV-Aided Edge-Computing Networks"

This is the github repo for the work "Communication-Assisted Multi-Agent Reinforcement Learning Improves Task-Offloading in UAV-Aided Edge-Computing Networks".

# Additional Experiements Results.
Please refer to https://github.com/chenbq/CAVDN/blob/main/Additional%20Experiments.pdf

###  Detailed settings regarding the neural networks and the training process for the paper are provided as follows.

The FNN in encode module and intention module have three hidden layers with [128, 128, 64] neurons. The RNN has one hidden layer with 64 neurons. The FNN in combine module has two hidden layers having [128, 128] neurons. For all networks, $\it ReLU$ and $\it linear$ activation functions are used for the hidden layers and the output layer, respectively. The number of time steps in one epoch is $T = 25$, and $\delta = 1$ s. To balancing different optimization objective, we set $\lambda=0.5$ and $\beta=0.5$. The replay buffer size is $10^5$. For network update, we set $K=32$, $\gamma = 0.95$, $\tau = 0.001$. We use the Adam optimizer with an initial learning rate of $0.001$. 

Key parameters to train the models
| Parameters | Value |
| --- | --- |
| $T$ | 25 |
| $\delta$ | 1 |
| $\lambda$ | 0.5 |
| $\beta$ | 0.5 |
| Buffer Size | $10^5$ |
| $K$ | 32 |
| $\gamma$ | 0.95 |
| $\tau$ | 0.001 |



### Complexity Analysis of CAVDN
The complexity of CAVDN can be reflected by the number of parameters of the agent network. Therefore, we analyze the number of parameters to discuss the complexity of the algorithm.
Let $U_{l}^{enc}$ denote the number of neurons in the $l$th layer of the encoding network with $L^{enc}$ layers, where $1\leq l\leq L^{enc}$. 
Let $U_l^{int}$ denote the number of neurons in the $l$th layer of the intention network with $L^{int}$ layers, where $1\leq l\leq L^{int}$. 
Let $U_l^{comb}$ denote the number of neurons in the $l$th layer of the combining network with $L^{mix}$ layers, where $1\leq l\leq L^{comb}$. 
Then the total number of parameters of the agent network of our CAVDN is given by $\sum_{l=2}^{L^{enc}-1} {(U_{l-1}^{enc}U_l^{enc}+U_l^{enc}U_{l+1}^{enc})}+ \sum_{l=2}^{L^{int}-1} {(U_{l-1}^{int}U_l^{int}+U_l^{int}U_{l+1}^{int})}+ \sum_{l=2}^{L^{comb}-1} {(U_{l-1}^{comb}U_l^{comb}+U_l^{comb}U_{l+1}^{comb})}$.

###  Run the code
1. To train the model. run main.py
The hyper-parameters can be set as in /common/arguments.py
2. To produce the reward figure. run plot_reward_all.py
3. To produce the test figure. run main_eval.py
