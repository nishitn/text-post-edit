import torch.nn as nn
net=nn.Sequential(
    nn.Linear(20,10),
    nn.Tanh(),
    nn.Linear(10,10),
    nn.Tanh(),
    nn.Linear(10,1)
    )
h1=net.modules[1]()
