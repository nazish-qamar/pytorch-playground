import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

learning_rate = 0.01
model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For Lambda LR
# lambda1 = lambda epoch: epoch/10
# scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

# For multiplicative LR
lambda1 = lambda epoch: 0.95
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda1)

print(optimizer.state_dict())
for epoch in range(5):
    # loss.backward()
    optimizer.step()

    # validate(...)

    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])