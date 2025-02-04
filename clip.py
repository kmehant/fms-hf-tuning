# Third Party
import torch
import torch.nn as nn
import torch.optim as optim


# Define a deeper model
class MyDeepModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyDeepModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),  # First layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),  # Second layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),  # Third layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(
                128, output_dim
            ),  # Output layer (no activation, handled by loss function)
        )

    def forward(self, x):
        return self.network(x)


# Example input-output dimensions
input_dim = 100
output_dim = 10
model = MyDeepModel(input_dim, output_dim)

# Loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
max_norm = 1.0

# Dummy dataloader (replace with real data)
dataloader = [
    {"input": torch.randn(32, input_dim), "target": torch.randint(0, output_dim, (32,))}
    for _ in range(10)
]

# Training loop
for batch in dataloader:
    optimizer.zero_grad()

    # Convert lists to tensors if needed
    inputs = torch.tensor(batch["input"], dtype=torch.float32)
    targets = torch.tensor(batch["target"], dtype=torch.long)

    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    for param in model.parameters():
        print(param)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
