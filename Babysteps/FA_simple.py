import sys
import torch
from torch import nn
import matplotlib.pyplot as plt

# Step1: Beginnining by checking the MPS availability weil des fuckass macbook nervt
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using:", device)

# Step2: Training data
x_data = torch.linspace(2, 20, 400, device=device).unsqueeze(1)
y_data = torch.sin(3*x_data) * torch.exp(-0.4 * x_data)

# Step3: Build the neural network (oh god)
model = nn.Sequential( #simple network with a couple hidden layers (32 neurons each) and tanh function to approximate complexer funcitons
    nn.Linear(1, 32),
    nn.ELU(),
    nn.Linear(32, 32),    # approx gets better with more layers, but not wit more neurons per layer
    nn.ELU(),            # accuracy scales weirdly with more layers, maybe better activation functions? nahh tried some, ELU works wuite good
    nn.Linear(32, 32),
    nn.ELU(),         
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 32),
    nn.ELU(),
    nn.Linear(32, 1),
    ).to(device)

# Step4: Loss function and optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01) ## vielleicht da bissi rumspielen

# Step5: Training loop
for epoch in range(1000):
    prediction = model(x_data)
    loss = loss_func(prediction, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Step6: Prediction and plotting
with torch.no_grad():
    prediction1 = model(x_data)
    
plt.plot(x_data.cpu(), y_data.cpu(), color='skyblue', label="Actual function")
plt.plot(x_data.cpu(), prediction1.cpu(), linestyle="-.", color='red', label="Neural network")
plt.legend()
plt.grid()
plt.title("Function Approximation with Neural Network")
plt.show()
