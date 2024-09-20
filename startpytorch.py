import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

dados = {
    "Entrada": [10.00, 20.00, 30.00, 40.00, 50.00],
    "Saida": [11.00, 21.00, 31.00, 41.00, 51.00],
}

dados_df = pd.DataFrame(dados)
#print(dados_df)

numerical_features = ["Entrada"]

#create tensor of targets
y = torch.tensor(dados_df["Saida"].values, dtype = torch.float).view(-1, 1)

#Select the numeric columns that the neural network will use as input features to predict the target.
#create tensor of input features
X = torch.tensor(dados_df[numerical_features].values, dtype=torch.float)

#Create a neural netwrok. Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(1, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

#select loss and optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

#training loop
num_epochs = 100
for epoch in range(num_epochs):
    predictions = model(X)
    MSE = loss(predictions, y)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    if(epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {MSE.item()}')

model.eval()
print(model(torch.tensor([58.00], dtype = torch.float)))
