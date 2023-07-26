import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Assuming you have the data in the following format:
# X: a 2D array containing input features [outdoor_temp, horizontal_infrared, time]
# y: a 1D array containing the corresponding indoor temperature

def append_scalars_to_array(arr_list, scalar_list):
    return [np.concatenate((arr, [scalar])) for arr, scalar in zip(arr_list, scalar_list)]

def delete_index_to_array(arr_list, index):
    mask = np.ones(arr_list[0].shape, dtype=bool)
    mask[index] = False
    result = [arr[mask] for arr in arr_list]
    return result

OUTDOOR_TEMP = 0
INDOOR_TEMP = 1
DIFFUSE_SOLAR_LDF = 2
DIFFUSE_SOLAR_SDR = 3
SITE_DIRECT_SOLAR = 4
SITE_HORZ_INFRARED = 5
ELEC_COOLING = 6
HOUR = 7
DAY_OF_WEEK = 8
DAY = 9
MONTH = 10
COST_RATE = 11
COST = 12


pf = open('training_data.pt', 'rb')
buf = pickle.load(pf)
pf.close()

X_train = []
y_train = []

# Convert buf.buffer into a NumPy array
buf_array = np.array(buf.buffer,dtype=object)
# Extract the columns using slicing
s = buf_array[:, 0]
a = buf_array[:, 1]
v = buf_array[:, 2]
y = np.asfarray(v)
print('y:', y.dtype)
#sys.exit(1)

X = append_scalars_to_array(s, a)
# have to delete from outermost index
X = delete_index_to_array(s, COST)
X = delete_index_to_array(s, ELEC_COOLING)
X = np.array(X)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the regression model using PyTorch
class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionModel, self).__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.output_layer(x)
        return x
# Create the model instance
input_size = X_train.shape[1]
model = RegressionModel(input_size, 75)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), './model/regression_model.pth')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    mse = criterion(test_outputs.squeeze(), y_test)
    print("Mean Squared Error:", mse.item())

# Save the trained model
torch.save(model.state_dict(), './model/regression_model.pth')
