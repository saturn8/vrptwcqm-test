import os
import dimod
import numpy as np
import pandas as pd
from dwave.system import LeapHybridCQMSampler
import networkx as nx
import matplotlib.pyplot as plt

# Set API token
api_token = os.getenv('DWAVE_API_TOKEN')
if not api_token:
    raise ValueError("API token not valid. Please set DWAVE_API_TOKEN environment variable.")

# Load data from files
customers_df = pd.read_csv('Input/customers.csv')
vehicles_df = pd.read_csv('Input/vehicles.csv')
distances_df = pd.read_csv('Input/distances.csv', index_col=0)

# Define problem parameters from datasets
num_customers = len(customers_df)
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])
distances = distances_df.to_numpy()

# Variables: x[i][j][k] = 1 if vehicle k travels from i to j
x = {}
for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                x[i, j, k] = dimod.Binary(f'x_{i}_{j}_{k}')

# Create CQM
cqm = dimod.ConstrainedQuadraticModel()

# Objective: Minimize total distance
objective = dimod.BinaryQuadraticModel(dimod.BINARY)
for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                objective.set_linear(f'x_{i}_{j}_{k}', distances[i, j])

cqm.set_objective(objective)

# Constraints: Each customer is visited exactly once
for j in range(1, num_customers + 1):
    cqm.add_constraint(
        sum(x[i, j, k] for k in range(num_vehicles) for i in range(num_customers + 1) if i != j) == 1,
        label=f'visit_{j}'
    )

# Constraints: Time window
arrival_times = {}
for i in range(num_customers + 1):
    if i < len(time_windows):
        arrival_times[i] = dimod.Real(f't_{i}', lower_bound=time_windows[i][0], upper_bound=time_windows[i][1])
    else:
        print(f"Warning: Customer index {i} is out of range for time_windows.")

for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j and i in arrival_times and j in arrival_times:
                # Ensure arrival_times[i] and arrival_times[j] are used
                if i in service_times and j in service_times:
                    cqm.add_constraint(
                        arrival_times[j] - arrival_times[i] >= service_times[i] + distances[i, j] - (1 - x[i, j, k]) * 1000,
                        label=f'time_window_{i}_{j}_{k}'
                    )

# Solve CQM problem
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)
solution = sampleset.first.sample

# Extract solution
routes = []
for k in range(num_vehicles):
    route = []
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j and solution[f'x_{i}_{j}_{k}'] > 0.5:
                route.append((i, j))
    routes.append(route)

# Output routes as graphs
for k, route in enumerate(routes):
    G = nx.DiGraph()
    G.add_edges_from(route)
   
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title(f"Vehicle {k + 1} Route")
    plt.savefig(f"output/vehicle_{k + 1}_route.png")
    plt.clf()

print("Routes are saved to 'output' folder.")
