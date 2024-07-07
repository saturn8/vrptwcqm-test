import os
import dimod
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler


api_token = os.getenv('DWAVE_API_TOKEN')
if not api_token:
    raise ValueError("API token not defined. Set DWAVE_API_TOKEN environment variable.")
print(f"Using API token: {api_token}")

# Load data from files
distances_df = pd.read_csv('Input/distances.csv', index_col=0)
customers_df = pd.read_csv('Input/customers.csv')
vehicles_df = pd.read_csv('Input/vehicles.csv')


# Define problem parameters from datasets
num_customers = len(customers_df)  
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])
distances = distances_df.to_numpy()

# Distances array match required dimensions
if distances.shape != (num_customers, num_customers):
    raise ValueError(f"Distance matrix shape {distances.shape} does not match no of customers and depot {num_customers}")
print(f"Shape of distances array: {distances.shape}")

# Variables: x[i][j][k] = 1 if vehicle k travels from i to j
x = {}
for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                x[i, j, k] = dimod.Binary(f'x_{i}_{j}_{k}')

# Create CQM
cqm = dimod.ConstrainedQuadraticModel()

# Objective: Minimize total distance
objective = dimod.BinaryQuadraticModel(dimod.BINARY)
for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                objective.set_linear(f'x_{i}_{j}_{k}', distances[i, j])

cqm.set_objective(objective)

# Constraints: Each customer is visited exactly once
for j in range(1, num_customers):
    cqm.add_constraint(
        sum(x[i, j, k] for k in range(num_vehicles) for i in range(num_customers) if i != j) == 1,
        label=f'visit_{j}'
    )

# Constraints: Time window
arrival_times = {}
for i in range(num_customers):
    if i < len(time_windows):
        arrival_times[i] = dimod.Real(f't_{i}', lower_bound=time_windows[i][0], upper_bound=time_windows[i][1])
    else:
        print(f"Warning: Customer index {i} is out of range for time_windows.")

for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j and i in arrival_times and j in arrival_times:
                service_plus_distance = service_times[i] + distances[i, j]
                time_constraint = arrival_times[j] - arrival_times[i] - service_plus_distance + 1000 * x[i, j, k]
                cqm.add_constraint(time_constraint >= 0, label=f'time_window_{i}_{j}_{k}')

# Solve CQM
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)
solution = sampleset.first.sample

# Solution
routes = []
for k in range(num_vehicles):
    route = []
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j and solution[f'x_{i}_{j}_{k}'] > 0.5:
                route.append((i, j))
    routes.append(route)

# Generate graphs
for k, route in enumerate(routes):
    G = nx.DiGraph()
    G.add_edges_from(route)
   
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title(f"Vehicle {k + 1} Route")
    plt.savefig(f"output/vehicle_{k + 1}_route.png")
    plt.clf()

print("Routes are saved in 'output' folder.")
