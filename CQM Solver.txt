import dimod
import dwave.preprocessing as preprocessing
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler

# Load data from files
customers_df = pd.read_csv('input/customers.csv')
vehicles_df = pd.read_csv('input/vehicles.csv')
distances_df = pd.read_csv('input/distances.csv')

# Define the VRPTW problem parameters from datasets
num_customers = len(customers_df)  # Number of customers
num_vehicles = len(vehicles_df)  # Number of vehicles
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))  # Time windows for each customer
service_times = list(customers_df['service_time'])  # Service times for each customer
distances = distances_df.to_numpy()  # Distance matrix including depot

# Variables: x[i][j][k] = 1 if vehicle k travels from i to j
x = {}
for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                x[i, j, k] = dimod.Binary(f'x_{i}_{j}_{k}')

# Create the CQM
cqm = dimod.ConstrainedQuadraticModel()

# Objective: Minimize total distance
objective = dimod.BinaryQuadraticModel('BINARY')
for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j:
                objective.add_variable(x[i, j, k], distances[i, j])

cqm.set_objective(objective)

# Constraints: Each customer is visited exactly once
for j in range(1, num_customers + 1):
    cqm.add_constraint(sum(x[i, j, k] for k in range(num_vehicles) for i in range(num_customers + 1) if i != j) == 1, label=f'visit_{j}')

# Constraints: Time windows
arrival_times = {}
for i in range(num_customers + 1):
    arrival_times[i] = dimod.Real(f't_{i}', lower_bound=time_windows[i][0], upper_bound=time_windows[i][1])

for k in range(num_vehicles):
    for i in range(num_customers + 1):
        for j in range(1, num_customers + 1):
            if i != j:
                cqm.add_constraint(
                    arrival_times[j] >= arrival_times[i] + service_times[i] + distances[i, j] - (1 - x[i, j, k]) * 1000,
                    label=f'time_window_{i}_{j}_{k}'
                )

# Solve the CQM problem using D-Wave's hybrid solver
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm)
solution = sampleset.first.sample

# Extract the solution
routes = []
for k in range(num_vehicles):
    route = []
    for i in range(num_customers + 1):
        for j in range(num_customers + 1):
            if i != j and solution[f'x_{i}_{j}_{k}'] > 0.5:
                route.append((i, j))
    routes.append(route)

# Output the routes as graphs
for k, route in enumerate(routes):
    G = nx.DiGraph()
    G.add_edges_from(route)
   
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title(f"Vehicle {k + 1} Route")
    plt.savefig(f"output/vehicle_{k + 1}_route.png")
    plt.clf()

print("Routes have been saved to the 'output' folder.")
```
