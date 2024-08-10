import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, quicksum, Integer
import numpy as np

# Load input files
customers = pd.read_csv('Input/customers.csv')
distances = pd.read_csv('Input/distances.csv', index_col=0)
vehicles = pd.read_csv('Input/vehicles.csv')

# Debugging: print data
print("Customers Data:")
print(customers)
print("Distances Data:")
print(distances)
print("Vehicles Data:")
print(vehicles)

# Number of customers and vehicles
num_customers = len(customers)
num_vehicles = len(vehicles)

# Extract vehicle capacities
vehicle_capacities = vehicles['capacity'].values

# Time windows and service times
time_windows = customers[['start_time', 'end_time']].values
service_times = customers['service_time'].values

# Create distance matrix
distance_matrix = distances.values

# Check if distance matrix is square
if distance_matrix.shape[0] != distance_matrix.shape[1]:
    raise ValueError(f"The distance matrix is not square. Found shape: {distance_matrix.shape}")
print("Distance matrix shape:", distance_matrix.shape)

# Create binary variables
x = {}
for i in range(num_customers):
    for j in range(num_customers):
        if i != j:
            for k in range(num_vehicles):
                x[i, j, k] = Binary(f'x_{i}_{j}_{k}')

# Create CQM model
cqm = ConstrainedQuadraticModel()

# Objective: Minimize total distance
objective = quicksum(distance_matrix[i, j] * x[i, j, k]
                     for i in range(num_customers) for j in range(num_customers) if i != j for k in range(num_vehicles))
cqm.set_objective(objective)

# Constraint 1: Each customer is visited exactly once
for i in range(1, num_customers):
    cqm.add_constraint(quicksum(x[i, j, k] for j in range(num_customers) if i != j for k in range(num_vehicles)) == 1, label=f'visit_once_{i}')

# Constraint 2: Vehicle capacity
for k in range(num_vehicles):
    cqm.add_constraint(quicksum(customers['demand'][i] * quicksum(x[i, j, k] for j in range(num_customers) if i != j) for i in range(num_customers)) <= vehicle_capacities[k], label=f'vehicle_capacity_{k}')

# Time variables (Integer)
time_vars = {i: Integer(f't_{i}', lower_bound=0, upper_bound=1440) for i in range(num_customers)}

# Solve the CQM problem
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm, time_limit=120)

# Check if there are any feasible solutions
feasible_sampleset = sampleset.filter(lambda sample: sample.is_feasible)
if len(feasible_sampleset) == 0:
    print("No feasible solutions found. Reviewing constraints and data.")
else:
    best_solution = feasible_sampleset.first.sample
    print("Feasible solutions found!")

    # Extract routes from solution
    routes = [[] for _ in range(num_vehicles)]
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                for k in range(num_vehicles):
                    if best_solution[f'x_{i}_{j}_{k}']:
                        routes[k].append((i, j))

    # Plot the routes
    G = nx.DiGraph()
    positions = {i: (customers['x'][i], customers['y'][i]) for i in range(num_customers)}
    for k, route in enumerate(routes):
        color = np.random.rand(3,)
        for (i, j) in route:
            G.add_edge(i, j, color=color, weight=2)

    edges = G.edges()
    colors = [G[u][v]['color'] for u, v in edges]
    weights = [G[u][v]['weight'] for u, v in edges]

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos=positions, edge_color=colors, width=weights, with_labels=True, node_size=500, node_color='lightblue', font_size=10)
    plt.title('Vehicle Routes')

    # Ensure the output directory exists
    output_dir = '/workspaces/vrptwcqm-test/output/'
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot as a PNG image in the output directory
    output_file = os.path.join(output_dir, 'routes.png')
    plt.savefig(output_file)
    plt.show()

    print(f"Routes have been plotted and saved to '{output_file}'.")
