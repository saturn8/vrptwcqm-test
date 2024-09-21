import os
import dimod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dimod import ConstrainedQuadraticModel, quicksum, Real
from dwave.system import LeapHybridCQMSampler
from scipy.sparse import csr_matrix
import dask.dataframe as dd

def load_data(filepath):
    return dd.read_csv(filepath).compute() 

# Load large dataset
distances_df = load_data('Input/travel_times.csv').set_index('Unnamed: 0')
customers_df = load_data('Input/customers.csv')
vehicles_df = load_data('Input/vehicles.csv')

# Convert travel time into matrix
distances = csr_matrix(distances_df.to_numpy(dtype=float)) 

num_customers = len(customers_df) + 1  
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])

# Verify travel time matrix
if distances.shape != (num_customers, num_customers):
    raise ValueError(f"Distance matrix shape {distances.shape} does not match no. of customers {num_customers}")
print(f"Shape of distances array: {distances.shape}")

# CQM initialise
cqm = ConstrainedQuadraticModel()
sampler = LeapHybridCQMSampler()
x = {}
for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                x[i, j, k] = dimod.Binary(f'x_{i}_{j}_{k}')

start_times = {i: Real(f'start_time_{i}', lower_bound=0, upper_bound=max(customers_df['end_time'])) for i in range(1, num_customers)}

# Objective function - minimize total travel time
objective = dimod.BinaryQuadraticModel({}, {}, 0.0, 'BINARY')

# Add linear terms
for i in range(num_customers):
    for j in range(num_customers):
        for k in range(num_vehicles):
            if i != j and distances[i, j] != 0:
                objective.add_linear(f'x_{i}_{j}_{k}', distances[i, j])

cqm.set_objective(objective)

# Each customer visited once
for j in range(1, num_customers):
    cqm.add_constraint(
        quicksum(x[i, j, k] for i in range(num_customers) for k in range(num_vehicles) if i != j) == 1,
        label=f"visit_customer_{j}"
    )

# Each vehicle returns to depot
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(x[0, j, k] for j in range(1, num_customers)) == 1,
        label=f"leave_depot_{k+1}"
    )
    cqm.add_constraint(
        quicksum(x[i, 0, k] for i in range(1, num_customers)) == 1,
        label=f"return_depot_{k+1}"
    )

# Capacity constraint
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(
            customers_df['demand'][j-1] * quicksum(x[i, j, k] for i in range(1, num_customers) if i != j)
            for j in range(1, num_customers)
        ) <= vehicles_df['capacity'][k],
        label=f"capacity_vehicle_{k+1}"
    )

# Time window constraint
slack = 1  # 1-hour slack
for i in range(1, num_customers):
    lower_bound = time_windows[i-1][0] - slack
    upper_bound = time_windows[i-1][1] + slack
    
    # Ensure start time is within time window for each customer
    cqm.add_constraint(start_times[i] >= lower_bound, label=f"time_window_lower_bound_{i}")
    cqm.add_constraint(start_times[i] <= upper_bound, label=f"time_window_upper_bound_{i}")

# Solve CQM with a time limit
sampleset = sampler.sample_cqm(cqm, label="CVRPTW with Time Windows", time_limit=300)
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

# Find solution
if len(feasible_sampleset) > 0:
    solution = feasible_sampleset.first.sample
    print("Feasible solutions found!")
else:
    print("No feasible solution found")
    exit()

# find routes
routes = {k: [] for k in range(num_vehicles)}

for k in range(num_vehicles):
    current_node = 0  
    visited = set([0])
    while len(visited) < num_customers:
        next_node = None
        for j in range(num_customers):
            if current_node != j and solution.get(f'x_{current_node}_{j}_{k}', 0) and j not in visited:
                next_node = j
                break
        if next_node is None:
            break
        routes[k].append(next_node)
        visited.add(next_node)
        current_node = next_node
    routes[k].append(0)  

# verify customers are covered
all_visited = set().union(*[set(route) for route in routes.values()])
missing_customers = set(range(1, num_customers)) - all_visited
if missing_customers:
    print(f"Customers not covered: {missing_customers}")
    for missing in missing_customers:
        routes[0].insert(-1, missing) 
    print(f"Reassigned missing customers to Vehicle 1: {missing_customers}")
else:
    print("All customers are covered!")

# Plot data
def plot_routes(routes, plot_graph=True):
    if not plot_graph: 
        return
    plt.figure(figsize=(10, 6))
    plt.plot(0, 0, 'ro', markersize=10, label='Depot')

    for i in range(1, num_customers):
        plt.plot(customers_df['x'][i-1], customers_df['y'][i-1], 'bo', markersize=7)
        plt.text(customers_df['x'][i-1] + 0.5, customers_df['y'][i-1] + 0.5, f'Customer {i}', fontsize=9)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for k, route in routes.items():
        route_coords = [(0, 0)] + [(customers_df['x'][i-1], customers_df['y'][i-1]) for i in route[:-1]] + [(0, 0)]
        route_x, route_y = zip(*route_coords)
        plt.plot(route_x, route_y, colors[k % len(colors)] + '-', linewidth=2, label=f'Vehicle {k+1} Route')

    plt.title('Routes')
    plt.legend()
    plt.show()

# Plot routes for small dataset
plot_routes(routes, plot_graph=False)

# Save routes to file
output_file = 'output/routes_large.csv'
pd.DataFrame.from_dict(routes, orient='index').to_csv(output_file)
print(f"Solution saved to {output_file}")
