import os
import dimod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, Binary, quicksum


# Fetch input data from files
distances_df = pd.read_csv('Input/distances.csv', index_col=0)
customers_df = pd.read_csv('Input/customers.csv')
vehicles_df = pd.read_csv('Input/vehicles.csv')

# Assighned parameters
num_customers = len(customers_df) + 1  # Add 1 for the depot
num_vehicles = len(vehicles_df)
time_windows = list(zip(customers_df['start_time'], customers_df['end_time']))
service_times = list(customers_df['service_time'])
distances = distances_df.to_numpy()

# Verify array 
if distances.shape != (num_customers, num_customers):
    raise ValueError(f"Distance matrix shape {distances.shape} does not match number of customers {num_customers}")
print(f"Shape of distances array: {distances.shape}")

# Apply CQM 
cqm = ConstrainedQuadraticModel()

# Decision variables
x = {}
for k in range(num_vehicles):
    for i in range(num_customers):
        for j in range(num_customers):
            if i != j:
                x[i, j, k] = Binary(f'x_{i}_{j}_{k}')

# Objective function
objective = quicksum(distances[i, j] * x[i, j, k]
                     for i in range(num_customers)
                     for j in range(num_customers)
                     for k in range(num_vehicles)
                     if i != j)
cqm.set_objective(objective)

# C1. Each customer is visited exactly once
for j in range(1, num_customers):  # Adjusting the loop to account for the depot at index 0
    cqm.add_constraint(
        quicksum(x[i, j, k] for i in range(num_customers) for k in range(num_vehicles) if i != j) == 1,
        label=f"visit_customer_{j}"
    )

# C2. Each vehicle leaves the depot and returns to the depot
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(x[0, j, k] for j in range(1, num_customers)) == 1,
        label=f"leave_depot_{k+1}"
    )
    cqm.add_constraint(
        quicksum(x[i, 0, k] for i in range(1, num_customers)) == 1,
        label=f"return_depot_{k+1}"
    )

# C3. capacity constraint
for k in range(num_vehicles):
    cqm.add_constraint(
        quicksum(customers_df['demand'][j-1] * quicksum(x[i, j, k] for i in range(num_customers) if i != j)
                 for j in range(1, num_customers)) <= vehicles_df['capacity'][k],
        label=f"capacity_vehicle_{k+1}"
    )

# C4. elimination constraints
u = {i: Binary(f'u_{i}') for i in range(1, num_customers)}

for k in range(num_vehicles):
    for i in range(1, num_customers):
        for j in range(1, num_customers):
            if i != j:
                cqm.add_constraint(u[i] - u[j] + (num_customers - 1) * x[i, j, k] <= num_customers - 2,
                                   label=f"subtour_elimination_{i}_{j}_{k}")

# Solve CQM 
sampler = LeapHybridCQMSampler()
sampleset = sampler.sample_cqm(cqm, label="CVRPTW")
feasible_sampleset = sampleset.filter(lambda d: d.is_feasible)

# extract the best solution
if len(feasible_sampleset) > 0:
    solution = feasible_sampleset.first.sample
    print("Feasible solutions found!")
else:
    print("No feasible solution found")
    exit()

# Extract routes
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

# at least one vehicle visits all customers 
all_visited = set()
for k in range(num_vehicles):
    all_visited.update(routes[k])

missing_customers = set(range(1, num_customers)) - all_visited
if missing_customers:
    print(f"Customers not covered: {missing_customers}")
    # Reassign the missing customers 
    for missing in missing_customers:
        routes[0].insert(-1, missing)
    print(f"Reassigned missing customers to Vehicle 1: {missing_customers}")
else:
    print("All customers are covered!")

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(0, 0, 'ro', markersize=10, label='Depot')

# Plot customers
for i in range(1, num_customers):
    plt.plot(customers_df['x'][i-1], customers_df['y'][i-1], 'bo', markersize=7)
    plt.text(customers_df['x'][i-1] + 0.5, customers_df['y'][i-1] + 0.5, f'Customer {i}', fontsize=9)

# Plot routes
colors = ['b', 'g', 'r', 'c', 'm', 'y']
for k, route in routes.items():
    route_coords = [(0, 0)] + [(customers_df['x'][i-1], customers_df['y'][i-1]) for i in route[:-1]] + [(0, 0)]
    route_x, route_y = zip(*route_coords)
    plt.plot(route_x, route_y, colors[k % len(colors)] + '-', linewidth=2, label=f'Vehicle {k+1} Route')

    # Assign edges
    for i in range(len(route) - 1):
        x_mid = (route_x[i] + route_x[i + 1]) / 2
        y_mid = (route_y[i] + route_y[i + 1]) / 2
        distance = distances_df.iloc[route[i], route[i + 1]]
        plt.text(x_mid, y_mid, f'{distance}', fontsize=8, color='black')


plt.title('CVRPTW Solution')
plt.legend()

# Save solution
plt.savefig('output/routes.png')
plt.show()
