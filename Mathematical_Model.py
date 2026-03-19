import pandas as pd
import numpy as np
from math import sqrt
import gurobipy
from gurobipy import GRB, Model, quicksum
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D     
import matplotlib.patches as mpatches

# Haversine distance calculation function    
def haversine_distance(p1, p2):
   
    lat1, lon1 = p1
    lat2, lon2 = p2

    # Earth radius in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

# Reading the dataset
df = pd.read_excel("cluster_results_expected_format.xlsx", sheet_name="Depot1", index_col=0) 

 
xcoord = list(df['Longitude']) # Reading X-coordinate from the dataset
ycoord = list(df['Latitude']) # Reading Y-coordinate from the dataset
q = list(df['Demand']) # Reading Demand from the dataset
e = list(df['Earliest']) # Reading Earliest start time from the dataset
l = list(df['Latest']) # Reading Latest start time from the dataset
s = list(df['Service Time']) # Reading Service time from the dataset

# Defining the sets
location_set = len(df) 

# Amount and type of vehicles
n_small = 20
n_large = 10
n_jumbo = 24
n_truck = 6  
  

vehicle_set = n_small + n_large + n_jumbo + n_truck + 1   # +1 for "dummy 0 index in python"

# Build a type list indexed by k (keep index 0 dummy)
veh_type = ["DUMMY"] \
         + (["Small"] * n_small) \
         + (["Large"] * n_large) \
         + (["Jumbo"] * n_jumbo) \
         + (["Truck"] * n_truck)
         
# Defining the parameters
# Haversine distance calculation, Dij parameter
D = [[haversine_distance((ycoord[i], xcoord[i]), (ycoord[j], xcoord[j])) for j in range(location_set)] for i in range(location_set)]

# Convert D to a NumPy array
D = np.array(D)

# Defining dict for mat model parameters
cap_by_type = {
    "Small": 35,
    "Large": 23,
    "Jumbo": 31,
    "Truck": 6
}

fixed_cost_by_type = {
    "Small": 2000,
    "Large": 2400,
    "Jumbo": 2800,
    "Truck": 3500
}

dist_cost_by_type = {
    "Small": 6,
    "Large": 8,
    "Jumbo": 11,
    "Truck": 14
}


# Vehicle-specific dictionaries 
C = {k: cap_by_type[veh_type[k]]   for k in range(1, vehicle_set)}
F = {k: fixed_cost_by_type[veh_type[k]] for k in range(1, vehicle_set)}
A = {k: dist_cost_by_type[veh_type[k]]  for k in range(1, vehicle_set)}

vehicle_speed_kmh = 60  # Average vehicle speed which is 60 km / h

# Tij parameter calculation, distance / vehicle speed minute unit
t = [[(D[i][j] / vehicle_speed_kmh) * 60
      for j in range(location_set)]
     for i in range(location_set)]


# Tight Big-M: arc-based Mij
Mij = np.zeros((location_set, location_set))
for i in range(location_set):
    for j in range(location_set):
        if i == j:
            continue
        Mij[i, j] = max(0, l[i] + s[i] + t[i][j] - e[j])

# Defining the model
VRP = gurobipy.Model()

# Defining the decision variables
x = VRP.addVars(location_set, location_set, vehicle_set, lb=0, ub=1, vtype=GRB.BINARY, name='X') # 1 if arc (i, j) is traversed by a vehicle k; 0 otherwise
u = VRP.addVars(location_set, vehicle_set, lb = 0, vtype = GRB.CONTINUOUS,name = 'U') # Load on the vehicle type k upon its arrival at node i
b = VRP.addVars(location_set, lb = 0, vtype = GRB.CONTINUOUS,name = 'B') # Service start time of node i by one of the vehicles


# Defining the objective function, goal is minimizing the total cost (fixed cost + distance per km cost)
VRP.setObjective(
    quicksum(
        F[k] * quicksum(x[0, j, k] for j in range(1, location_set))
        for k in range(1, vehicle_set)
    )
    +
    quicksum(
        A[k] * D[i, j] * x[i, j, k]
        for k in range(1, vehicle_set)
        for i in range(location_set)
        for j in range(location_set)
        if i != j
    ),
    GRB.MINIMIZE)


# Adding the constraints

# Each customer is visited exactly once by one vehicle constraints. These are the flow balance constraints
VRP.addConstrs(quicksum(x[i,j,k] for j in range (location_set) for k in range (1, vehicle_set) if i != j) == 1 for i in range(1, location_set)) 

VRP.addConstrs(quicksum(x[i,j,k] for i in range (location_set) for k in range (1, vehicle_set) if i != j) == 1 for j in range(1, location_set)) 


# Starting from the depot constraint
VRP.addConstrs(quicksum(x[0, j, k] for j in range(1, location_set) if j != 0) == 1 for k in range(1, vehicle_set))

# Returning to the depot constraint
VRP.addConstrs(quicksum(x[i, 0, k] for i in range(1, location_set) if i != 0) == 1 for k in range(1, vehicle_set))

# Flow constraints to ensure each vehicle completes a single tour
VRP.addConstrs(quicksum(x[i, j, k] for j in range(location_set) if i != j) - quicksum(x[j, i, k] for j in range(location_set) if i != j) == 0 for i in range(1, location_set) for k in range(1, vehicle_set))

# This constraint ensures that number of vehicles leaving the depot is greater than 1
VRP.addConstr(quicksum(x[0, j, k] for j in range(1, location_set) for k in range(1, vehicle_set)) >= 1) 


# Vehicle capacity satisfaction and ensuring the demand of the customers constraints
VRP.addConstrs(u[j,k] <= u[i,k] - q[i] * x[i, j, k] + C[k] * (1-x[i,j,k])  for i in range(location_set) for j in range(1, location_set) for k in range(1, vehicle_set) if i != j)

# Visit indicator constraints, node i is visited by vehicle k
VRP.addConstrs((u[i,k] >= q[i] * quicksum(x[h, i, k] for h in range(location_set) if h != i) for i in range(1, location_set) for k in range(1, vehicle_set)))

VRP.addConstrs((u[i,k] <= C[k] * quicksum(x[h, i, k] for h in range(location_set) if h != i) for i in range(1, location_set) for k in range(1, vehicle_set)))


# These are the time window constraints
VRP.addConstrs((b[i] + s[i] + t[i][j] - Mij[i, j] * (1 - x[i, j, k]) <= b[j] for i in range(location_set) for j in range(1, location_set) for k in range(1, vehicle_set) if i != j))

VRP.addConstrs(b[j] >= e[j] for j in range(location_set))

VRP.addConstrs(b[j] <= l[j] for j in range(location_set))

# It deletes the impossible (unneccessary arcs) arcs in terms of time window ranges from the model. It accelartes the gap reduction process minimizes the search spaces
for i in range(location_set):
    for j in range(1, location_set):
        if i == j:
            continue
        if e[i] + s[i] + t[i][j] > l[j]:
            for k in range(1, vehicle_set):
                x[i, j, k].ub = 0


# Adding the optimization parameters
VRP.setParam("TimeLimit", 3600) # Time limit
VRP.setParam("MIPFocus", 1) # 0 balance, 1 focuses on finding a feasible solution, 2 focuses on gap improvement, 3 focuses on incumbent improvement
VRP.setParam("Heuristics", 0.25)     # It is the effort that gurobi gives to heuristics. 0 means closed 1 means max. 0.25 is balanced
VRP.setParam("NoRelHeurTime", 300)   # It focuses on relaxation based heuristics for the first 300 seconds. Goal is finding an incumbent
VRP.setParam("RINS", 20)             # Relaxation Induced Neighborhood Search, tries 20 times
VRP.setParam("Presolve", 1) # Presolve level. 0 means closed, 1 conservative, 2 aggressive. It tightens the bond
VRP.setParam("Cuts", 2) # Level of cuts. O means closed, 1 low, 2 medium, 3 aggressive
VRP.setParam("Symmetry", 2) # Tries to find a symmetry and tries to remove it. We have same vehicle types thats why it can be helpful
VRP.setParam("Threads", 6) # It determines how many CPU Threads will be used concurrently

# Finding the solution
VRP.update()
VRP.optimize()

status = VRP.status
object_Value = VRP.objVal

print()
print("Model status is: ", status)
print()
print("Objective Function value is: ", object_Value)


# Printing the decision varibales which are not zeros
if status !=3 and status != 4:
    for v in VRP.getVars():
        if VRP.objVal < 1e+99 and v.x!=0:
            print('%s %f'%(v.Varname,v.x))
            

# Print section
if VRP.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL] and VRP.SolCount > 0:

    routes = {}

    # Extract routes
    for k in range(1, vehicle_set):
        if sum(x[0, j, k].X for j in range(1, location_set)) > 0.5:
            route = [0]
            current = 0

            while True:
                next_nodes = [
                    j for j in range(location_set)
                    if j != current and x[current, j, k].X > 0.5
                ]
                if not next_nodes:
                    break

                next_node = next_nodes[0]
                route.append(next_node)
                current = next_node

                if current == 0:
                    break

            routes[k] = route

    def route_distance(route):
        return sum(D[route[i], route[i+1]] for i in range(len(route) - 1))

    def route_load(route):
        return sum(q[i] for i in route if i != 0)


    # Printing the routes
    print("\nEstablished Routes:")
    for k, route in routes.items():
        print(f"Vehicle {k} route: {route}")

    # Printing the route summary
    print("\nRoute Summary:")
    for k, route in routes.items():
        dist = route_distance(route)
        load = route_load(route)
        print(f"Vehicle {k} | Distance: {dist:.4f} | Load: {load:.2f} | Cap: {C[k]}")

# Visualization section
plt.figure(figsize=(10, 8), facecolor='white')
ax = plt.gca()
ax.set_facecolor('white')
ax.grid(False)

# Customers
plt.scatter(
    xcoord[1:],   # X = Longitude
    ycoord[1:],   # Y = Latitude
    s=40,
    c='black',
    label="Customers",
    zorder=3
)

# Depot
plt.scatter(
    xcoord[0],    # X = Longitude
    ycoord[0],    # Y = Latitude
    s=150,
    c='red',
    marker='s',
    label="Depot",
    zorder=5
)


# Node labels
for i in range(location_set):
    plt.text(
        xcoord[i] + 0.0001,   # lon
        ycoord[i] + 0.0001,   # lat
        str(i),
        fontsize=9,
        zorder=6
    )


# Routes
cmap = plt.cm.get_cmap("tab20")

for k, route in routes.items():
    color = cmap((k - 1) % 20)

    for idx in range(len(route) - 1):
        ii = route[idx]
        jj = route[idx + 1]

        plt.plot(
            [xcoord[ii], xcoord[jj]],   # Longitude
            [ycoord[ii], ycoord[jj]],   # Latitude
            linewidth=2,
            color=color,
            label=f"Vehicle {k}" if idx == 0 else None,
            zorder=2
        )


# Labels & legend
plt.title("Visualization of the Established VRP Routes")
plt.xlabel("Longitude (X)")
plt.ylabel("Latitude (Y)")

handles, labels = plt.gca().get_legend_handles_labels()
unique = dict(zip(labels, handles))
plt.legend(unique.values(), unique.keys(), loc="best")

plt.show()
                       