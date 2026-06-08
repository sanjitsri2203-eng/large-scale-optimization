import pandas as pd         # pandas is used to read data from the Excel file.
import gurobipy as gp       # gurobipy is the Python package for the Gurobi optimization solver.
from gurobipy import GRB    # GRB gives us common Gurobi constants such as BINARY and MINIMIZE.

# ==========================================================
# 1. Read demand and cost data
# ==========================================================

# Demand data: w[i, j] = demand from node i to node j
df = pd.read_excel("CAB25.xlsx", "A")
w = dict(((int(a), int(b)), float(c)) for a, b, c in df.values)

# Cost data: c[i, j] = travel cost from node i to node j
df1 = pd.read_excel("CAB25.xlsx", "B")
c = dict(((int(a), int(b)), float(cval)) for a, b, cval in df1.values)

# ==========================================================
# 2. Define model settings
# ==========================================================

N = 10                       # number of nodes used
nodes = range(1, N + 1)      # node numbers: 1 to 10

p = 4                        # number of hubs to open
alpha = 0.5                  # discount factor for hub-to-hub travel

# ==========================================================
# 3. Calculate route costs
# ==========================================================

# C[i, j, k, m] = cost of route i -> k -> m -> j
C = {}

for i in nodes:
    for j in nodes:
        for k in nodes:
            for m in nodes:
                C[i, j, k, m] = c[i, k] + alpha * c[k, m] + c[m, j]

# ==========================================================
# 4. Create optimization model
# ==========================================================

model = gp.Model("Multiple_Allocation_Hub_Location_Problem")

# ==========================================================
# 5. Decision variables
# ==========================================================

# y[k] = 1 if node k is opened as a hub, otherwise 0
y = model.addVars(nodes, vtype=GRB.BINARY, name="y")

# x[i, j, k, m] = 1 if flow from i to j uses hubs k and m
x = model.addVars(nodes, nodes, nodes, nodes, vtype=GRB.BINARY, name="x")

# ==========================================================
# 6. Objective function
# ==========================================================

# Minimize total demand-weighted transportation cost
model.setObjective(
    gp.quicksum(
        w[i, j] * C[i, j, k, m] * x[i, j, k, m]
        for i in nodes
        for j in nodes
        for k in nodes
        for m in nodes
    ),
    GRB.MINIMIZE
)

# ==========================================================
# 7. Constraints
# ==========================================================

# Open exactly p hubs
model.addConstr(
    gp.quicksum(y[k] for k in nodes) == p,
    name="Hub_Opening_Limit"
)

# Each origin-destination pair must use exactly one hub pair
model.addConstrs(
    (
        gp.quicksum(x[i, j, k, m] for k in nodes for m in nodes) == 1
        for i in nodes
        for j in nodes
        if i != j
    ),
    name="Flow_Allocation"
)

# First hub k can be used only if it is opened
model.addConstrs(
    (
        x[i, j, k, m] <= y[k]
        for i in nodes
        for j in nodes
        for k in nodes
        for m in nodes
        if i != j
    ),
    name="First_Hub_Activation"
)

# Second hub m can be used only if it is opened
model.addConstrs(
    (
        x[i, j, k, m] <= y[m]
        for i in nodes
        for j in nodes
        for k in nodes
        for m in nodes
        if i != j
    ),
    name="Second_Hub_Activation"
)

# ==========================================================
# 8. Solve the model
# ==========================================================

model.optimize()

# ==========================================================
# 9. Print results
# ==========================================================

if model.status == GRB.OPTIMAL:

    print("\nOptimal Objective Value:", model.objVal)

    print("\nOpened Hubs:")
    for k in nodes:
        if y[k].X > 0.5:
            print(f"Hub opened at node {k}")

    print("\nRouting Decisions:")
    for i in nodes:
        for j in nodes:
            if i != j:
                for k in nodes:
                    for m in nodes:
                        if x[i, j, k, m].X > 0.5:
                            print(f"Flow from {i} to {j} is routed through hubs ({k}, {m})")
