import numpy as np
import pandas as pd
import math
import time
import random
try:
    from gurobipy import Model, GRB, quicksum
except ImportError:
    print("Warning: gurobipy is not installed. Please install it using: pip install gurobipy")

# ==========================================================
# 1. Data and parameters
# ==========================================================
n = 10
p = 4
alpha = 0.5

N = list(range(1, n + 1))                                       # Nodes in the network
A = [(i, j, k, m) for i in N for j in N for k in N for m in N]  # All possible origin-destination-hub-hub combinations
D = [(i, j) for i in N for j in N]                              # All origin-destination pairs
D_set = [(i, j) for i in N for j in N if i != j]                # Origin-destination pairs with i ≠ j

df_w = pd.read_excel("CAB25.xlsx", sheet_name="A")
w = {(int(a), int(b)): float(w_val) for a, b, w_val in df_w.values}

df_c = pd.read_excel("CAB25.xlsx", sheet_name="B")
c_param = {(int(a), int(b)): float(c_val) for a, b, c_val in df_c.values}

# C[i,j,k,m] is the cost of route i -> k -> m -> j.
C = {}
for i in N:
    for j in N:
        for k in N:
            for m in N:
                C[(i, j, k, m)] = c_param[(i, k)] + alpha * c_param[(k, m)] + c_param[(m, j)]

# ==========================================================
# 2. Master Problem
# ==========================================================

master = Model("master")
master.setParam("OutputFlag", 1)

# y[k] = 1 if node k is opened as a hub.
y_vars = {k: master.addVar(vtype=GRB.BINARY, name=f"y_{k}") for k in N}

# theta approximates the routing cost obtained from the dual subproblem.
theta_var = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
master.update()

master.setObjective(theta_var, GRB.MINIMIZE)

# Exactly p hubs must be opened.
master.addConstr(quicksum(y_vars[k] for k in N) == p, name="hub_count")
master.update()


# ==========================================================
# 3. Initial solution for the first subproblem
# ==========================================================

for k in N:
    y_vars[k].start = 0

for k in range(1, p + 1):
    y_vars[k].start = 1

theta_var.start = 0
master.update()

fixed_y = {k: y_vars[k].start for k in N}


# ==========================================================
# 4. Benders Decomposition Loop
# ==========================================================

Lower_bound = 0
Upper_bound = math.inf
max_iter = 500
tolerance = 1e-6
iteration = 0
start_time = time.time()

optimal_cut_count = 0
feasibility_cut_count = 0

best_y = fixed_y.copy()
best_objective = math.inf

print("\nClassical Benders Decomposition for MAHLP")

while (Upper_bound - Lower_bound > tolerance) and (iteration < max_iter):
    iteration += 1

    # Use the latest master solution. In the first iteration, use the start.
    try:
        current_y = {k: 1 if y_vars[k].X > 0.5 else 0 for k in N}
    except AttributeError:
        current_y = fixed_y

    fixed_y = current_y

    # ======================================================
    # 4.1 Dual Subproblem
    # ======================================================

    sub = Model("dual_subproblem")
    sub.setParam("OutputFlag", 0)
    sub.setParam("Method", 0)

    # pi[i,j] is unrestricted.
    pi_vars = {}
    for (i, j) in D_set:
        pi_vars[(i, j)] = sub.addVar(
            lb=-GRB.INFINITY,
            vtype=GRB.CONTINUOUS,
            name=f"pi_{i}_{j}"
        )

    # lambda and mu are nonnegative dual variables.
    lambda_vars = {}
    mu_vars = {}
    for (i, j) in D_set:
        for k in N:
            for m in N:
                lambda_vars[(i, j, k, m)] = sub.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"lambda_{i}_{j}_{k}_{m}"
                )
                mu_vars[(i, j, k, m)] = sub.addVar(
                    lb=0.0,
                    vtype=GRB.CONTINUOUS,
                    name=f"mu_{i}_{j}_{k}_{m}"
                )
    sub.update()

    Sum_sub1 = quicksum(pi_vars[(i, j)] for (i, j) in D_set)
    Sum_sub2 = -quicksum(
        fixed_y[k] * lambda_vars[(i, j, k, m)]
        for (i, j) in D_set
        for k in N
        for m in N
    )
    Sum_sub3 = -quicksum(
        fixed_y[m] * mu_vars[(i, j, k, m)]
        for (i, j) in D_set
        for k in N
        for m in N
    )

    sub.setObjective(Sum_sub1 + Sum_sub2 + Sum_sub3, GRB.MAXIMIZE)

    for (i, j) in D_set:
        for k in N:
            for m in N:
                rhs = w[(i, j)] * C[(i, j, k, m)]
                sub.addConstr(
                    pi_vars[(i, j)]
                    - lambda_vars[(i, j, k, m)]
                    - mu_vars[(i, j, k, m)]
                    <= rhs,
                    name=f"dualconstr_{i}_{j}_{k}_{m}"
                )

    sub.update()
    sub.optimize()

    if sub.status == GRB.OPTIMAL:
        sub_obj_val = sub.objVal
        print("Subproblem (dual) optimal objective:", sub_obj_val)

        pi_val = {key: pi_vars[key].X for key in pi_vars}
        lambda_val = {key: lambda_vars[key].X for key in lambda_vars}
        mu_val = {key: mu_vars[key].X for key in mu_vars}

        if sub_obj_val < Upper_bound:
            Upper_bound = sub_obj_val
            best_objective = sub_obj_val
            best_y = fixed_y.copy()

        # Benders optimality cut:
        # theta >= sum(pi) - sum(y[k] * lambda) - sum(y[m] * mu)
        constant = sum(pi_val[(i, j)] for (i, j) in D_set)
        coefficient = {k: 0.0 for k in N}

        for (i, j) in D_set:
            for k in N:
                for m in N:
                    coefficient[k] += lambda_val[(i, j, k, m)]
                    coefficient[m] += mu_val[(i, j, k, m)]

        Cut_rhs = constant - quicksum(coefficient[k] * y_vars[k] for k in N)
        master.addConstr(theta_var >= Cut_rhs, name=f"BendersCut_{iteration}")
        master.update()
        optimal_cut_count += 1

    elif sub.status in (GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        print("Subproblem infeasible or unbounded! Adding feasibility cut.")

        sub.setParam("DualReductions", 0)
        sub.setParam("InfUnbdInfo", 1)
        sub.optimize()

        if sub.status == GRB.UNBOUNDED:
            ray_pi = {key: pi_vars[key].UnbdRay for key in pi_vars}
            ray_lambda = {key: lambda_vars[key].UnbdRay for key in lambda_vars}
            ray_mu = {key: mu_vars[key].UnbdRay for key in mu_vars}

            ray_constant = sum(ray_pi[(i, j)] for (i, j) in D_set)
            ray_coefficient = {k: 0.0 for k in N}

            for (i, j) in D_set:
                for k in N:
                    for m in N:
                        ray_coefficient[k] += ray_lambda[(i, j, k, m)]
                        ray_coefficient[m] += ray_mu[(i, j, k, m)]

            Feas_rhs = ray_constant - quicksum(ray_coefficient[k] * y_vars[k] for k in N)
            master.addConstr(Feas_rhs <= 0, name=f"FeasCut_{iteration}")
            master.update()
            feasibility_cut_count += 1
        else:
            raise RuntimeError("Dual subproblem did not provide an unbounded ray.")

    else:
        print("Subproblem status:", sub.status)

    # ======================================================
    # 4.2 Solve Master Problem after adding the new cut
    # ======================================================

    master.setParam("OutputFlag", 0)
    master.optimize()

    if master.status != GRB.OPTIMAL:
        raise RuntimeError("Master problem was not solved to optimality.")

    Lower_bound = master.ObjVal
    gap = Upper_bound - Lower_bound

    print("---------------------- iteration", iteration, "-------------------")
    print("LB =", Lower_bound, ", UB =", Upper_bound, ", gap =", gap)

    sub.dispose()
# ==========================================================
# 5. Final results
# ==========================================================

print("\nFinal Upper Bound:", Upper_bound)
print("Final Lower Bound:", Lower_bound)
print("Final Gap:", Upper_bound - Lower_bound)
print("Time elapsed:", time.time() - start_time)

if Upper_bound - Lower_bound <= tolerance:
    print("Status: converged")
else:
    print("Status: maximum iterations reached before convergence")

print("\nBest objective value:", best_objective)

print("\nOpened hubs from best feasible solution:")
for k in N:
    if best_y[k] > 0.5:
        print(f"y[{k}] = 1")

selected_hubs = [k for k in N if best_y[k] > 0.5]
total_routing_cost = 0.0

print("\nRouting decisions from best feasible solution:")
for (i, j) in D_set:
    best_route_cost = math.inf
    best_pair = None

    for k in selected_hubs:
        for m in selected_hubs:
            current_cost = w[(i, j)] * C[(i, j, k, m)]

            if current_cost < best_route_cost:
                best_route_cost = current_cost
                best_pair = (k, m)

    total_routing_cost += best_route_cost
    print(f"Flow from {i} to {j} is routed through hubs {best_pair}")

print("\nCheck cost from final routing:", total_routing_cost)
print("theta =", theta_var.X)
print("Optimality cuts added:", optimal_cut_count)
print("Feasibility cuts added:", feasibility_cut_count)
