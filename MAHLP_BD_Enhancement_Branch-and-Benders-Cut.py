import math
import time
import pandas as pd
from gurobipy import GRB, LinExpr, Model, quicksum

# ==========================================================
# 1. Data and parameters
# ==========================================================

n = 10
p = 4
alpha = 0.50

N = list(range(1, n + 1))
D_set = [(i, j) for i in N for j in N if i != j]

df_w = pd.read_excel("CAB25.xlsx", sheet_name="A")
w = {(int(a), int(b)): float(w_val) for a, b, w_val in df_w.values}

df_c = pd.read_excel("CAB25.xlsx", sheet_name="B")
c_param = {(int(a), int(b)): float(c_val) for a, b, c_val in df_c.values}

# C[i,j,k,m] is the unit route cost for i -> k -> m -> j.
C = {}
for i in N:
    for j in N:
        for k in N:
            for m in N:
                C[(i, j, k, m)] = (
                    c_param[(i, k)]
                    + alpha * c_param[(k, m)]
                    + c_param[(m, j)]
                )
# ==========================================================
# Global callback counters
# ==========================================================

iteration_count = 0
optimal_cut_count = 0
feasibility_cut_count = 0


# ==========================================================
# 2. Initial feasible solution
# ==========================================================

def setInitialSolution(y_vars, eta_vars):
    """Give Gurobi a simple first hub selection."""

    for k in N:
        y_vars[k].start = 0

    for k in range(1, p + 1):
        y_vars[k].start = 1

    for (i, j) in eta_vars:
        eta_vars[(i, j)].start = 0

# ==========================================================
# 3. Master Problem
# ==========================================================

def setupMasterProblemModel():
    """Build the MAHLP master problem used by the BBC callback."""

    master = Model("MAHLP_BBC_ij_Master")
    master.setParam("LazyConstraints", 1)

    global y_vars, eta_vars

    # y[k] = 1 means node k is selected as a hub.
    y_vars = {
        k: master.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        for k in N
    }

    # eta[i,j] estimates the routing cost for OD pair (i,j).
    eta_vars = {
        (i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
        for (i, j) in D_set
    }

    master.update()

    master.setObjective(
        quicksum(eta_vars[(i, j)] for (i, j) in D_set),
        GRB.MINIMIZE
    )

    # Exactly p hubs must be opened.
    master.addConstr(quicksum(y_vars[k] for k in N) == p, name="hub_count")
    master.update()

    print("MAHLP BBC ij master problem built.")
    return master


# ==========================================================
# 4. Dual Subproblem for one OD pair
# ==========================================================

def subProblem_ij(i, j, fixed_y):
    """
    Solve the dual subproblem only for OD pair (i, j).

    For fixed hubs, each OD pair is independent. This function is the
    MAHLP version of the per-(i,j) subproblem used in SAHLLP_BBC_ij.py.
    """

    sub = Model(f"MAHLP_dual_subproblem_{i}_{j}")
    sub.setParam("OutputFlag", 0)
    sub.setParam("DualReductions", 0)
    sub.setParam("InfUnbdInfo", 1)
    sub.setParam("Method", 0)

    # pi is unrestricted because it belongs to an equality constraint.
    pi = sub.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")

    # lambda and mu are nonnegative because they belong to hub-availability limits.
    lambda_vars = {}
    mu_vars = {}
    for k in N:
        for m in N:
            lambda_vars[(k, m)] = sub.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"lambda_{i}_{j}_{k}_{m}"
            )
            mu_vars[(k, m)] = sub.addVar(
                lb=0.0,
                vtype=GRB.CONTINUOUS,
                name=f"mu_{i}_{j}_{k}_{m}"
            )

    sub.update()

    # max pi - sum(y[k] lambda[k,m]) - sum(y[m] mu[k,m])
    sub.setObjective(
        pi
        - quicksum(fixed_y[k] * lambda_vars[(k, m)] for k in N for m in N)
        - quicksum(fixed_y[m] * mu_vars[(k, m)] for k in N for m in N),
        GRB.MAXIMIZE
    )

    # pi - lambda[k,m] - mu[k,m] <= w[i,j] * C[i,j,k,m]
    for k in N:
        for m in N:
            sub.addConstr(
                pi - lambda_vars[(k, m)] - mu_vars[(k, m)]
                <= w[(i, j)] * C[(i, j, k, m)],
                name=f"dualconstr_{i}_{j}_{k}_{m}"
            )

    sub.update()
    sub.optimize()

    status = sub.status

    if status == GRB.OPTIMAL:
        pi_value = pi.X
        lambda_value = {key: lambda_vars[key].X for key in lambda_vars}
        mu_value = {key: mu_vars[key].X for key in mu_vars}
        sub_obj = sub.objVal
        sub.dispose()
        return pi_value, lambda_value, mu_value, status, sub_obj

    if status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
        ray_pi = pi.UnbdRay
        ray_lambda = {key: lambda_vars[key].UnbdRay for key in lambda_vars}
        ray_mu = {key: mu_vars[key].UnbdRay for key in mu_vars}
        sub.dispose()
        return ray_pi, ray_lambda, ray_mu, status, None

    sub.dispose()
    return None, None, None, status, None


# ==========================================================
# 5. Lazy callback
# ==========================================================

def callBackFunction(model, where):
    """Solve all OD subproblems and add lazy Benders cuts."""

    global iteration_count, optimal_cut_count, feasibility_cut_count

    if where == GRB.Callback.MIPSOL:
        iteration_count += 1
        print(f"\n--- BBC ij callback iteration {iteration_count} ---")

        # Current integer hub solution from the master problem.
        yHat = {
            k: 1 if model.cbGetSolution(y_vars[k]) > 0.5 else 0
            for k in N
        }

        cuts_added_now = 0
        feasibility_cuts_now = 0

        for (i, j) in D_set:
            pi_value, lambda_value, mu_value, status, sub_obj = subProblem_ij(i, j, yHat)

            if status == GRB.OPTIMAL:
                expr = LinExpr(pi_value)
                rhs_value = pi_value

                for h in N:
                    coefficient_h = 0.0

                    for m in N:
                        coefficient_h += lambda_value[(h, m)]

                    for k in N:
                        coefficient_h += mu_value[(k, h)]

                    expr += -coefficient_h * y_vars[h]
                    rhs_value += -coefficient_h * yHat[h]

                eta_hat = model.cbGetSolution(eta_vars[(i, j)])

                # Add the cut when the current eta underestimates the OD cost.
                if eta_hat < rhs_value - 1e-6:
                    model.cbLazy(eta_vars[(i, j)] >= expr)
                    cuts_added_now += 1

            elif status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
                expr = LinExpr(pi_value)

                for h in N:
                    coefficient_h = 0.0

                    for m in N:
                        coefficient_h += lambda_value[(h, m)]

                    for k in N:
                        coefficient_h += mu_value[(k, h)]

                    expr += -coefficient_h * y_vars[h]

                model.cbLazy(expr <= 0)
                feasibility_cuts_now += 1

            else:
                print(f"Subproblem ({i},{j}) status:", status)

        optimal_cut_count += cuts_added_now
        feasibility_cut_count += feasibility_cuts_now

        print("OD optimality cuts added:", cuts_added_now)
        print("OD feasibility cuts added:", feasibility_cuts_now)

# ==========================================================
# 6. Final routing display
# ==========================================================

def printFinalRouting(best_y):
    """Print one cheapest route for each OD pair using the final hubs."""

    selected_hubs = [k for k in N if best_y[k] > 0.5]
    total_routing_cost = 0.0

    print("\nOpened hubs:")
    for k in selected_hubs:
        print(f"y[{k}] = 1")

    print("\nRouting decisions:")
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

# ==========================================================
# 7. Run BBC
# ==========================================================

def runCallBackBenders():
    """Run Benders Branch-and-Cut with one subproblem for each OD pair."""

    global iteration_count, optimal_cut_count, feasibility_cut_count

    iteration_count = 0
    optimal_cut_count = 0
    feasibility_cut_count = 0

    master = setupMasterProblemModel()
    setInitialSolution(y_vars, eta_vars)

    print("Starting master problem solve with ij-decomposed callback...")
    start_time = time.time()
    master.optimize(callback=callBackFunction)
    total_time = time.time() - start_time

    if master.status != GRB.OPTIMAL:
        print("\nMaster problem ended with status:", master.status)
        return master

    final_obj = master.ObjVal
    final_y = {k: y_vars[k].X for k in N}

    print("\nFinal master objective:", final_obj)
    printFinalRouting(final_y)

    print("\nEta values:")
    for (i, j) in D_set:
        value = eta_vars[(i, j)].X
        if value > 1e-6:
            print(f"eta[{i},{j}] = {value}")

    print("\nTotal callback iterations:", iteration_count)
    print("Optimality cuts added:", optimal_cut_count)
    print("Feasibility cuts added:", feasibility_cut_count)
    print("Total time taken: {:.2f} seconds".format(total_time))

    return master


if __name__ == "__main__":
    runCallBackBenders()
