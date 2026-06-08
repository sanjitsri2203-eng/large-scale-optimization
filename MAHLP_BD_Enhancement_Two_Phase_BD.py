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
# Global storage and counters
# ==========================================================

storedOptimalCuts = []
storedFeasibilityCuts = []

iteration_count = 0
optimal_cut_count = 0
feasibility_cut_count = 0

phase1_optimal_cut_count = 0
phase1_feasibility_cut_count = 0

y_vars = {}
eta_vars = {}

# ==========================================================
# 2. Helpers for cut construction
# ==========================================================

def build_od_cut_data(pi_value, lambda_value, mu_value):
    """Return constant and y-coefficients for one OD-level MAHLP cut."""

    coefficient = {h: 0.0 for h in N}

    for h in N:
        for m in N:
            coefficient[h] += lambda_value[(h, m)]

        for k in N:
            coefficient[h] += mu_value[(k, h)]

    return pi_value, coefficient


def build_cut_expression(constant, coefficient):
    """Build constant - sum_h coefficient[h] * y[h]."""

    expr = LinExpr(constant)

    for h in N:
        expr += -coefficient[h] * y_vars[h]

    return expr


def evaluate_cut_rhs(constant, coefficient, fixed_y):
    """Evaluate a cut at a specific y solution."""

    return constant - sum(coefficient[h] * fixed_y[h] for h in N)


def add_optimality_cut_to_master(master, eta_vars, i, j, constant, coefficient, prefix):
    """Add and store one OD-level optimality cut."""

    global phase1_optimal_cut_count

    expr = build_cut_expression(constant, coefficient)
    master.addConstr(eta_vars[(i, j)] >= expr, name=f"{prefix}_{i}_{j}")
    storedOptimalCuts.append(("optimal", i, j, constant, coefficient.copy()))
    phase1_optimal_cut_count += 1
    master.update()


def add_feasibility_cut_to_master(master, i, j, constant, coefficient, prefix):
    """Add and store one OD-level feasibility cut."""

    global phase1_feasibility_cut_count

    expr = build_cut_expression(constant, coefficient)
    master.addConstr(expr <= 0, name=f"{prefix}_{i}_{j}")
    storedFeasibilityCuts.append(("feasibility", i, j, constant, coefficient.copy()))
    phase1_feasibility_cut_count += 1
    master.update()


# ==========================================================
# 3. Dual Subproblem for one OD pair
# ==========================================================

def subProblem_ij(i, j, fixed_y):
    """Solve the MAHLP dual subproblem only for OD pair (i, j)."""

    sub = Model(f"MAHLP_dual_subproblem_{i}_{j}")
    sub.setParam("OutputFlag", 0)
    sub.setParam("DualReductions", 0)
    sub.setParam("InfUnbdInfo", 1)
    sub.setParam("Method", 0)

    pi = sub.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"pi_{i}_{j}")

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

    sub.setObjective(
        pi
        - quicksum(fixed_y[k] * lambda_vars[(k, m)] for k in N for m in N)
        - quicksum(fixed_y[m] * mu_vars[(k, m)] for k in N for m in N),
        GRB.MAXIMIZE
    )

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
# 4. Phase I master: LP relaxation
# ==========================================================

def setupMasterProblemModelPhase1():
    """Build the LP-relaxed MAHLP master for Phase I."""

    master = Model("MAHLP_BBC_ij_master_phase1")
    master.setParam("OutputFlag", 0)

    global y_vars, eta_vars

    y_vars = {
        k: master.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"y_{k}")
        for k in N
    }
    eta_vars = {
        (i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
        for (i, j) in D_set
    }

    master.update()

    master.addConstr(quicksum(y_vars[k] for k in N) == p, name="hub_count")
    master.setObjective(quicksum(eta_vars[(i, j)] for (i, j) in D_set), GRB.MINIMIZE)
    master.update()

    return master, eta_vars


def setMagnantiWongInitialPoint():
    """Use a central fractional point for the Phase I master."""

    central_y = p / n

    for k in N:
        y_vars[k].start = central_y

    for (i, j) in eta_vars:
        eta_vars[(i, j)].start = 0

    return {k: central_y for k in N}


def blend_core_point(y_core, y_hat, weight=0.5):
    """Move the core point toward the latest LP master solution."""

    return {
        k: (1 - weight) * y_core[k] + weight * y_hat[k]
        for k in N
    }


# ==========================================================
# 5. Phase I warm-start
# ==========================================================

def add_phase1_cuts_for_all_od(master, eta_vars, fixed_y, prefix):
    """Solve all ij subproblems and add their Phase I cuts."""

    total_sub_obj = 0.0
    unbounded_found = False

    for (i, j) in D_set:
        pi_value, lambda_value, mu_value, status, sub_obj = subProblem_ij(i, j, fixed_y)

        if status == GRB.OPTIMAL:
            constant, coefficient = build_od_cut_data(pi_value, lambda_value, mu_value)
            add_optimality_cut_to_master(
                master,
                eta_vars,
                i,
                j,
                constant,
                coefficient,
                prefix=f"{prefix}OptCut"
            )
            total_sub_obj += sub_obj

        elif status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
            constant, coefficient = build_od_cut_data(pi_value, lambda_value, mu_value)
            add_feasibility_cut_to_master(
                master,
                i,
                j,
                constant,
                coefficient,
                prefix=f"{prefix}FeasCut"
            )
            unbounded_found = True

        else:
            raise RuntimeError(f"Phase I subproblem ({i},{j}) returned status {status}")

    return total_sub_obj, unbounded_found


def runWarmStartPhase():
    """Generate stored cuts from the LP-relaxed master and ij DSPs."""

    global storedOptimalCuts, storedFeasibilityCuts
    global phase1_optimal_cut_count, phase1_feasibility_cut_count

    storedOptimalCuts.clear()
    storedFeasibilityCuts.clear()
    phase1_optimal_cut_count = 0
    phase1_feasibility_cut_count = 0

    master, eta_vars = setupMasterProblemModelPhase1()
    y_core = setMagnantiWongInitialPoint()

    phase1_start = time.time()

    master.optimize()
    if master.status != GRB.OPTIMAL:
        raise RuntimeError("Initial Phase I master was not solved to optimality.")

    lower_bound = master.ObjVal
    upper_bound = math.inf
    tolerance = 1e-4
    max_phase1_iter = 20
    iteration = 0

    while (upper_bound - lower_bound > tolerance) and (iteration < max_phase1_iter):
        iteration += 1
        print(f"\n=== Phase I Warm-Start Iter {iteration} ===")

        add_phase1_cuts_for_all_od(
            master,
            eta_vars,
            y_core,
            prefix=f"Phase1Core_{iteration}"
        )

        master.optimize()
        if master.status != GRB.OPTIMAL:
            raise RuntimeError("Phase I master was not solved to optimality.")

        lower_bound = master.ObjVal
        y_hat = {k: y_vars[k].X for k in N}

        sub_obj, unbounded_found = add_phase1_cuts_for_all_od(
            master,
            eta_vars,
            y_hat,
            prefix=f"Phase1Master_{iteration}"
        )

        if not unbounded_found:
            upper_bound = min(upper_bound, sub_obj)

        y_core = blend_core_point(y_core, y_hat)

        master.optimize()
        if master.status != GRB.OPTIMAL:
            raise RuntimeError("Phase I master was not solved to optimality after cuts.")

        lower_bound = master.ObjVal
        gap = upper_bound - lower_bound

        print(f"  LB={lower_bound:.4f}, UB={upper_bound:.4f}, Gap={gap:.4f}")

    phase1_time = time.time() - phase1_start

    print(
        f"\nPhase I complete in {phase1_time:.2f} seconds "
        f"(stored optimality cuts={phase1_optimal_cut_count}, "
        f"stored feasibility cuts={phase1_feasibility_cut_count})"
    )
    return master, eta_vars, phase1_time


# ==========================================================
# 6. Phase II master and stored cuts
# ==========================================================

def setupMasterProblemModelPhase2():
    """Build the integer BBC master for Phase II."""

    master = Model("MAHLP_BBC_ij_master_phase2")
    master.setParam("LazyConstraints", 1)

    global y_vars, eta_vars

    y_vars = {
        k: master.addVar(vtype=GRB.BINARY, name=f"y_{k}")
        for k in N
    }
    eta_vars = {
        (i, j): master.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"eta_{i}_{j}")
        for (i, j) in D_set
    }

    master.update()

    master.addConstr(quicksum(y_vars[k] for k in N) == p, name="hub_count")
    master.setObjective(quicksum(eta_vars[(i, j)] for (i, j) in D_set), GRB.MINIMIZE)
    master.update()

    return master, eta_vars


def addStoredCutsToModel(master, eta_vars):
    """Load Phase I cuts into the Phase II master."""

    for idx, cut in enumerate(storedOptimalCuts):
        _, i, j, constant, coefficient = cut
        expr = build_cut_expression(constant, coefficient)
        master.addConstr(
            eta_vars[(i, j)] >= expr,
            name=f"StoredOptCut_{idx}_{i}_{j}"
        )

    for idx, cut in enumerate(storedFeasibilityCuts):
        _, i, j, constant, coefficient = cut
        expr = build_cut_expression(constant, coefficient)
        master.addConstr(expr <= 0, name=f"StoredFeasCut_{idx}_{i}_{j}")

    master.update()


def setInitialIntegerSolution():
    """Set a simple integer starting solution for Phase II."""

    for k in N:
        y_vars[k].start = 0

    for k in range(1, p + 1):
        y_vars[k].start = 1

    for (i, j) in eta_vars:
        eta_vars[(i, j)].start = 0


# ==========================================================
# 7. Phase II callback
# ==========================================================

def callBackFunction(model, where):
    """Add BBC cuts using one dual subproblem per OD pair."""

    global iteration_count, optimal_cut_count, feasibility_cut_count

    if where == GRB.Callback.MIPSOL:
        iteration_count += 1
        print(f"\n--- Phase II ij callback iteration {iteration_count} ---")

        y_hat = {
            k: 1 if model.cbGetSolution(y_vars[k]) > 0.5 else 0
            for k in N
        }

        cuts_added_now = 0
        feasibility_cuts_now = 0
        total_sub_obj = 0.0

        for (i, j) in D_set:
            pi_value, lambda_value, mu_value, status, sub_obj = subProblem_ij(i, j, y_hat)

            if status == GRB.OPTIMAL:
                constant, coefficient = build_od_cut_data(pi_value, lambda_value, mu_value)
                expr = build_cut_expression(constant, coefficient)

                rhs_value = evaluate_cut_rhs(constant, coefficient, y_hat)
                eta_hat = model.cbGetSolution(eta_vars[(i, j)])
                total_sub_obj += sub_obj

                if eta_hat < rhs_value - 1e-6:
                    model.cbLazy(eta_vars[(i, j)] >= expr)
                    cuts_added_now += 1

            elif status in (GRB.UNBOUNDED, GRB.INF_OR_UNBD):
                constant, coefficient = build_od_cut_data(pi_value, lambda_value, mu_value)
                expr = build_cut_expression(constant, coefficient)
                model.cbLazy(expr <= 0)
                feasibility_cuts_now += 1

            else:
                print(f"Callback subproblem ({i},{j}) returned status {status}.")

        optimal_cut_count += cuts_added_now
        feasibility_cut_count += feasibility_cuts_now

        print(
            f"Callback: subproblem={total_sub_obj:.4f}, "
            f"OD cuts added={cuts_added_now}, "
            f"feasibility cuts added={feasibility_cuts_now}"
        )


# ==========================================================
# 8. Final routing display
# ==========================================================

def printFinalRouting(best_y):
    """Print one cheapest route for each OD pair using final hubs."""

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
# 9. Phase II solve and main driver
# ==========================================================

def runPhase2Benders():
    """Solve the integer BBC master after loading Phase I cuts."""

    global iteration_count, optimal_cut_count, feasibility_cut_count

    iteration_count = 0
    optimal_cut_count = 0
    feasibility_cut_count = 0

    master, eta_vars = setupMasterProblemModelPhase2()
    addStoredCutsToModel(master, eta_vars)
    setInitialIntegerSolution()

    print("Starting Phase II integer BBC solve with ij-decomposed subproblems...")
    phase2_start = time.time()
    master.optimize(callback=callBackFunction)
    phase2_time = time.time() - phase2_start

    if master.status != GRB.OPTIMAL:
        print("Phase II ended with status:", master.status)
        return master, phase2_time

    final_y = {k: y_vars[k].X for k in N}

    print("\nFinal master objective:", master.ObjVal)
    printFinalRouting(final_y)

    print("\nSum of eta values:", sum(eta_vars[(i, j)].X for (i, j) in D_set))
    print("Phase II callback iterations:", iteration_count)
    print("Phase II optimality cuts added:", optimal_cut_count)
    print("Phase II feasibility cuts added:", feasibility_cut_count)
    print(f"Phase II time: {phase2_time:.2f} seconds")
    return master, phase2_time


def runTwoPhaseBenders():
    """Run Phase I warm start followed by Phase II BBC."""

    total_start = time.time()

    print("=== Phase I: Warm-Start Cut Generation with ij Subproblems ===")
    _, _, phase1_time = runWarmStartPhase()

    print("\n=== Phase II: Integer BBC with Stored ij Cuts ===")
    _, phase2_time = runPhase2Benders()

    print("\nOverall time:", time.time() - total_start)
    print("Phase I time:", phase1_time)
    print("Phase II time:", phase2_time)
    print("Stored Phase I optimality cuts:", len(storedOptimalCuts))
    print("Stored Phase I feasibility cuts:", len(storedFeasibilityCuts))


if __name__ == "__main__":
    runTwoPhaseBenders()
