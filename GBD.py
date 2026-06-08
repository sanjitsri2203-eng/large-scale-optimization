!pip install gurobipy

import gurobipy as gp
from gurobipy import GRB

tol = 1e-6

# -----------------------------
# Master problem
# min eta
# -----------------------------
master = gp.Model("GBD")
master.Params.OutputFlag = 0

y = master.addVar(lb=0, vtype=GRB.INTEGER, name="y")
eta = master.addVar(lb=0, name="eta")

master.setObjective(eta, GRB.MINIMIZE)

LB = -1e20
UB = 1e20
it = 1

while True:
    # solve master
    master.optimize()

    yk = round(y.X)
    etak = eta.X
    LB = master.ObjVal

    print(f"Iteration {it}: Solve master -> LB = {LB:.6f}, y = {yk}, eta = {etak:.6f}")

    # -----------------------------
    # Subproblem with y fixed
    # -----------------------------
    sub = gp.Model("subproblem")
    sub.Params.OutputFlag = 0

    x = sub.addVar(lb=0, name="x")
    sub.setObjective(x*x + yk, GRB.MINIMIZE)

    c1 = sub.addConstr(3*x + 2*yk >= 8, name="c1")
    c2 = sub.addConstr(x + 2*yk >= 6, name="c2")

    sub.optimize()

    xk = x.X
    subobj = sub.ObjVal
    UB = min(UB, subobj)

    # KKT multipliers
    lam1 = c1.Pi
    lam2 = c2.Pi

    print(f"             Solve subproblem -> x = {xk:.6f}, UB = {UB:.6f}")
    print(f"             Multipliers: lam1 = {lam1:.6f}, lam2 = {lam2:.6f}")

    if abs(UB - LB) <= tol:
        print("STOP as LB = UB")
        break

    # Simplified GBD cut:
    # eta >= xk^2 + y + lam1(8 - 3xk - 2y) + lam2(6 - xk - 2y)

    const_term = xk*xk + lam1*(8 - 3*xk) + lam2*(6 - xk)
    coef_y = 1 - 2*lam1 - 2*lam2

    master.addConstr(eta >= const_term + coef_y*y)

    print(f"             Cut added: eta >= {const_term:.6f} {coef_y:+.6f} y\n")

    it += 1