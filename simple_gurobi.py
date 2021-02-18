from typing import Iterable, Optional
from dataclasses import dataclass
import gurobipy as gp
import numpy
from gurobipy import GRB


# AUTHOR: DUSTIN KENEFAKE
# LICENSE: MIT
# DATE: 2020


@dataclass
class SolverOutput:
    """
    Solver information object

    Members:
    obj: objective value of the optimal solution \n
    sol: x*, numpy.ndarray \n

    Optional Parameters -> None or numpy.ndarray type

    slack: the slacks associated with every constraint \n
    active_set: the active set of the solution, including strongly and weakly active constraints \n
    dual: the lagrange multipliers associated with the problem\n

    """
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]

    def __eq__(self, other):
        if not isinstance(other, SolverOutput):
            return NotImplemented

        return numpy.allclose(self.slack, other.slack) and numpy.allclose(self.active_set,
                                                                          other.active_set) and numpy.allclose(
            self.dual, other.dual) and numpy.allclose(self.sol, other.sol) and (self.obj - other.obj) ** 2 < 10 ** -10



def get_program_parameters(Q: Optional[numpy.ndarray], c: Optional[numpy.ndarray], A: Optional[numpy.ndarray],
                           b: Optional[numpy.ndarray]):
    """ Given a set of possibly None optimization parameters determine the number of variables and constraints """
    num_c = 0
    num_v = 0

    if Q is not None:
        num_v = Q.shape[0]

    if A is not None:
        num_v = A.shape[1]
        num_c = b.shape[0]

    if c is not None:
        num_v = numpy.size(c)

    return num_v, num_c

def solve_miqp_gurobi(Q: numpy.ndarray = None, c: numpy.ndarray = None, A: numpy.ndarray = None,
                      b: numpy.ndarray = None,
                      equality_constraints: Iterable[int] = None,
                      bin_vars: Iterable[int] = None, verbose: bool = False,
                      get_duals: bool = True) -> Optional[SolverOutput]:
    """
    This is the breakout for solving mixed integer quadratic programs with gruobi

    The Mixed Integer Quadratic program programming problem
        min_{xy} 1/2 [xy]^T*Q*[xy] + c^T*[xy]

        s.t.   A[xy] <= b
               Aeq*[xy] = beq

               xy is the parameter vector of mixed real and binary inputs

    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A SolverOuput Object
    """

    model = gp.Model()

    if not verbose:
       model.setParam("OutputFlag", 0)

    if equality_constraints is None:
        equality_constraints = []

    if bin_vars is None:
        bin_vars = []

    if len(bin_vars) == 0:
        model.setParam("Method", 0)

    if len(bin_vars) == 0 and Q is None:
        model.setParam("Method", 0)
        model.setParam("Quad", 0)

    # model.setParam('NumericFocus', 3)
    # model.setParam("FeasibilityTol", 10 ** (-9))
    # model.setParam("OptimalityTol", 10 ** (-9))
    # model.setParam("Presolve", 2)

    # define num variables and num constraints variables
    num_vars, num_constraints = get_program_parameters(Q, c, A, b)

    if A is None and Q is None:
        return None

    var_types = [GRB.BINARY if i in bin_vars else GRB.CONTINUOUS for i in range(num_vars)]
    x = model.addMVar(num_vars, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=var_types)

    if num_constraints != 0:
        # sense = numpy.chararray(num_constraints)
        sense = [GRB.LESS_EQUAL for _ in range(num_constraints)]
        for i in equality_constraints:
            sense[i] = GRB.EQUAL

        # sense.fill(GRB.LESS_EQUAL)
        # sense[equality_constraints] = GRB.EQUAL
        # inequality = [i for i in range(num_constraints) if i not in equality_constraints]
        # sense[inequality] = GRB.LESS_EQUAL

        model.addMConstr(A, x, sense, b)

    objective = 0

    if Q is not None and c is None:
        objective = .5 * (x @ Q @ x)

    if c is not None and Q is None:
        objective = c.flatten() @ x

    if Q is not None and c is not None:
        objective = .5 * (x @ Q @ x) + c.flatten() @ x

    model.setObjective(objective, sense=GRB.MINIMIZE)

    model.optimize()
    model.update()

    # get gurobi status
    status = model.status
    # if not solved return None
    if status != GRB.OPTIMAL and status != GRB.SUBOPTIMAL:
        return None

    # create the solver return object
    sol = SolverOutput(obj=model.getAttr("ObjVal"), sol=numpy.array(x.X), slack=None,
                       active_set=None, dual=None)

    # if we have a constrained system we need to add in the slack variables and active set
    if num_constraints != 0:

        if get_duals:
            # dual variables only really make sense if the system doesn't have binaries
            if len(bin_vars) == 0:
                sol.dual = numpy.array(model.getAttr("Pi"))

        sol.slack = numpy.array(model.getAttr("Slack"))
        sol.active_set = numpy.where((A @ sol.sol.flatten() - b.flatten()) ** 2 < 10 ** -12)[0]

    return sol


def solve_qp_gurobi(Q: numpy.ndarray, c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                    equality_constraints: Iterable[int] = None,
                    verbose=False,
                    get_duals=True) -> Optional[SolverOutput]:
    """
    This is the breakout for solving mixed integer quadratic programs with gruobi

    The Mixed Integer Quadratic program programming problem
        min_{xy} 1/2 [xy]^T*Q*[xy] + c^T*[xy]

        s.t.   A[xy] <= b
              Aeq*[xy] = beq

              xy is the parameter vector of mixed real and binary inputs

    :param Q: Square matrix, can be None
    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param verbose: Flag for output of underlying solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A SolverOuput Object
    """
    return solve_miqp_gurobi(Q=Q, c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


# noinspection PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList,PyArgumentList
def solve_lp_gurobi(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray, equality_constraints=None, verbose=False,
                    get_duals=True) -> Optional[SolverOutput]:
    """
    This is the breakout for solving mixed integer linear programs with gruobi, This is feed directly into the
    MIQP solver that is defined in the same file.

    The Mixed Integer Linear program programming problem
        min_{xy} c^T*[xy]

        s.t.    A[xy] <= b
                Aeq*[xy] = beq

                xy is the parameter vector of mixed real and binary inputs

    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param verbose: Flag for output of underlying solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True

    :return: A SolverOuput Object
    """

    # Simple short cuts that indicate a unbounded or infeasible LP

    if A is None or b is None:
        return None

    if numpy.size(A) == 0 or numpy.size(b) == 0:
        return None

    return solve_miqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, verbose=verbose,
                             get_duals=get_duals)


def solve_milp_gurobi(c: numpy.ndarray, A: numpy.ndarray, b: numpy.ndarray,
                      equality_constraints: Iterable[int] = None,
                      bin_vars: Iterable[int] = None, verbose=False, get_duals=True) -> Optional[
    SolverOutput]:
    """
    This is the breakout for solving mixed integer linear programs with gruobi, This is feed directly into the
    MIQP solver that is defined in the same file.

    The Mixed Integer Linear program programming problem
        min_{xy} c^T*[xy]

        s.t.   A[xy] <= b
               Aeq*[xy] = beq

                xy is the parameter vector of mixed real and binary inputs

    :param c: Column Vector, can be None
    :param A: Constraint LHS matrix, can be None
    :param b: Constraint RHS matrix, can be None
    :param equality_constraints: List of Equality constraints
    :param bin_vars: List of binary variable indices
    :param verbose: Flag for output of underlying solver, default False
    :param get_duals: Flag for returning dual variable of problem, default True (false for all mixed integer models)

    :return: A SolverOuput Object
    """
    if A is None or b is None:
        return None

    if numpy.size(A) == 0 or numpy.size(b) == 0:
        return None

    return solve_miqp_gurobi(c=c, A=A, b=b, equality_constraints=equality_constraints, bin_vars=bin_vars,
                             verbose=verbose, get_duals=get_duals)
