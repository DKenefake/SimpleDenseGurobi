# SimpleDenseGurobi
This is a small library to simplify solving dense optimization problems with [Gurobi](https://www.gurobi.com/) in Python. This has been designed to interface  with Gurobi such that there is minimal overhead with interacting with the solver. This is not ment for solving large sparse optimization problems, but to solve many small optimization problems quickley. 

SimpleDenseGurobi considers problems of the following form.

<img src="https://github.com/DKenefake/SimpleDenseGurobi/blob/main/problem%20definition.svg">

This interface takes numpy arrays for the problem definitions, and iterables for equality constraint indices and binary variable indices. Any input parameter is nullable. Matrices c, and b should be column vectors.

```python 
sol = solve_lp(c, A, b) #solve LP
sol = solve_lp(None, A, b) #testing feasibility
sol = solve_qp(Q, c, A, b) #solve QP
sol = solve_qp(Q, c, None, None) #unconstrained QP
sol = solve_miqp(Q, c, A, b, equality_constraints, binary_variables) #Constrained MIQP with equality constraints
```

The solve_XXX interface returns a SolutionOutput object if optimal else None. The SolutionOuput object is generally defined in the following manner. 

```python
@dataclass
class SolverOutput:
    obj: float
    sol: numpy.ndarray

    slack: Optional[numpy.ndarray]
    active_set: Optional[numpy.ndarray]
    dual: Optional[numpy.ndarray]
```

Support for MIQCQP and QCQP problem types are forthcoming.

