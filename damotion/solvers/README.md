<h1>The Solver Component</h1>
This component of the library is associated with creating mathematical programs of the form

<h2> Decision Variables </h2>

Decision variables are the variables solved for within a program. Each variable is represented by a `damotion::symbolic::Variable` object, which has a unique ID. These variables can be grouped together using standard `Eigen` containers, which is useful for defining decision variable vectors and matrices.

<h2> Parameters </h2>

Parameters can be added to a program such that programs can be solved whilst altering the variables $p$ that may not need to be solved for but are required to define elements of the program.

<h2> Constraints </h2>
<h2> Costs </h2>

<h2> Bindings </h2>

Each constraint and cost above is created as a symbolic expression with decision variables $x$, parameters $p$ and produces a known output based on the expression. These expressions have no understanding of the program until they are bound to the program variables and parameters. This has the additional benefit of only creating one constraint of one type, and reusing the constraint on different sets of variables/parameters such as integration-based constraints.

<h2> Program </h2>
A `Program` class instance is responsible for building the mathematical program by adding all previously mentioned components together, including decision variables, parameters, costs and constraints.

Bindings are performed implictly when adding a cost or constraint to a `Program` class instance, which will be of the form `program.Add<Cost/Constraint>(std::shared_ptr<Cost/Constraint>, {variables}, {parameters})`. A binding is then created that associates this cost/constraint entity in the program with those variables.