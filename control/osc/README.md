<h1> The Operational Space Control Module </h1>
This module concerns itself with task-space control of rigid body systems experiencing contact in the environment.

The basic controller is designed as a wrapper around a opt::Program such that it provides catered functions that allow the addition of motion tasks and contact points which are then translated into their appropriate constraints and objectives. This also handles creation of program parameters related to these tracking tasks.

If greater functionality is required, such as adding non-conventional costs/constraints or parameters, the opt::Program class can be directly accessed and modified to provide this desired functionality. If this is still to restrictive then it is encouraged to make a custom program from scratch using the solver module.