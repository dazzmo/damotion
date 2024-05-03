<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->

<!-- PROJECT LOGO -->
<br>
<div align="center">

  <h1 align="center">Damotion</h1>

  <p align="left">
    A work-in-progress robotics tool-kit for optimisation, planning and estimation of robotics systems. Built upon the CasADi and Pinocchio C++ libraries.
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project

<p align="left">
The Damotion library is designed to provide functions and utilities for personal use in robotics-related projects. It is built around the CasADi library, utilising its symbolic algebra capabilities to compute and evaluate complicated expressions associated with robotics research areas such as control, optimisation and estimation. To further aid these computations, we utilise Pinocchio, a mature rigid-body dynamics library for use in computing these expressions. This library incorporates these two libraries to offer utilities in optimisation, control and estimation with an emphasis on efficient computations for real-time application.
</p>
<p align="right">(<a href="#readme-top">back to top</a>)
</p>

## Getting Started
<a name="getting-started"></a>

### Prerequisites

Damotion requires the following third-party libraries in order to be built and installed.
* [Eigen3](https://gitlab.com/libeigen/eigen)
* [CasADi](https://github.com/casadi/casadi)
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio)
* [Boost](https://www.boost.org/) (Version 1.70 or higher)

For testing purposes we have (this is later be a toggle-able option):
* [googletest](https://github.com/google/googletest)
* [glog](https://github.com/google/glog)

We also include interfaces to open-source solvers for numerical optimisation, we currently include:
* [Ipopt](https://github.com/coin-or/Ipopt)
* [qpOASES](https://github.com/coin-or/qpOASES) (-DWITH_QPOASES=ON in <a href="#installation">Installation</a>). Be sure to install qpOASES as a shared library, as this is what is expected by damotion.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation
<a name="installation"></a>

1. Clone the repo
   ```sh
   git clone https://github.com/dazzmo/damotion
   ```
2. Build the library
    ```sh
    cd damotion
    mkdir build && cd build
    cmake ..
    make
   ```
3. Installation of the library can then be performed by
    ```sh
    make install
    ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU LESSER GENERAL PUBLIC LICENSE License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Damian Abood - damian.abood@sydney.edu.au

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements
Thank you to Jesse Morris for his assistance with build-related concerns and improving the layout of the library.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
