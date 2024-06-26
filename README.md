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

<!-- TABLE OF CONTENTS -->
<navigation>
  <summary><b>Table of Contents</b></summary>
  <ul>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ul>
</navigation>



<!-- ABOUT THE PROJECT -->
## About The Project

<p align="left">
The Damotion library was designed for personal use in robotics projects. It is built around the CasADi library, utilising the symbolic algebra capabilities to compute and evaluate complicated expressions associated with robotics research areas such as control, optimisation and estimation. To further aid these computations, we utilises Pinocchio, a mature rigid-body dynamics library for use in computing these expressions. This library combines these two libraries to offer utilities in optimisation, control and estimation with an emphasis on efficient computations for real-time application.
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
* [QPOASES](https://github.com/coin-or/qpOASES)

> NOTE: qpOASES __must__ be built as a shared library. This may not be enabled by default, so check the options in their CMakeLists.txt!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

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

Distributed under the GNU GENERAL PUBLIC LICENSE V3.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Development

We use [pre-commit](https://pre-commit.com/) to standardise code formatting. Please install prior to development.

<!-- CONTACT -->
## Contact
Damian Abood - damian.abood@sydney.edu.au

<p align="right">(<a href="#readme-top">back to top</a>)</p>
