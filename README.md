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
<details>
  <summary>Table of Contents</summary>
  <ol>
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
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<p align="left">
The Damotion library is built on a CasADi foundation, utilising the symbolic algebra capabilities to compute
</p>
<p align="right">(<a href="#readme-top">back to top</a>)
</p>

## Getting Started
<a name="getting-started"></a>

### Prerequisites

Damotion requires the following third-party libraries in order to be built and installed.
* [Eigen3]()
* [CasADi](https://github.com/casadi/casadi)
* [Pinocchio]()

For numeric solvers, we currently have:
* [Ipopt]()
* [QPOASES]()

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

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/dazzmo/robotics-toolbox/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the GNU LESSER GENERAL PUBLIC LICENSE License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Damian Abood - TBA@TBA.COM

<p align="right">(<a href="#readme-top">back to top</a>)</p>