# Machine Learning for Micromagnetics
This software package implements machine learning techniques to predict coercivity in magnetic materials using both experimental and micromagnetic simulated parameters.

## Overview
The package consists of two main components:

- **Train ML Models:** Includes linear and non-linear models, such as Artificial Neural Networks (ANNs), trained on experimental and simulated micromagnetic data.

 - **Predict Coercivity:** Enables prediction of coercivity for new magnetic materials using pre-trained ML models.

## Paper Reference
For detailed information on the ML framework used in this package, please refer to the following paper:

- **How to Cite:**
``` 
  @article{PhysRevApplied.22.024046,
  title = {Accurate machine-learning predictions of coercivity in high-performance permanent magnets},
  author = {Bhandari, Churna and Nop, Gavin N. and Smith, Jonathan D.H. and Paudyal, Durga},
  journal = {Phys. Rev. Appl.},
  volume = {22},
  issue = {2},
  pages = {024046},
  numpages = {22},
  year = {2024},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevApplied.22.024046},
  url = {https://link.aps.org/doi/10.1103/PhysRevApplied.22.024046}
}
```
- **Primary Class:** cond-mat.mtrl-sci

## Prerequisites
To effectively use this software package, ensure you have the necessary dependencies and data prepared as described in the documentation.

## Directory Structure
### machine_learning_micromagnetics**

- **data**
  - mumax.csv
  - experimental.xlsx

- **docs**
  - documentation.md
  - usage_instructions.md

- **src**
  - train.py
  - predict.py

- **LICENSE**
- **README.md**

## Data
To reproduce the results presented in our paper, you can download the necessary datasets from the provided link in the Workflow section. The sample data included here can serve as a starting point, but users are encouraged to generate their own micromagnetic databases. Experimental data can be obtained from the associated paper.

## Authors
This software package was primarily developed by Churna Bhandari.

## License
The magnetic database is released under the MIT License.
