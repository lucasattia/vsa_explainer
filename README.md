<p align="center">
  <img src="assets/logo.png" alt="vsa_explainer Logo" width="200"/>
</p>
<h3 align="center">
vsa_explainer: A simple Python package to visualize and explain RDKit SlogP_VSA, SMR_VSA, PEOE_VSA, EState_VSA, VSA_EState descriptor and atomic contributions
</h3>
<br/>

[![PyPI](https://img.shields.io/pypi/v/vsa_explainer.svg)](https://pypi.org/project/vsa_explainer/)
[![Python](https://img.shields.io/pypi/pyversions/vsa_explainer.svg)](https://pypi.org/project/vsa_explainer/)
[![Python Tests](https://github.com/srijitseal/vsa_explainer/actions/workflows/ci.yml/badge.svg)](https://github.com/srijitseal/vsa_explainer/actions/workflows/ci.yml)
[![Repo Size](https://img.shields.io/github/repo-size/srijitseal/vsa_explainer.svg)](https://github.com/srijitseal/vsa_explainer)

---

## 📌 Installation
```bash
pip install vsa_explainer
```

## 📌 Quick Usage
```python
from vsa_explainer import visualize_vsa_contributions

# Highlight per-atom contributions to SMR_VSA7 and EState_VSA5
smiles = "C1CO[C@@H]1CN2C3=C(C=CC(=C3)C(=O)O)N=C2CN4CCC(CC4)C5=NC(=CC=C5)OCC6=C(C=C(C=C6)C#N)F"
visualize_vsa_contributions(smiles, ["SMR_VSA7", "EState_VSA5"])
```

<p align="center">
  <img src="assets/output_one.png" alt="vsa_explainer output 1" width="500"/>
</p>
<p align="center">
  <img src="assets/output_two.png" alt="vsa_explainer output 2" width="500"/>
</p>

- Draws an SVG of your molecule with atoms colored by their contribution to each selected VSA descriptor.
- Displays a table reporting per-atom values, contributions, and percentage of the total.

Grid view can enable the direct comparison of multiple molecules and multiple features simultaneously. 
<p align="center">
  <img src="assets/grid.PNG" alt="grid view" width="500"/>
</p>


## 📌 Support

- **SMR_VSA**  
  MOE-type descriptors using MR contributions and surface area contributions

- **SlogP_VSA**  
  MOE-type descriptors using LogP contributions and surface area contributions

- **PEOE_VSA**  
  MOE-type descriptors using partial charges and surface area contributions

- **EState_VSA**  
  MOE-type descriptors using EState indices and surface area contributions (developed at RD, not described in the CCG paper)

- **VSA_EState**  
  MOE-type descriptors using EState indices and surface area contributions (developed at RD, not described in the CCG paper)


## 📌 Contributing
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add feature"`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📌 License
Released under the MIT License. See LICENSE for details.

✨ Enjoy exploring molecular surface areas with vsa_explainer!
