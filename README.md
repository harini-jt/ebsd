# Grain Segmentation & EBSD Image Analysis

Electron Backscattered Diffraction (EBSD) Image Analysis is a 
software developed using PyQt5, for EBSD Analysis.

This software is developed for Scientific Purpose.

EBSD package developed with utility functions for each process of ebsd such as grain segmentation and generating kernel average misorientaion, dislocation density, grain reference reference orientation deviation, inverse pole figure, pole figure maps.



## Features
- Cross platform
- Various Segmentation Methods(KAM Based, Watershed)
- Interactive Grain Selection plots
- Inverse Pole Figure and Pole Figure Maps
- Grain Reference Orientaion Deviation, Dislocation Density
- Exportable csv file for Grain Details 




  
## Lessons Learned

I learned to device various algorithms 
for solving problems in generating EBSD maps 
along side learning to work with a completely
new platform, PyQt5 from scratch.

Also learned Crystallography and Mathematics behind it.


  
## Optimizations

- Speed up many time consuming algorithms `Just-In-Time (JIT)` compiler offered by [NUMBA](http://numba.pydata.org/) was used

- Also GPU programming with [CUDA](https://numba.readthedocs.io/en/stable/cuda/index.html) package in NUMBA using Python to improve faster execution of algorithms in large image data sets.



  
## Run Locally

Clone the project

```bash
  git clone https://github.com/harini-ht/ebsd
```

Go to the project directory

To run the exe file of the app:

- The **dist** folder contains `EBSD Analysis.exe` file which is a standalone executable file.

To run the app in a Python IDE/Code Editor:
```bash
  python -u main.py
```
  
## Usage/Examples

```python
from ebsd import *
import matplotlib.pyplot as plt
fname = 'Data.ctf'
Eulers, data10, cols, rows, xstep, ystep, title = read_data(fname)
image = rgb_img(Eulers)
plt.imshow(image)
```

  
## Screenshots

![App Screenshot](https://user-images.githubusercontent.com/74011816/134781158-9de4ccaf-3e16-4e39-9958-8254e8598a44.png)

  
## Acknowledgements

 - [Dr. Anish Kumar](https://sites.google.com/site/vanianish/)
 - Dr. Vani Shankar
 
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/harini-ht)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harini-t-55b2821bb)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/harini_jt)
