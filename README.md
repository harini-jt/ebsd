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
- Grain Reference Orientation Deviation, Dislocation Density
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

- Install dependencies
```bash
   pip install -r requirements.txt  
```
- Run the main file
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

![App Screenshot1](https://user-images.githubusercontent.com/74011816/134781158-9de4ccaf-3e16-4e39-9958-8254e8598a44.png)
### Toolbar
![image2](https://user-images.githubusercontent.com/74011816/143666127-91eb124c-ce86-4c3b-aad4-0f7048a2dae5.png)
![image3](https://user-images.githubusercontent.com/74011816/143666146-c6774d96-8e56-47bd-bed7-003e174a1925.png)
![image4](https://user-images.githubusercontent.com/74011816/143666368-d20678ef-0aab-4d40-8631-90e8dcf44faf.png)

  
## Acknowledgements

 - [Dr. Anish Kumar](https://sites.google.com/site/vanianish/)
 - Dr. Vani Shankar
 
