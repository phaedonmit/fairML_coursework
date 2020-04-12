# fairML_coursework
All the code is included in the file:

- cw_code.py 

You can run this using the following instructions on the Imperial lab:
- Step 1: Type in the terminal: . /vol/lab/course/557/bin/conda-setup
- Step 2: Activate the virtual environment: conda activate aif360
- Step 3: Run the script using: python3 cw_code.py

The code was developed and tested on Python version 3.7.6 using the aif360 library. It is also backwards compatible with Python 3.5 however the following libraries **need** to be available for in order to produce the required plots:

- matplotlib
- seaborn

The code is setup to run using the Adult dataset and a Logistic Regression classifier. This settings can be changed in Lines 242-243 for the compas database and an SVM classifier. By default the plotting is turned OFF. The plotting functionality can be turned on by setting Line 11 as follows:

- FLAG_PLOTS = True