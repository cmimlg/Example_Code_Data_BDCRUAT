# Example Code and Data for Big Data Classification and Regression Using Augmented Trees
This directory contains the code and data files needed to illustrate regression using augmented trees on a big dataset. The datafile for this example is "jan_feb_pp_data.csv". This is data associated with the airline delay regression task. It is recomended that you create a directory on your computer to run these examples or clone this repository.

---
# Python Package Pre-requisites
To run the code in this directory you will need to install the following packages:
1. The sklearn package
2. The pandas package
3. The GPy package
4. The matplotlib package

---
# Running the example
1. In the read_data() method edit the directory location to reflect the directory from which you will be running this example. This directory should contain the data file "jan_feb_pp_data.csv".
2. Get to your python interactive console and perform the following:
3. `import DTS_AD`
4. ` from DTS_AD import *`
5. The next line executes the tree based segmentation algorithm with a leaf size of 1000
6. `tc = run_algorithms(1000, False)`
7. On a laptop with 16 GB of RAM, the algorithm took about 90 minutes to complete. One of the algorithms considered is GP regression. The solution for GP regression involves computing a matrix inverse - an expensive computation. Your mileage may vary. Note the accuracy of the test set is part of the output. You will observe a lot of warning messages about a deprecated method from the GPy module. 
8. To see how gradient boosted trees performs on this dataset, run the following:
9. `fit_xgb(True)`
10. You should see the accuracy as part of the logger output.
11. Similarly, to see how random forests (takes a longer time than the gbt example) performs on this dataset, run the following:
12. `fit_rf(True)`

