# Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines 
https://www.drivendata.org/competitions/66/flu-shot-learning

author: kogni7

date: Winter 2022/2023

This project uses only the data sets provided by drivendata. These data sets contain information about people and if they got an H1N1 and/or flu vaccine. The task is to predict if an invidual will get a vaccine.

The file system for this project is:
FluAI (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
            * output.md
    * README.md
    * main.cpp
    * tools.hpp

This project uses a Deep Learning approach using tensorflow.

Start the program with:
```
export LD_LIBRARY_PATH=$HOME/tensorflow_root/lib:$LD_LIBRARY_PATH     
g++ main.cpp  -I$HOME/tensorflow_root/include -L$HOME/tensorflow_root/lib -o main -ltensorflow_cc -ltensorflow_framework -O3 -march=native   
./main
```

Note: Although a simple Neural Network is constructed and seeds are set, the results may not exactly reproducible.

References:
    * https://www.tensorflow.org/api_docs/cc
    * https://github.com/bennyfri/TFMacCpp
    * https://stackoverflow.com/questions/59336899/which-loss-function-and-metrics-to-use-for-multi-label-classification-with-very
    * https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a