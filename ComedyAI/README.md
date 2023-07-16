# Recommendation Engine Creation Challenge

author: kogni7

date: Winter 2022/2023

This project uses only the data sets provided by ZINDI. These data sets contain information about ratings of comedy events. The task is to predict the rating of other comedy events.

The file system for this project is:
ComedyAI (root)
   * Data
   * Submission
      * 1 - x: Submission directories named by the version number
         * submission.csv
         * output.md
   * README.md
   * main.c
   * tools.h

This project uses LightGBM.

Start the program with:
```
export LD_LIBRARY_PATH=$HOME/LightGBM:$LD_LIBRARY_PATH    
gcc main.c -L$HOME/LightGBM/ -I$HOME/LightGBM/include -o main -l_lightgbm -O3 -mtune=native
./main
```

Note: The preprocessing of the data may take a while. Furthermore, the results might not be exactly reproducible, although seeds are set.

Reference:
* https://stdin.top/posts/csv-in-c/
* https://lightgbm.readthedocs.io/en/v3.3.5/C-API.html
