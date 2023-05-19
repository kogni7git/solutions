#!/bin/sh

# clean submission directory

dir=$(ls -t Submission | head -1)
cd Submission/$dir
mv submission/part-*.csv submission.csv
rm -rf submission