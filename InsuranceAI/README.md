# Insurance Prediction Challenge
https://zindi.africa/competitions/insurance-prediction-challenge

author: kogni7

date: 2022/23

This project uses only the data sets provided by ZINDI. These data sets contain information about buildings. These are the only used features in this project. The task is to predict if a building will have an insurance claim.

The file system for this project is:
* InsuranceAI (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
            * output.md
    * src/main/java/org.InsuranceAI
        * Main
        * Modelling
        * Preparing
    * pom.xml
    * Dockerfile
    * README.md

This project uses SMILE.

Start the program with:
```
docker build -t insurance_ai .
docker run -it --rm --name insurance_ai -v $PWD/Submission:/Submission insurance_ai
```