# Tanzania Tourism Prediction by Pycon Tanzania Community
https://zindi.africa/competitions/tanzania-tourism-prediction

author: kogni7

date: 2022/23

This project uses only the data sets provided by ZINDI. These data sets contain information about tourist expenditures. These are the only used features in this project. The task is to predict the tourist expenditure.

The file system for this project is:
* TourismAI (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
            * output.md
    * src
        * main.rs
        * modelling.rs
        * preparing.rs
    * Cargo.toml
    * Cargo.lock
    * Dockerfile
    * README.md

This project uses linfa.

Start the program with:
```
docker build -t tourism_ai .
docker run -it --rm --name tourism_ai -v $PWD/Submission:/Submission tourism_ai
```