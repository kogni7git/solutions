# Richter's Predictor: Modeling Earthquake Damage
https://www.drivendata.org/competitions/57/nepal-earthquake/

author: kogni7

date: 2022/23

This project uses only the data sets provided by drivendata. These data sets contain information about buildings. These are the only used features in this project. The task is to predict the level of damage caused by an earthquake.

The file system for this project is:
* EarthquakeAI (root)
    * Data
    * Submission
        * 1 - x: Submission directories named by the version number
            * submission.csv
            * output.md
    * src/main/kotlin
        * Main.kt
        * Modelling.kt
        * Preparing.kt
    * pom.xml
    * Dockerfile
    * README.md

This project uses deeplearning4j and SMILE.

Start the program with:
```
docker build -t earthquake_ai .
docker run -it --rm --name earthquake_ai -v $PWD/Submission:/Submission earthquake_ai
```

References:
* https://github.com/Kotlin/kotlin-jupyter/blob/master/samples/DeepLearning4j.ipynb