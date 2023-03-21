# A simple Shiny App with the iris and the Puromycin dataset

author: kogni7

date: February/March 2023

This a simple Shiny App. It uses the iris and Puromycin datasets delivered with R.

The App trains a Machine Learning model on 80% of the dataset and predicts on the other 20%.

There are five Machine Learning techniques to choose from. The training data can be resampled.

To run the shiny app, open an R-session in the parent directory of SimpleShinyApp and type:
```
library(shiny)
runApp('SimpleShinyApp', launch.browser = TRUE)
```

References:

- <https://w3schools.com>
