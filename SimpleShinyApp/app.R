#### LIBRARIES ####
library(datasets)
library(e1071) # Bayes, SVM
library(class) # knn
library(party) # Tree
library(randomForest) # RandomForest
library(ggplot2)
library(ggpubr)

#### SEED ####
SEED = 42
BUTTON <- 0

#### PREPARATION ####
colnames(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Class")
colnames(Puromycin) <- c("Concentration", "Rate", "Class")

prepare_data <- function(data) {
  set.seed(SEED)
  X_ind <- sample(1:nrow(data), 0.8*nrow(data), replace=FALSE)
  X_test_ind <- setdiff(c(1:nrow(data)), X_ind)
  return(list("X" = data[X_ind,], "X_test" = data[X_test_ind,]))
}

prediction <- function(data, classifier) {
  predictions_X <- predict(classifier, newdata=data$X)
  data$X$Classification <- data$X$Class == predictions_X
  data$X$Classification[data$X$Classification == "TRUE"] = "correct"
  data$X$Classification[data$X$Classification == "FALSE"] = "incorrect"
  accuracies_X <- accuracy(predictions_X, data$X$Class)
  
  predictions <- predict(classifier, newdata=data$X_test)
  data$X_test$Classification <- data$X_test$Class == predictions
  data$X_test$Classification[data$X_test$Classification == "TRUE"] = "correct"
  data$X_test$Classification[data$X_test$Classification == "FALSE"] = "incorrect"
  accuracies <- accuracy(predictions, data$X_test$Class)

  return(list("data"=data, "accuracies_X"=accuracies_X, "accuracies"=accuracies))
}

accuracy <- function(predictions, truth) {
  return(sum(as.numeric(predictions == truth))/length(truth))
}

customized_theme <- theme(panel.background = element_rect(fill="#ffffcc", colour="#ffffcc", size=0.5, linetype="solid"),
                          panel.grid.major = element_blank(), 
                          panel.grid.minor = element_blank(),
                          axis.line = element_line(size=1, colour="black", linetype=1, arrow=arrow(angle=15, length=unit(.30, "cm"))),
                          axis.ticks = element_line(color="black"),
                          axis.ticks.length = unit(5, units="pt"),
                          axis.text = element_text(size=14, family="Times"),
                          axis.title = element_text(size=18, family="Times"),
                          legend.background = element_rect(fill="#ffc966"),
                          legend.key = element_rect(fill="#ffc966", colour="#ffc966"),
                          legend.text = element_text(size = 18, family="Times"),
                          legend.title = element_text(size = 20, family="Times"),
                          plot.background = element_rect(fill = "#ffffcc"),
                          plot.title = element_text(size = 22, family="Times"))

plot_iris <- function(model) {
  plot_train_Sepal <- ggplot(model$data$X, aes(Sepal.Length, Sepal.Width)) + xlab("Sepal Length") + ylab("Sepal Width") + ggtitle("iris train Sepal") + customized_theme
  plot_train_Sepal <- plot_train_Sepal + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))

  plot_test_Sepal <- ggplot(model$data$X_test, aes(Sepal.Length, Sepal.Width)) + xlab("Sepal Length") + ylab("Sepal Width") + ggtitle("iris test Sepal") + customized_theme
  plot_test_Sepal <- plot_test_Sepal + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))

  plot_train_Petal <- ggplot(model$data$X, aes(Petal.Length, Petal.Width)) + xlab("Petal Length") + ylab("Petal Width") + ggtitle("iris train Petal") + customized_theme
  plot_train_Petal <- plot_train_Petal + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))

  plot_test_Petal <- ggplot(model$data$X_test, aes(Petal.Length, Petal.Width)) + xlab("Petal Length") + ylab("Petal Width") + ggtitle("iris test Petal") + customized_theme
  plot_test_Petal <- plot_test_Petal + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))

  list(plot_train_Sepal, plot_test_Sepal, plot_train_Petal, plot_test_Petal)
}
  
plot_Puromycin <- function(model) {
  plot_train <- ggplot(model$data$X, aes(Concentration, Rate)) + ggtitle("Puromycin train") + customized_theme
  plot_train <- plot_train + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))

  plot_test <- ggplot(model$data$X_test, aes(Concentration, Rate)) + ggtitle("Puromycin test") + customized_theme
  plot_test <- plot_test + geom_point(aes(shape=Class, colour=Classification), size=4) + scale_colour_manual(values = c("correct" = "#008000", "incorrect" = "#e60000"))
  
  list(plot_train, plot_test)
}

plot_models <- function(plots_iris, plots_Puromycin) { ggarrange(plots_iris[[1]], plots_iris[[2]], plots_iris[[3]], plots_iris[[4]], plots_Puromycin[[1]], plots_Puromycin[[2]], ncol=2, nrow = 3, common.legend = FALSE, legend = "right") }

plot_accuracy <- function(model_iris, model_Puromycin) {
  df_iris <- data.frame(x = c("train", "test"), acc = c(model_iris$accuracies_X, model_iris$accuracies))
  df_Puromycin <- data.frame(x = c("train", "test"), acc = c(model_Puromycin$accuracies_X, model_Puromycin$accuracies))

  customized_theme <- customized_theme + theme(axis.line = element_line(size=1, colour="black", linetype=1), axis.title = element_text(size=22, family="Times"))

  plot_i <- ggplot(df_iris, aes(x = x, y = acc)) + ylab("Accuracy") + ggtitle("iris") + customized_theme + xlab("")
  plot_i <- plot_i + geom_bar(stat = "identity", fill = "#00e6e6") + xlim("train", "test")
  plot_i <- plot_i + scale_y_continuous(labels = scales::percent, expand = c(0,0), limits = c(0,1))

  plot_P <- ggplot(df_Puromycin, aes(x = x, y = acc)) + ylab("Accuracy") + ggtitle("Puromycin") + customized_theme + xlab("") + ylab("")
  plot_P <- plot_P + geom_bar(stat = "identity", fill = "#e600e6") + xlim("train", "test")
  plot_P <- plot_P + scale_y_continuous(labels = scales::percent, expand = c(0,0), limits = c(0,1))

  ggarrange(plot_i, plot_P, ncol = 2, nrow = 1)
}

#### SERVER ####
server <- function(input, output) {

  #### I BAYES ####
  bayes_iris <- reactive({
    SEED <<- SEED + input$button[1] - BUTTON
    BUTTON <<- input$button[1]

    data <- prepare_data(iris)
    classifier <- naiveBayes(Class ~ ., data=data$X)
    prediction(data, classifier)
  })

  bayes_Puromycin <- reactive({
    activated <- input$button[1]

    data <- prepare_data(Puromycin)
    classifier <- naiveBayes(Class ~ ., data=data$X)
    prediction(data, classifier)
    
  })
  
  #### II KNN ####
  neighbours_iris <- reactive({
    SEED <<- SEED + input$button[1] - BUTTON
    BUTTON <<- input$button[1]

    data <- prepare_data(iris)
    predictions_X <- knn(train=data$X[,1:length(data$X)-1], test=data$X[,1:length(data$X)-1], cl=data$X$Class, k=input$k)
    predictions <- knn(train=data$X[,1:length(data$X)-1], test=data$X_test[,1:length(data$X_test)-1], cl=data$X$Class, k=input$k)

    data$X$Classification <- data$X$Class == predictions_X
    data$X$Classification[data$X$Classification == "TRUE"] = "correct"
    data$X$Classification[data$X$Classification == "FALSE"] = "incorrect"
    accuracies_X <- accuracy(predictions_X, data$X$Class)
      
    data$X_test$Classification <- data$X_test$Class == predictions
    data$X_test$Classification[data$X_test$Classification == "TRUE"] = "correct"
    data$X_test$Classification[data$X_test$Classification == "FALSE"] = "incorrect"
    accuracies <- accuracy(predictions, data$X_test$Class)
      
    return(list("data"=data, "accuracies_X"=accuracies_X, "accuracies"=accuracies))
  })
  
  neighbours_Puromycin <- reactive({
    activated <- input$button[1]
  
    data <- prepare_data(Puromycin)
    predictions_X <- knn(train=data$X[,1:length(data$X)-1], test=data$X[,1:length(data$X)-1], cl=data$X$Class, k=input$k)
    predictions <- knn(train=data$X[,1:length(data$X)-1], test=data$X_test[,1:length(data$X_test)-1], cl=data$X$Class, k=input$k)
    
    data$X$Classification <- data$X$Class == predictions_X
    data$X$Classification[data$X$Classification == "TRUE"] = "correct"
    data$X$Classification[data$X$Classification == "FALSE"] = "incorrect"
    accuracies_X <- accuracy(predictions_X, data$X$Class)
    
    data$X_test$Classification <- data$X_test$Class == predictions
    data$X_test$Classification[data$X_test$Classification == "TRUE"] = "correct"
    data$X_test$Classification[data$X_test$Classification == "FALSE"] = "incorrect"
    accuracies <- accuracy(predictions, data$X_test$Class)
    return(list("data"=data, "accuracies_X"=accuracies_X, "accuracies"=accuracies))
  })

  #### III Tree ####
  tree_iris <- reactive({
    SEED <<- SEED + input$button[1] - BUTTON
    BUTTON <<- input$button[1]

    data <- prepare_data(iris)
    classifier <- ctree(Class ~ ., data=data$X, controls=ctree_control(maxdepth=input$maxdepth))
    prediction(data, classifier)
  })
  
  tree_Puromycin <- reactive({
    activated <- input$button[1]
    
    data <- prepare_data(Puromycin)
    classifier <- ctree(Class ~ ., data=data$X, controls=ctree_control(maxdepth=input$maxdepth))
    prediction(data, classifier)
  })

  #### IV FOREST ####
  forest_iris <- reactive({    
    SEED <<- SEED + input$button[1] - BUTTON
    BUTTON <<- input$button[1]
  
    data <- prepare_data(iris)
    classifier <- randomForest(Class ~ ., data=data$X, ntree=input$ntree)
    prediction(data, classifier)
  })

  forest_Puromycin <- reactive({
    activated <- input$button[1]
    
    data <- prepare_data(Puromycin)
    classifier <- randomForest(Class ~ ., data=data$X, ntree=input$ntree)
    prediction(data, classifier)
  })

  #### V SVM ####
  support_iris <- reactive({
    SEED <<- SEED + input$button[1] - BUTTON
    BUTTON <<- input$button[1]
    
    data <- prepare_data(iris)
    classifier <- svm(Class ~ ., data=data$X, scale=input$scale, type=input$typ, kernel=input$kernel, cost=input$cost, nu=input$nu)
    prediction(data, classifier)
  })
  
  support_Puromycin <- reactive({
    activated <- input$button[1]
    
    data <- prepare_data(Puromycin)
    classifier <- svm(Class ~ ., data=data$X, scale=input$scale, type=input$typ, kernel=input$kernel, cost=input$cost, nu=input$nu)
    prediction(data, classifier)
  })

  #### TEXT OUTPUT ####
  output$text_algorithm <- renderPrint({
    if (input$algorithm == "bayes") { cat("Naive Bayes Classifier") }
    else if (input$algorithm == "knn") { cat("k-Nearest Neighbour Classifier") }
    else if (input$algorithm == "tree") { cat("Tree") }
    else if (input$algorithm == "forest") { cat("Random Forest") }
    else if (input$algorithm == "svm") { cat("Support Vector Machine") }
  })
  output$text_iris_train <- renderPrint({
    if (input$algorithm == "bayes") { cat(format(bayes_iris()$accuracies_X * 100, digits=4), "%") }
    else if (input$algorithm == "knn") { cat(format(neighbours_iris()$accuracies_X * 100, digits=4), "%") }
    else if (input$algorithm == "tree") { cat(format(tree_iris()$accuracies_X * 100, digits=4), "%") }
    else if (input$algorithm == "forest") { cat(format(forest_iris()$accuracies_X * 100, digits=4), "%") }
    else if (input$algorithm == "svm") { cat(format(support_iris()$accuracies_X * 100, digits=4), "%") }
  })
  output$text_iris_test <- renderPrint({
    if (input$algorithm == "bayes") { cat(format(bayes_iris()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "knn") { cat(format(neighbours_iris()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "tree") { cat(format(tree_iris()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "forest") { cat(format(forest_iris()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "svm") { cat(format(support_iris()$accuracies * 100, digits=4), "%") }
  })
  output$text_Puromycin_train <- renderPrint({
      if (input$algorithm == "bayes") { cat(format(bayes_Puromycin()$accuracies_X * 100, digits=4), "%") }
      else if (input$algorithm == "knn") { cat(format(neighbours_Puromycin()$accuracies_X * 100, digits=4), "%") }
      else if (input$algorithm == "tree") { cat(format(tree_Puromycin()$accuracies_X * 100, digits=4), "%") }
      else if (input$algorithm == "forest") { cat(format(forest_Puromycin()$accuracies_X * 100, digits=4), "%") }
      else if (input$algorithm == "svm") { cat(format(support_Puromycin()$accuracies_X * 100, digits=4), "%") }
  })
  output$text_Puromycin_test <- renderPrint({
    if (input$algorithm == "bayes") { cat(format(bayes_Puromycin()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "knn") { cat(format(neighbours_Puromycin()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "tree") { cat(format(tree_Puromycin()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "forest") { cat(format(forest_Puromycin()$accuracies * 100, digits=4), "%") }
    else if (input$algorithm == "svm") { cat(format(support_Puromycin()$accuracies * 100, digits=4), "%") }
  })
  
  #### PLOT OUTPUT ####
  output$plot <- renderPlot({
    if (input$algorithm == "bayes") { plot_models(plot_iris(bayes_iris()), plot_Puromycin(bayes_Puromycin())) }
    else if (input$algorithm == "knn") { plot_models(plot_iris(neighbours_iris()), plot_Puromycin(neighbours_Puromycin())) }
    else if (input$algorithm == "tree") { plot_models(plot_iris(tree_iris()), plot_Puromycin(tree_Puromycin())) }
    else if (input$algorithm == "forest") { plot_models(plot_iris(forest_iris()), plot_Puromycin(forest_Puromycin())) }
    else if (input$algorithm == "svm") { plot_models(plot_iris(support_iris()), plot_Puromycin(support_Puromycin())) }
  })
  
  output$plot_accuracy <- renderPlot({
    if (input$algorithm == "bayes") { plot_accuracy(bayes_iris(), bayes_Puromycin()) }
    else if (input$algorithm == "knn") { plot_accuracy(neighbours_iris(), neighbours_Puromycin()) }
    else if (input$algorithm == "tree") { plot_accuracy(tree_iris(), tree_Puromycin()) }
    else if (input$algorithm == "forest") { plot_accuracy(forest_iris(), forest_Puromycin()) }
    else if (input$algorithm == "svm") { plot_accuracy(support_iris(), support_Puromycin()) }
  })
}

shinyApp(ui = htmlTemplate("www/index.html"), server)