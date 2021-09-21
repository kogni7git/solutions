# This is PumpAI.
# author: Andreas Schmitt
# date: May/July 2020

rm(list=ls())
setwd("./DrivenData/PumpAI")
set.seed(42)

VERSION = 12

library(doMC)
registerDoMC(cores=5)

# Loading the data
X <- readr::read_csv("./Data/training_features.csv")
y <- readr::read_csv("./Data/training_labels.csv")
X_test <- readr::read_csv("./Data/test_features.csv")
y_test <- readr::read_csv("./Data/SubmissionFormat.csv")

#################
# PREPROCESSING #
#################

# Remove unnecessary columns
X <- dplyr::select(X, -id, -amount_tsh, -date_recorded, -funder, -installer, -wpt_name, -num_private, -recorded_by, -payment_type,
                   -waterpoint_type_group, -scheme_name, -quantity_group, -extraction_type_class, -extraction_type_group, -source_class,
                   -source_type, -quality_group, -management_group, -subvillage, -region_code, -district_code, -lga, -ward, -scheme_management)

X_test <- dplyr::select(X_test, -id, -amount_tsh, -date_recorded, -funder, -installer, -wpt_name, -num_private, -recorded_by, -payment_type,
                        -waterpoint_type_group, -scheme_name, -quantity_group, -extraction_type_class, -extraction_type_group, -source_class,
                        -source_type, -quality_group, -management_group, -subvillage, -region_code, -district_code, -lga, -ward, -scheme_management)

# Mutate to factor
X <- dplyr::mutate_if(X, is.character, function(x) factor(x))
X_test <- dplyr::mutate_if(X_test, is.character, function(x) factor(x))
X <- dplyr::mutate_if(X, is.logical, function(x) factor(x))
X_test <- dplyr::mutate_if(X_test, is.logical, function(x) factor(x))

# Label encoding
X <- dplyr::mutate_if(X, is.factor, function(x) as.numeric(x))
X_test <- dplyr::mutate_if(X_test, is.factor, function(x) as.numeric(x))

# Convert outlier to NAs
# If gps_height is 0, replace it with NA.
X <- dplyr::mutate(X, gps_height = replace(gps_height, gps_height == 0, NA))
X_test <- dplyr::mutate(X_test, gps_height = replace(gps_height, gps_height == 0, NA))

# If longitude is 0, replace it with NA.
X <- dplyr::mutate(X, longitude = replace(longitude, longitude == 0, NA))
X_test <- dplyr::mutate(X_test, longitude = replace(longitude, longitude == 0, NA))

# If latitude is -2e-08, replace it with NA.
X <- dplyr::mutate(X, latitude = replace(latitude, latitude == -2e-08, NA))
X_test <- dplyr::mutate(X_test, latitude = replace(latitude, latitude == -2e-08, NA))

# If population is 0 or 1, replace it with NA.
X <- dplyr::mutate(X, population = replace(population, population == 0, NA))
X <- dplyr::mutate(X, population = replace(population, population == 1, NA))
X_test <- dplyr::mutate(X_test, population = replace(population, population == 0, NA))
X_test <- dplyr::mutate(X_test, population = replace(population, population == 1, NA))

# If construction_year is 0, replace it with NA.
X <- dplyr::mutate(X, construction_year = replace(construction_year, construction_year == 0, NA))
X_test <- dplyr::mutate(X_test, construction_year = replace(construction_year, construction_year == 0, NA))

# Merge features and labels
status_group <- factor(y$status_group)
rm(y)
X <- tibble::add_column(X, status_group)

# Feature engineering: New features
X["coordinate"] <- abs(X["gps_height"]$gps_height * X["longitude"]$longitude * X["latitude"]$latitude) / 1000
X["location"] <- X["basin"]$basin * X["region"]$region
X["water"] <- X["extraction_type"]$extraction_type * X["water_quality"]$water_quality * X["quantity"]$quantity * X["source"]$source * X["waterpoint_type"]$waterpoint_type

X_test["coordinate"] <- abs(X_test["gps_height"]$gps_height * X_test["longitude"]$longitude * X_test["latitude"]$latitude) / 1000
X_test["location"] <- X_test["basin"]$basin * X_test["region"]$region
X_test["water"] <- X_test["extraction_type"]$extraction_type * X_test["water_quality"]$water_quality * X_test["quantity"]$quantity * X_test["source"]$source * X_test["waterpoint_type"]$waterpoint_type

# Impute NAs
library(mice)

X_imputed <- mice(X)
X_imputed <- tibble::tibble(complete(X_imputed))

X_test_imputed <- mice(X_test)
X_test_imputed <- tibble::tibble(complete(X_test_imputed))

############
# TRAINING #
############
# C5.0
library(caret)
library(C50)
library(plyr)

trainControl <- trainControl(method = "cv", number = 10, returnResamp = "final")

c50_grid <- expand.grid(trials = c(99, 100), model = c("tree"), winnow = c(TRUE))
c50_model <- try(train(status_group ~ ., data = X_imputed, method = "C5.0", metric = "Accuracy", tuneGrid = c50_grid, trControl = trainControl))

# Variable importance
c50_importance <- varImp(c50_model, scale = FALSE)

##############
# PREDICTION #
##############

y_test["status_group"] = predict(c50_model, X_test_imputed)

# Write to CSV
write.csv(y_test, paste0("./Submission/Version ", VERSION, "/test_labels_c50.csv"), row.names = FALSE, quote = FALSE)

# Save the environment
save(list=ls(), file = paste0("./Submission/Version ", VERSION, "/PumpAI_Data.Rdata"))
rm(list=ls())
