#include <LightGBM/c_api.h>

#include "tools.h"

struct LGBMSettings {
    int max_depth;
    float learning_rate;
    int num_trees;
    int epochs;
};

int main() {

    int SEED = 42;

    // Welcome!
    printf("Welcome! This program trains an LightGBM model for the Recommendation Engine Creation Challenge on zindi.africa (https://zindi.africa/competitions/recommendation-engine-creation-challenge).\n");

    struct LGBMSettings settings;

    printf("max_depth: ");
    scanf("%i", &settings.max_depth);
    printf("learning_rate: ");
    scanf("%f", &settings.learning_rate);
    printf("num_trees: ");
    scanf("%i", &settings.num_trees);
    printf("epochs: ");
    scanf("%i", &settings.epochs);

    // Load data!
    printf("\nData will be loaded and prepared!");

    createData();

    DatasetHandle data;
    char parametersData[100];
    sprintf(parametersData, "header=true label_column=name:Rating data_random_seed=%i", SEED);
    const char* filenameTrain = "train_preprocessed.csv";
    LGBM_DatasetCreateFromFile(filenameTrain, parametersData, NULL, &data);
    
    DatasetHandle val_data;
    const char* filenameVal = "val_preprocessed.csv";
    LGBM_DatasetCreateFromFile(filenameVal, parametersData, data, &val_data);
    
    // TRAINING
    char parametersTrain[10000];
    sprintf(parametersTrain, "objective=regression metric=rmse num_trees=%i learning_rate=%f max_depth=%i num_leaves=16 min_data_in_leaf=64 early_stopping_rounds=300 force_col_wise=true bagging_freq=10 bagging_fraction=0.5 bagging_seed=%i seed=%i",
            settings.num_trees, settings.learning_rate, settings.max_depth, SEED, SEED);

    BoosterHandle booster;
    LGBM_BoosterCreate(data, parametersTrain, &booster);
    
    LGBM_BoosterAddValidData(booster, val_data);
    
    char *results;
    results = (char *) malloc(50000000);
    for (int i = 0; i < settings.epochs; i++) {
        int f;
        LGBM_BoosterUpdateOneIter(booster, &f);
        
        int len;
        int* out_len = &len;
        LGBM_BoosterGetEvalCounts(booster, out_len);

        double *out_results;
        out_results = (double *) malloc(50000000);
        LGBM_BoosterGetEval(booster, 0, out_len, out_results);

        double *out_results_val;
        out_results_val = (double *) malloc(50000000);
        LGBM_BoosterGetEval(booster, 1, out_len, out_results_val);
     
        if (i % 100 == 0) {
            char item[1000];
            sprintf(item, "Iteration: %i; RMSE(Train): %f; RMSE(Val): %f\n", i, *out_results, *out_results_val);
            printf(item);
            strcat(results, item);
        }

        free(out_results);
        free(out_results_val);
    }

    printf("\nExperiment successful!");

    LGBM_DatasetFree(data);
    LGBM_DatasetFree(val_data);

    // PREDICTION
    char parametersTest[10000];
    sprintf(parametersTest, "objective=regression num_trees=%i learning_rate=%f max_depth=%i num_leaves=16 min_data_in_leaf=64 early_stopping_rounds=300 force_col_wise=true seed=%i",
            settings.num_trees, settings.learning_rate, settings.max_depth, SEED);
    const char* filenameTest = "test_preprocessed.csv";
    const char* filenameSubmission = "submission.csv";
    LGBM_BoosterPredictForFile(booster, filenameTest, 1, C_API_PREDICT_RAW_SCORE, 0, -1, parametersTest, filenameSubmission);
    
    LGBM_BoosterFree(booster);

    int counter = 1;
    char dir[] = "Submission/1";
    while (access(dir, F_OK) == 0) {
        counter++;
        sprintf(dir, "Submission/%d", counter);
    }

    mkdir(dir, 0700);

    // Make Submission!
    char dirSubmission[100];
    sprintf(dirSubmission, "Submission/%d/submission.csv", counter);
    FILE *fileSubmission = fopen(dirSubmission, "w");
    fprintf(fileSubmission , "%s", make_submission());
    fclose(fileSubmission);
 
    char output[10000];
    sprintf(output, "# Experiment Number %d\n## Settings:\n* max_depth: %i\n* learning_rate: %f\n* num_trees: %i\n* epochs: %i\n## Results: \n%s",
            counter, settings.max_depth, settings.learning_rate, settings.num_trees, settings.epochs, results);
    char dirOutput[100];
    sprintf(dirOutput, "Submission/%d/output.md", counter);
    FILE *fileOutput = fopen(dirOutput, "w");
    fprintf(fileOutput, "%s", output);
    fclose(fileOutput);

    remove("train_preprocessed.csv");
    remove("val_preprocessed.csv");
    remove("test_preprocessed.csv");
    remove("submission.csv");

    printf("\nsubmission.csv and output.md are saved under Submission/%d", counter);

    return 0;
}