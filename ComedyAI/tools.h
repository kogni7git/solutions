#include <unistd.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <stdlib.h>

void createData() {
    /* We encode the features numerically, do some feature engineering (i.e. create additive and multiplicative features) and create a validation set (~20% of the data). */

    // make list of codes    
    FILE *file;
    char row[100];
    char* Viewers_ID[50000];
    char* Joke_identifier[150];

    file = fopen("Data/train.csv", "r");
    int counter = 0;
    int counter_1 = 0;
    int counter_2 = 0;

    while (!feof(file)) {
        memset(row, '\0', 100);
        fgets(row, 100, file);

        char* Viewers_ID_item = strtok(row, ",");
        char* Joke_identifier_item = strtok(NULL, ",");
  
        if (Viewers_ID_item != NULL && Joke_identifier_item != NULL) {
            if (counter == 0) { // header
                counter++;
            } else {
                int c = 1;
                for (int i = 0; i < counter_1; i++) {
                    if (strcmp(Viewers_ID_item, Viewers_ID[i]) == 0) {   
                        c = 0;
                        break;
                    }
                }     
                if (c == 1) {
                    Viewers_ID[counter_1] = strdup(Viewers_ID_item);
                    counter_1++;
                }
                c = 1;
                for (int i = 0; i < counter_2; i++) {
                    if (strcmp(Joke_identifier_item, Joke_identifier[i]) == 0) {
                        c = 0;
                        break;
                    }
                }
                if (c == 1) {
                    Joke_identifier[counter_2] = strdup(Joke_identifier_item);
                    counter_2++;
                }
            }
        }
    }
    fclose(file);

    // Validation set
    srand(42);
    int VAL_SIZE = 150000; // ~20% of train
    int index[VAL_SIZE];
    for (int i = 0; i < VAL_SIZE; i++) {
        int r = rand() % 612702;
        index[i] = r;
    }

    // train and val set
    char *train_array;
    train_array = (char *) malloc(50000000);
    char *val_array;
    val_array = (char *) malloc(50000000);

    file = fopen("Data/train.csv", "r");
    counter = 0;

    while (!feof(file)) {
        memset(row, '\0', 100);
        fgets(row, 100, file);       
        char* Viewers_ID_item = strtok(row, ",");
        char* Joke_identifier_item = strtok(NULL, ",");
        char* Response_ID_item = strtok(NULL, ",");
        char* Rating_item = strtok(NULL, ",");

        if (Viewers_ID_item != NULL && Joke_identifier_item != NULL && Response_ID_item != NULL && Rating_item != NULL) {
            if (counter == 0) {
                char header[100];
                sprintf(header, "%s", "Viewers_ID,Joke_identifier,FE+,FE*,Rating\n");
                strcat(train_array, header);
                strcat(val_array, header);
            } else {
                int train = 1;
                for (int j = 0; j < VAL_SIZE; j++) {
                    if (counter == index[j]) {
                        train = 0;
                        break;
                    }
                }
                int i_1;
                for (int i = 0; i < 50000; i++) {
                    if (strcmp(Viewers_ID[i], Viewers_ID_item) == 0) {
                        char item[5];
                        sprintf(item, "%i,", i);
                        (train == 1) ? strcat(train_array, item) : strcat(val_array, item);
                        i_1 = i;
                        break;
                    }
                }         
                int i_2;
                for (int i = 0; i < 127; i++) {
                    if (strcmp(Joke_identifier[i], Joke_identifier_item) == 0) {
                        char item[5];
                        sprintf(item, "%i,", i);
                        (train == 1) ? strcat(train_array, item) : strcat(val_array, item);
                        i_2 = i;
                        break;
                    }
                }
                // Feature Engineering: +
                char item[5];
                sprintf(item, "%i,", i_1 + i_2);
                (train == 1) ? strcat(train_array, item) : strcat(val_array, item);
                // Feature Engineering: *
                sprintf(item, "%i,", i_1 * i_2);
                (train == 1) ? strcat(train_array, item) : strcat(val_array, item);
                // Rating
                sprintf(item, "%f\n", atof(Rating_item));
                (train == 1) ? strcat(train_array, item) : strcat(val_array, item);
            }
            counter++;
        }
    }
    fclose(file);

    FILE *train_preprocessed = fopen("train_preprocessed.csv", "w");
    fprintf(train_preprocessed , "%s", train_array);
    fclose(train_preprocessed);

    FILE *val_preprocessed = fopen("val_preprocessed.csv", "w");
    fprintf(val_preprocessed, "%s", val_array);
    fclose(val_preprocessed);

    free(train_array);
    free(val_array);

    // test set
    char *test_array;
    test_array = (char *) malloc(50000000);
   
    file = fopen("Data/test.csv", "r");
    counter = 0;

    while (!feof(file)) {
        memset(row, '\0', 100);
        fgets(row, 100, file);
        char* Viewers_ID_item = strtok(row, ",");
        char* Joke_identifier_item = strtok(NULL, ",");
        char* Response_ID_item = strtok(NULL, ",");

        if (Viewers_ID_item != NULL && Joke_identifier_item != NULL && Response_ID_item != NULL) {
            if (counter == 0) {
                char header[100];
                sprintf(header, "%s", "Viewers_ID,Joke_identifier,FE+,FE*\n");
                strcat(test_array, header);
            } else {
                int i_1;
                for (int i = 0; i < 50000; i++) {
                    if (strcmp(Viewers_ID[i], Viewers_ID_item) == 0) {
                        char item[5];
                        sprintf(item, "%i,", i);
                        strcat(test_array, item);
                        i_1 = i;
                        break;
                    }
                }   
                int i_2;
                for (int i = 0; i < 127; i++) {
                    if (strcmp(Joke_identifier[i], Joke_identifier_item) == 0) {
                        char item[5];
                        sprintf(item, "%i,", i);
                        strcat(test_array, item);
                        i_2 = i;
                        break;
                    }  
                    if (i == 126) {
                        char item[5];
                        sprintf(item, "%i,", 150); // jokes which only are in the test set
                        strcat(test_array, item);
                        i_2 = i;
                    }
                }
                // Feature Engineering: +
                char item[5];
                sprintf(item, "%i,", i_1 + i_2);
                strcat(test_array, item);
                // Feature Engineering: *
                sprintf(item, "%i\n", i_1 * i_2);
                strcat(test_array, item);
            }
            counter++;
        }
    }
    fclose(file);

    FILE *test_preprocessed = fopen("test_preprocessed.csv", "w");
    fprintf(test_preprocessed , "%s", test_array);
    fclose(test_preprocessed);

    free(test_array);
}


const char* make_submission() {
    char *submission;
    submission = (char *) malloc(50000000);
    strcat(submission, "Response_ID,Rating\n");

    FILE *file;
    char row[100];
    file = fopen("submission.csv", "r");

    FILE *fileIDs;
    char rowIDs[100];
    fileIDs = fopen("Data/SampleSubmission.csv", "r");
    
    int counter = 0;
    while (!feof(fileIDs)) {
        memset(rowIDs, '\0', 100);
        fgets(rowIDs, 100, fileIDs);
        char* Response_ID_item = strtok(rowIDs, ",");

        if (Response_ID_item != NULL) {
            if (counter == 0) {
                counter++;
            } else {
                memset(row, '\0', 100);
                fgets(row, 100, file);
                char item[100];
                sprintf(item, "%s,%s", Response_ID_item, row);
                strcat(submission, item);
            }
        }
    }
    fclose(file);

    return submission;
}