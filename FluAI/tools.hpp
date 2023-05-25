#include <filesystem>
#include <fstream>

#define TRAINING_SAMPLES 21366
// VALIDATION_SAMPLES is 20% of training samples
#define VALIDATION_SAMPLES 5341   
#define TRAINING_PLUS_VALIDATION_SAMPLES 26707
#define TEST_SAMPLES 26708

using namespace std;


double mean(int arr[], int N) {
    return accumulate(arr, arr + N, 0.0) / N;
}

double mean(double arr[], int N) {
    return accumulate(arr, arr + N, 0.0) / N;
}

// standard deviation
double sd(int arr[], int N, double m) {
    double s = 0.0;
    for (int i = 0; i < N; i++)
        s += pow((arr[i] - m), 2);
    return sqrt(s/N);
}


class trainingAndTestData {
    private:
        string workingTrainingFeatures[TRAINING_PLUS_VALIDATION_SAMPLES+1];
        string workingTrainingLabels[TRAINING_PLUS_VALIDATION_SAMPLES+1];
        string workingTestFeatures[TEST_SAMPLES+1];
        
        int (*trainingFeaturesUnscaled)[8] = new int[TRAINING_SAMPLES][8];
        int (*validationFeaturesUnscaled)[8] = new int[VALIDATION_SAMPLES][8];
        int (*testFeaturesUnscaled)[8] = new int[TEST_SAMPLES][8];
  
        list<string> age_group;
        list<string> education;
        list<string> race;
        list<string> sex;
        list<string> income_poverty;
        list<string> marital_status;
        list<string> rent_or_own;
        list<string> employment_status;
        list<string> hhs_geo_region;
        list<string> census_msa;
        list<string> employment_industry;
        list<string> employment_occupation;

    public:
        double (*trainingFeatures)[8] = new double[TRAINING_SAMPLES][8];
        double (*validationFeatures)[8] = new double[VALIDATION_SAMPLES][8];
        double (*testFeatures)[8] = new double[TEST_SAMPLES][8];
        int (*trainingLabels)[3] = new int[TRAINING_SAMPLES][3];
        int (*validationLabels)[3] = new int[VALIDATION_SAMPLES][3];

        void clean(string file, int samples) {
            // remove unnecessary , (comma)
            fstream fin;
            fin.open(file, ios::in);
    
            string l;
            int counter = 0;
            string rows[samples+1];

            while(counter != samples+1) {
                getline(fin, l);
  
                stringstream s(l);
                string word;
                int col = 0;

                string r = "";
                string sign = ",";

                while(getline(s, word, ',')) {
                    if (word[0] == '"') sign = "";
                    else if (word.back() == '"') sign = ",";  // , (comma) necessary for CSV
                    r += word + sign;            
                    col++;
                }
                rows[counter] = r;
                counter++;
            }

            fstream fout;
            fout.open("data.csv", ios::out);  
            for (int i = 0; i < samples+1; i++)
                fout << rows[i] << "\n";
        }

        void makeCodes() {
            // Make codes of categorical columns! 
            fstream fin;
            fin.open("data.csv", ios::in);

            vector<string> row;
            string l;
            int counter = 0;
  
            while(counter != TRAINING_PLUS_VALIDATION_SAMPLES+1) {
                getline(fin, l);

                if (counter != 0) {
                    stringstream s(l);
                    string word;
                    int col = 0;

                    while(getline(s, word, ',')) {
                        if (word != "") {
                            if (col == 22) this->age_group.push_back(word);
                            else if (col == 23) this->education.push_back(word);
                            else if (col == 24) this->race.push_back(word);                 
                            else if (col == 25) this->sex.push_back(word);
                            else if (col == 26) this->income_poverty.push_back(word);
                            else if (col == 27) this->marital_status.push_back(word);
                            else if (col == 28) this->rent_or_own.push_back(word);
                            else if (col == 29) this->employment_status.push_back(word);
                            else if (col == 30) this->hhs_geo_region.push_back(word);
                            else if (col == 31) this->census_msa.push_back(word);
                            else if (col == 34) this->employment_industry.push_back(word);
                            else if (col == 35) this->employment_occupation.push_back(word);
                        }
                        col++;
                    }
                }
                counter++;
            }

            this->age_group.sort();
            this->age_group.sort();
            this->education.sort();
            this->race.sort();
            this->sex.sort();
            this->income_poverty.sort();
            this->marital_status.sort();
            this->rent_or_own.sort();
            this->employment_status.sort();
            this->hhs_geo_region.sort();
            this->census_msa.sort();
            this->employment_industry.sort();
            this->employment_occupation.sort();

            this->age_group.unique();
            this->education.unique();
            this->race.unique();
            this->sex.unique();
            this->income_poverty.unique();
            this->marital_status.unique();
            this->rent_or_own.unique();
            this->employment_status.unique();
            this->hhs_geo_region.unique();
            this->census_msa.unique();
            this->employment_industry.unique();
            this->employment_occupation.unique();

        }

        void code(int samples, int train_test) {
            // Code categorical columns!
            fstream fin;
            fin.open("data.csv", ios::in);

            string l;
            int counter = 0;

            while(counter != samples+1) {
                getline(fin, l);

                stringstream s(l);
                string word;
                int col = 0;
                string r = "";

                while(getline(s, word, ',')) {
                    if (counter == 0) r += word + ","; // header
                    else {
                        if (word != "") {
                            if (col == 0) {
                                r += word + ",";
                            } else if (col == 22) {
                                auto element = find(this->age_group.begin(), this->age_group.end(), word);
                                r += to_string(distance(this->age_group.begin(), element) + 1) + ",";
                            } else if (col == 23) {
                                auto element = find(this->education.begin(), this->education.end(), word);
                                r += to_string(distance(this->education.begin(), element) + 1) + ",";
                            } else if (col == 24) {
                                auto element = find(this->race.begin(), this->race.end(), word);
                                r += to_string(distance(this->race.begin(), element) + 1) + ",";
                            } else if (col == 25) {
                                auto element = find(this->sex.begin(), this->sex.end(), word);
                                r += to_string(distance(this->sex.begin(), element) + 1) + ",";
                            } else if (col == 26) {
                                auto element = find(this->income_poverty.begin(), this->income_poverty.end(), word);
                                r += to_string(distance(this->income_poverty.begin(), element) + 1) + ",";
                            } else if (col == 27) {
                                auto element = find(this->marital_status.begin(), this->marital_status.end(), word);
                                r += to_string(distance(this->marital_status.begin(), element) + 1) + ",";
                            } else if (col == 28) {
                                auto element = find(this->rent_or_own.begin(), this->rent_or_own.end(), word);
                                r += to_string(distance(this->rent_or_own.begin(), element) + 1) + ",";
                            } else if (col == 29) {
                                auto element = find(this->employment_status.begin(), this->employment_status.end(), word);
                                r += to_string(distance(this->employment_status.begin(), element) + 1) + ",";
                            } else if (col == 30) {
                                auto element = find(this->hhs_geo_region.begin(), this->hhs_geo_region.end(), word);
                                int index = distance(this->hhs_geo_region.begin(), element);
                                r += to_string(index+1) + ",";
                            } else if (col == 31) {
                                auto element = find(this->census_msa.begin(), this->census_msa.end(), word);
                                r += to_string(distance(this->census_msa.begin(), element) + 1) + ",";
                            } else if (col == 34) {
                                auto element = find(this->employment_industry.begin(), this->employment_industry.end(), word);
                                r += to_string(distance(this->employment_industry.begin(), element) + 1) + ",";
                            } else if (col == 35) {
                                auto element = find(this->employment_occupation.begin(), this->employment_occupation.end(), word);
                                r += to_string(distance(this->employment_occupation.begin(), element) + 1);
                            } else r += to_string(stoi(word)+1) + ",";
                        } else r += "0,"; //NA
                    }
                    col++;
                }
                // last column
                if (col != 36) r += "0";
                if (train_test == 0) this->workingTrainingFeatures[counter] = r;
                else if (train_test == 1) this->workingTestFeatures[counter] = r;
                counter++;
            }
            remove("data.csv");
        }

        void readLabels(string file, int samples) {
            fstream fin;
            fin.open(file, ios::in);

            string l;
            int counter = 0;

            while(counter != samples+1) {
                getline(fin, l);
                this->workingTrainingLabels[counter] = l;
                counter++;
            }
        }

        void createValidationSet() {
            // Create indices for validation set
            srand(42);
            list<int> indexList;
            while (indexList.size() < VALIDATION_SAMPLES+1) {
                int r = rand() % TRAINING_PLUS_VALIDATION_SAMPLES;
                indexList.push_back(r);
                indexList.unique();
                indexList.sort();
                indexList.unique();
            }
            vector<int> index(indexList.size());
            copy(indexList.begin(), indexList.end(), index.begin());

            fstream foutTrainingFeatures;
            foutTrainingFeatures.open("training_features.csv", ios::out);
            fstream foutValidationFeatures;
            foutValidationFeatures.open("validation_features.csv", ios::out);
            fstream foutTrainingLabels;
            foutTrainingLabels.open("training_labels.csv", ios::out);
            fstream foutValidationLabels;
            foutValidationLabels.open("validation_labels.csv", ios::out);

            // header
            foutTrainingFeatures << this->workingTrainingFeatures[0] << "\n";
            foutValidationFeatures << this->workingTrainingFeatures[0] << "\n";
            foutTrainingLabels << this->workingTrainingLabels[0] << "\n";
            foutValidationLabels << this->workingTrainingLabels[0] << "\n";

            for (int i = 1; i < TRAINING_PLUS_VALIDATION_SAMPLES+1; i++) {
                int training_set = 1;
                for (int j = 0; j < VALIDATION_SAMPLES; j++) {
                    if (i == index.at(j)) {
                        training_set = 0;
                        break;
                    }
                }        
                if (training_set == 1) {
                    foutTrainingFeatures << this->workingTrainingFeatures[i] << "\n";
                    foutTrainingLabels << this->workingTrainingLabels[i] << "\n";
                } else {
                    foutValidationFeatures << this->workingTrainingFeatures[i] << "\n";
                    foutValidationLabels << this->workingTrainingLabels[i] << "\n";
                }
            }

            fstream foutTestFeatures;
            foutTestFeatures.open("test_features.csv", ios::out);
            for (int i = 0; i < TEST_SAMPLES+1; i++)
                foutTestFeatures << this->workingTestFeatures[i] << "\n";
        }

        void createFeatures(string file, int samples) {

            int (*data)[36] = new int[samples][36];

            fstream fin;
            fin.open(file, ios::in);
            
            string l;
            int counter = 0;
            
            while(counter != samples+1) {
                getline(fin, l);
  
                stringstream s(l);
                string word;
                int col = 0;

                while(getline(s, word, ',')) {
                    if (counter != 0) data[counter-1][col] = stoi(word);
                    col++;
                }
                counter++;
            }

            /* FEATURE ENGINEERING (i.e. Reduction of the number of features)
               In general, features with the same number of instances will be summed up as one feature by using this formula:
               new_feature = SUM_i=0^Number_of_features feature[i] * Number_of_instances^i

               For example: 
               f1|f2|f3  becomes  f4  because
               1 |0 |2            19          1*3^0 + 0*3^1 + 2*3^2 = 19
   
               Some features will be just multiplied. */

            int (*dataUnscaled)[8] = new int[samples][8];

            for (int i = 0; i < samples; i++) {
                // first column
                dataUnscaled[i][0] = data[i][0];
                // 3a
                int element = 0;
                int counter = 0;
                int columns_3a[8] = {3, 4, 5, 6, 7, 8, 9, 10};
                for (int c : columns_3a) {
                    element += data[i][c] * pow(3, counter);
                    counter++;
                }
                dataUnscaled[i][1] = element;
                // 3b
                element = 0;
                counter = 0;
                int columns_3b[8] = {11, 12, 13, 14, 15, 27, 28, 31};
                for (int c : columns_3b) {
                    element += data[i][c] * pow(3, counter);
                    counter++;
                }
                dataUnscaled[i][2] = element;
                // 4
                element = 0;
                counter = 0;
                int columns_4[4] = {2, 24, 26, 29};
                for (int c : columns_4) {
                    element += data[i][c] * pow(4, counter);
                    counter++;
                }
                dataUnscaled[i][3] = element;
                // 5
                element = 0;
                counter = 0;
                int columns_5[5] = {1, 22, 23, 32, 33};
                for (int c : columns_5) {
                    element += data[i][c] * pow(5, counter);
                    counter++;
                }
                dataUnscaled[i][4] = element;
                // 6
                element = 0;
                counter = 0;
                int columns_6[6] = {16, 17, 18, 19, 20, 21};
                for (int c : columns_6) {
                    element += data[i][c] * pow(6, counter);
                    counter++;
                }
                dataUnscaled[i][5] = element;
                // 2 and 10
                dataUnscaled[i][6] = data[i][25] * data[i][30];
                // last two features
                dataUnscaled[i][7] = data[i][34] * data[i][35];
            }

            if (samples == TRAINING_SAMPLES) this->trainingFeaturesUnscaled = dataUnscaled;
            else if (samples == VALIDATION_SAMPLES) this->validationFeaturesUnscaled = dataUnscaled;
            else if (samples == TEST_SAMPLES) this->testFeaturesUnscaled = dataUnscaled;
        }

        void createLabels(string file, int samples) {
            fstream fin;
            fin.open(file, ios::in);
            
            string l;
            int counter = 0;
            
            while(counter != samples+1) {
                getline(fin, l);
  
                stringstream s(l);
                string word;
                int col = 0;

                while(getline(s, word, ',')) {
                    if (counter != 0) {
                        if (file == "training_labels.csv") this->trainingLabels[counter-1][col] = stoi(word);
                        else if (file == "validation_labels.csv") this->validationLabels[counter-1][col] = stoi(word);
                    }
                    col++;
                }
                counter++;
            }          
        }

        void scaleFeatures() {
            remove("training_features.csv");
            remove("validation_features.csv");
            remove("test_features.csv");
            remove("training_labels.csv");
            remove("validation_labels.csv");

            for (int row = 0; row < TRAINING_SAMPLES; row++) this->trainingFeatures[row][0] = this->trainingFeaturesUnscaled[row][0];
            for (int row = 0; row < VALIDATION_SAMPLES; row++) this->validationFeatures[row][0] = this->validationFeaturesUnscaled[row][0];
            for (int row = 0; row < TEST_SAMPLES; row++) this->testFeatures[row][0] = this->testFeaturesUnscaled[row][0];

            for (int col = 1; col < 8; col++) {
                int column[TRAINING_SAMPLES];
                for (int row = 0; row < TRAINING_SAMPLES; row++) column[row] = this->trainingFeaturesUnscaled[row][col];
                double m = mean(column, TRAINING_SAMPLES);
                double s = sd(column, TRAINING_SAMPLES, m);
                for (int row = 0; row < TRAINING_SAMPLES; row++) this->trainingFeatures[row][col] = (this->trainingFeaturesUnscaled[row][col] - m) / s;
            }
            for (int col = 1; col < 8; col++) {
                int column[VALIDATION_SAMPLES];
                for (int row = 0; row < VALIDATION_SAMPLES; row++) column[row] = this->validationFeaturesUnscaled[row][col];
                double m = mean(column, VALIDATION_SAMPLES);
                double s = sd(column, VALIDATION_SAMPLES, m);
                for (int row = 0; row < VALIDATION_SAMPLES; row++) this->validationFeatures[row][col] = (this->validationFeaturesUnscaled[row][col] - m) / s;
            }
            for (int col = 1; col < 8; col++) {
                int column[TEST_SAMPLES];
                for (int row = 0; row < TEST_SAMPLES; row++) column[row] = this->testFeaturesUnscaled[row][col];
                double m = mean(column, TEST_SAMPLES);
                double s = sd(column, TEST_SAMPLES, m);
                for (int row = 0; row < TEST_SAMPLES; row++) this->testFeatures[row][col] = (this->testFeaturesUnscaled[row][col] - m) / s;
            }
        }
};