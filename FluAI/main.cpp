#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tools.hpp"

using namespace std;
namespace fs = std::filesystem;
using namespace tensorflow;
using namespace tensorflow::ops;


struct {
    int epochs;
    float learningRate;
    int batchSize;
    int numberBatches;
    int numberValidationBatches;
    int hiddenSize;
    } settings;


int main() {
    // Welcome!
    cout << "Welcome! This program trains an Artificial Neural Network for the Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines competition on drivendata.org (https://www.drivendata.org/competitions/66/flu-shot-learning/).\n";
    cout << "Please enter the number of epochs: ";
    cin >> settings.epochs;
    cout << "\nPlease enter the learning rate: ";
    cin >> settings.learningRate;
    cout << "\nPlease enter the batch size: ";    
    cin >> settings.batchSize;
    cout << "\nPlease enter the number of batches [maximum " << to_string(int(floor(TRAINING_SAMPLES/settings.batchSize))) << "]: ";
    cin >> settings.numberBatches;
    // the number of batches for validation is the same ratio as for the training set
    settings.numberValidationBatches = int((settings.numberBatches/float(TRAINING_SAMPLES/settings.batchSize)) * float(VALIDATION_SAMPLES/settings.batchSize));
    cout << "\nPlease enter the hidden size: ";
    cin >> settings.hiddenSize;
    cout << "\nAn Artificial Neural Network with " << settings.hiddenSize << " hidden neurons will be trained " << settings.epochs << " with the batch size " << settings.batchSize << " and the learning rate " << settings.learningRate << " on 80% of the data and evaluated on 20% of the data. The data consists of " << settings.numberBatches << " batches.";

    // Load data!
    cout << "\nData will be loaded and prepared!\n";
    
    trainingAndTestData data;
    // Clean (i.e. remove commas) training features!
    data.clean("Data/training_set_features.csv", TRAINING_PLUS_VALIDATION_SAMPLES);
    // Make codes of categorical columns!
    data.makeCodes();
    // Code categorical columns!
    data.code(TRAINING_PLUS_VALIDATION_SAMPLES, 0);
    // Read labels!
    data.readLabels("Data/training_set_labels.csv", TRAINING_PLUS_VALIDATION_SAMPLES);

    // Clean test features!
    data.clean("Data/test_set_features.csv", TEST_SAMPLES);
    // Code categorical columns!
    data.code(TEST_SAMPLES, 1);

    // Create validation set!
    data.createValidationSet();

    // Create features and labels!
    data.createFeatures("training_features.csv", TRAINING_SAMPLES);
    data.createFeatures("validation_features.csv", VALIDATION_SAMPLES);
    data.createFeatures("test_features.csv", TEST_SAMPLES);
    data.createLabels("training_labels.csv", TRAINING_SAMPLES);
    data.createLabels("validation_labels.csv", VALIDATION_SAMPLES);

    // Scale features!
    data.scaleFeatures();

    // Train model!
    string RESULTS = "\n";

    Scope root = Scope::NewRootScope();
    ClientSession session(root);

    // Architecture
    auto W1 = Variable(root, TensorShape({7, settings.hiddenSize}), DT_FLOAT);
    auto W2 = Variable(root, TensorShape({settings.hiddenSize, settings.hiddenSize}), DT_FLOAT);
    auto W3 = Variable(root, TensorShape({settings.hiddenSize, 2}), DT_FLOAT);
    auto B1 = Variable(root, TensorShape({settings.hiddenSize}), DT_FLOAT);
    auto B2 = Variable(root, TensorShape({settings.hiddenSize}), DT_FLOAT);
    auto B3 = Variable(root, TensorShape({2}), DT_FLOAT);

    Scope vars = root.NewSubScope("Variables");
    auto m_var_w1 = Variable(vars, {7, settings.hiddenSize}, DT_FLOAT);
    auto v_var_w1 = Variable(vars, {7, settings.hiddenSize}, DT_FLOAT);
    auto m_var_w2 = Variable(vars, {settings.hiddenSize, settings.hiddenSize}, DT_FLOAT);
    auto v_var_w2 = Variable(vars, {settings.hiddenSize, settings.hiddenSize}, DT_FLOAT);
    auto m_var_w3 = Variable(vars, {settings.hiddenSize, 2}, DT_FLOAT);
    auto v_var_w3 = Variable(vars, {settings.hiddenSize, 2}, DT_FLOAT);

    auto m_var_b1 = Variable(vars, {settings.hiddenSize}, DT_FLOAT);
    auto v_var_b1 = Variable(vars, {settings.hiddenSize}, DT_FLOAT);
    auto m_var_b2 = Variable(vars, {settings.hiddenSize}, DT_FLOAT);
    auto v_var_b2 = Variable(vars, {settings.hiddenSize}, DT_FLOAT);
    auto m_var_b3 = Variable(vars, {2}, DT_FLOAT);
    auto v_var_b3 = Variable(vars, {2}, DT_FLOAT);

    RandomUniform::Seed(42);
    TF_CHECK_OK(session.Run({{Assign(root, W1, RandomUniform(root, {7, settings.hiddenSize}, DT_FLOAT))},
                             {Assign(root, W2, RandomUniform(root, {settings.hiddenSize, settings.hiddenSize}, DT_FLOAT))},
                             {Assign(root, W3, RandomUniform(root, {settings.hiddenSize, 2}, DT_FLOAT))},
                             {Assign(vars, m_var_w1, ZerosLike(vars, m_var_w1))},
                             {Assign(vars, v_var_w1, ZerosLike(vars, v_var_w1))},
                             {Assign(vars, m_var_w2, ZerosLike(vars, m_var_w2))},
                             {Assign(vars, v_var_w2, ZerosLike(vars, v_var_w2))},
                             {Assign(vars, m_var_w3, ZerosLike(vars, m_var_w3))},
                             {Assign(vars, v_var_w3, ZerosLike(vars, v_var_w3))},
                             {Assign(root, B1, RandomUniform(root, {settings.hiddenSize}, DT_FLOAT))},
                             {Assign(root, B2, RandomUniform(root, {settings.hiddenSize}, DT_FLOAT))},
                             {Assign(root, B3, RandomUniform(root, {2}, DT_FLOAT))},
                             {Assign(vars, m_var_b1, ZerosLike(vars, m_var_b1))},
                             {Assign(vars, v_var_b1, ZerosLike(vars, v_var_b1))},
                             {Assign(vars, m_var_b2, ZerosLike(vars, m_var_b2))},
                             {Assign(vars, v_var_b2, ZerosLike(vars, v_var_b2))},
                             {Assign(vars, m_var_b3, ZerosLike(vars, m_var_b3))},
                             {Assign(vars, v_var_b3, ZerosLike(vars, v_var_b3))}}, nullptr));

    for (int epoch = 0; epoch < settings.epochs; epoch++) {
        // TRAINING
        double losses[settings.numberBatches];
        int B = 0;
        for (int batch = 0; batch < settings.numberBatches; batch++) {
            Tensor X(DT_FLOAT, TensorShape({settings.batchSize, 7}));
            Tensor y(DT_FLOAT, TensorShape({settings.batchSize, 2}));
            auto X_map = X.tensor<float, 2>();
            auto y_map = y.tensor<float, 2>();
            for (int i = 0; i < settings.batchSize; i++) {
                for (int j = 1; j < 7 + 1; j++){ // not first column
                    X_map(i, j-1) = data.trainingFeatures[i+B][j];
                    y_map(i, 0) = data.trainingLabels[i+B][1];
                    y_map(i, 1) = data.trainingLabels[i+B][2];
                }
            }
            B += settings.batchSize;

            Scope graph = root.NewSubScope("Graph");
            auto H1 = Relu(graph, BiasAdd(graph, MatMul(graph, X, W1), B1));
            auto H2 = Relu(graph, BiasAdd(graph, MatMul(graph, H1, W2), B2));
            auto O = Sigmoid(graph, BiasAdd(graph, MatMul(graph, H2, W3), B3));

            /* We have given a multilabel problem. So, the loss function is given by the Binary Cross-Entropy Loss:
               Loss = - (1/N) SUM_over_N (y_true * log(y_pred) + (1-y_true) * log(1-y_pred)) where N is the number of labels. */
            Scope l = root.NewSubScope("Loss");
            auto loss = Negate(l, Mean(l, Add(l, Multiply(l, y, Log(l, O)), Multiply(l, Subtract(l, OnesLike(l, y), y), Log(l, Subtract(l, OnesLike(l, y), O)))), 1));

            Scope g = root.NewSubScope("g");
            vector<Output> gradients;
            TF_CHECK_OK(AddSymbolicGradients(g, {loss}, {W1, W2, W3, B1, B2, B3}, &gradients));

            Scope adam = root.NewSubScope("adam");
            auto adam_1 = ApplyAdam(adam, W1, m_var_w1, v_var_w1, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[0]});
            auto adam_2 = ApplyAdam(adam, W2, m_var_w2, v_var_w2, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[1]});
            auto adam_3 = ApplyAdam(adam, W3, m_var_w3, v_var_w3, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[2]});
            auto adam_4 = ApplyAdam(adam, B1, m_var_b1, v_var_b1, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[3]});
            auto adam_5 = ApplyAdam(adam, B2, m_var_b2, v_var_b2, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[4]});
            auto adam_6 = ApplyAdam(adam, B3, m_var_b3, v_var_b3, 0.f, 0.f, settings.learningRate, 0.9f, 0.999f, 0.00000001f, {gradients[5]});

            vector<Tensor> outputloss;
            TF_CHECK_OK(session.Run({{loss}, {adam_1}, {adam_2}, {adam_3}, {adam_4}, {adam_5}, {adam_6}}, &outputloss));
            gradients.clear();

            double l_sum = 0.0;
            for (int i = 0; i < settings.batchSize; i++)
                l_sum += outputloss[0].vec<float>()(i);
            outputloss.clear();
            l_sum /= settings.batchSize;
            losses[batch] = l_sum;
        }

        // VALIDATION
        double val_losses[settings.numberValidationBatches];
        B = 0;
        for (int batch = 0; batch < settings.numberValidationBatches; batch++) {
            Tensor X_val(DT_FLOAT, TensorShape({settings.batchSize, 7}));
            Tensor y_val(DT_FLOAT, TensorShape({settings.batchSize, 2}));
            auto X_val_map = X_val.tensor<float, 2>();
            auto y_val_map = y_val.tensor<float, 2>();
            for (int i = 0; i < settings.batchSize; i++) {
                for (int j = 1; j < 7 + 1; j++) {
                    X_val_map(i, j-1) = data.validationFeatures[i+B][j];
                    y_val_map(i, 0) = data.validationLabels[i+B][1];
                    y_val_map(i, 1) = data.validationLabels[i+B][2];
                }
            }
            B += settings.batchSize;

            Scope graph = root.NewSubScope("Graph_val");
            auto H1 = Relu(graph, BiasAdd(graph, MatMul(graph, X_val, W1), B1));
            auto H2 = Relu(graph, BiasAdd(graph, MatMul(graph, H1, W2), B2));
            auto O = Sigmoid(graph, BiasAdd(graph, MatMul(graph, H2, W3), B3));

            vector<Tensor> outputloss;
            Scope l = root.NewSubScope("Loss_val");
            auto loss = Negate(l, Mean(l, Add(l, Multiply(l, y_val, Log(l, O)), Multiply(l, Subtract(l, OnesLike(l, y_val), y_val), Log(l, Subtract(l, OnesLike(l, y_val), O)))), 1));
            TF_CHECK_OK(session.Run({loss}, &outputloss));
            double l_sum = 0.0;
            for (int i = 0; i < settings.batchSize; i++)
                l_sum += outputloss[0].vec<float>()(i);
            outputloss.clear();
            l_sum /= settings.batchSize;
            val_losses[batch] = l_sum;
        }
        cout << "Epoch: " << to_string(epoch) << "; Loss: " << to_string(mean(losses, settings.numberBatches)) << "; Validation Loss: " << to_string(mean(val_losses, settings.numberValidationBatches)) << "\n";
        RESULTS += "Epoch: " + to_string(epoch) + "; Loss: " + to_string(mean(losses, settings.numberBatches)) + "; Validation Loss: " + to_string(mean(val_losses, settings.numberValidationBatches)) + "\n";
    }

    // PREDICTION
    int x = 0;
    string submission = "respondent_id,h1n1_vaccine,seasonal_vaccine\n";
    int B = 0;
    int testBatchSize = 1024;
    for (int batch = 0; batch < ceil(TEST_SAMPLES/float(testBatchSize)); batch++) {
        int samples;
        if (batch == floor(TEST_SAMPLES/float(testBatchSize))) {
            samples = ((TEST_SAMPLES/float(testBatchSize)) - floor(TEST_SAMPLES/float(testBatchSize))) * float(testBatchSize);
        } else samples = testBatchSize;

        Tensor X_test(DT_FLOAT, TensorShape({samples, 7}));
        auto X_test_map = X_test.tensor<float, 2>();
        for (int i = 0; i < samples; i++) {
            for (int j = 1; j < 7 + 1; j++)
                X_test_map(i, j-1) = data.testFeatures[i+B][j];
        }

        Scope graph = root.NewSubScope("Graph_test");
        auto H1 = Relu(graph, BiasAdd(graph, MatMul(graph, X_test, W1), B1));
        auto H2 = Relu(graph, BiasAdd(graph, MatMul(graph, H1, W2), B2));
        auto O = Sigmoid(graph, BiasAdd(graph, MatMul(graph, H2, W3), B3));

        vector<Tensor> outputs;
        TF_CHECK_OK(session.Run({O}, &outputs));
        for (int i = 0; i < samples; i++)
            submission = submission + to_string(int(data.testFeatures[i+B][0])) + "," + to_string(outputs[0].matrix<float>()(i, 0)) + "," + to_string(outputs[0].matrix<float>()(i, 1)) + "\n";
        B += samples;
    }

    cout << "\nExperiment successful!";

    // Make Submission!
    int counter = 1;
    while (access(("Submission/" + to_string(counter)).c_str(), F_OK) == 0)
        counter++;

    string output = "# Experiment Number " + to_string(counter) + "\n## Settings:\n* Epochs: " + to_string(settings.epochs) + "\n* Learning Rate: " + to_string(settings.learningRate) + "\n* Hidden Size: " + to_string(settings.hiddenSize) + "\n* Batch Size: " + to_string(settings.batchSize) + "\n* Number of Batches: " + to_string(settings.numberBatches) + "\n## Results: " + RESULTS;

    fs::create_directories("Submission/" + to_string(counter));
  
    string dirSubmission = "Submission/" + to_string(counter) + "/submission.csv";
    ofstream fileSubmission(dirSubmission);
    fileSubmission << submission;
    fileSubmission.close();

    string dirOutput = "Submission/" + to_string(counter) + "/output.md";
    ofstream fileOutput(dirOutput);
    fileOutput << output;
    fileOutput.close();
   
    cout << "\nsubmission.csv and output.md are saved under " << "Submission/" + to_string(counter);

    return 0;
};