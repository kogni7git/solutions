mod preparing;
mod modelling;

use std::fs;
use std::fs::File;
use linfa::dataset::Dataset;
use std::path::Path;


fn main() {
    // Welcome!
    println!("Welcome! This program trains a regression model (Elastic Net or SVM) for the Tanzania Tourism Prediction by Pycon Tanzania Community competition on zindi.africa (https://zindi.africa/competitions/tanzania-tourism-prediction)!");
    let possible_models = vec!["Elastic Net", "SVM"];

    println!("Please choose a regression model for training: ");
    let mut counter = 0;
    for i in possible_models.iter() {
        println!("{} for {}.", counter, i);
        counter += 1;
    }

    let mut chosen_model_string = String::new();
    println!("Please enter the number of the desired model: ");
    std::io::stdin().read_line(&mut chosen_model_string).unwrap();
    let chosen_model: usize = chosen_model_string.trim().parse().unwrap();

    println!("A {} will be trained and validated on 20% of the data.", possible_models[chosen_model]);

    // Load data!
    let train_csv = csv::ReaderBuilder::new().has_headers(true).from_reader(File::open("Data/Train.csv").unwrap());
    let train_csv_helper = csv::ReaderBuilder::new().has_headers(true).from_reader(File::open("Data/Train.csv").unwrap());
    let test_csv = csv::ReaderBuilder::new().has_headers(true).from_reader(File::open("Data/Test.csv").unwrap());
    let sample_submission_csv = csv::ReaderBuilder::new().has_headers(true).from_reader(File::open("Data/SampleSubmission.csv").unwrap());

    // Prepare data!
    let (features, labels, test_features) = preparing::prepare_data(train_csv, train_csv_helper, test_csv);

    // Train model!
    let dataset = Dataset::new(features, labels);
    let (prediction, mae, settings) = modelling::train_model(dataset, test_features, possible_models[chosen_model]);

    println!("Experiment successful!");
    println!("The MAE is {}.", &mae);

    // Make Submission!
    let submission = preparing::prepare_submission(sample_submission_csv, prediction);
    counter = 1;
    while Path::new(&("Submission/".to_owned() + &counter.to_string())).exists() {
        counter += 1;
    }

    let output = format!("# Experiment Number {}\n## Model:\n{}\n{}## Result:\nMAE: {}", counter, possible_models[chosen_model], settings, mae);
    fs::create_dir(&("Submission/".to_owned() + &counter.to_string())).unwrap();
    fs::write(&("Submission/".to_owned() + &counter.to_string() + &"/submission.csv"), submission).unwrap();
    fs::write(&("Submission/".to_owned() + &counter.to_string() + &"/output.md"), output).unwrap();
    println!("{}", "submission.csv and output.md are saved under ".to_owned() + "Submission/" + &counter.to_string());
}