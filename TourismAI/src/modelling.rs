use linfa::DatasetBase;
use linfa::metrics::SingleTargetRegression;
use linfa::prelude::{Fit, Predict};
use ndarray::{Array1, ArrayBase, Ix1, Ix2, OwnedRepr};
use linfa_elasticnet::ElasticNet;
use linfa_svm::Svm;


pub fn train_elastic_net(dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>, test_features: ArrayBase<OwnedRepr<f64>, Ix2>) -> (Array1<f64>, f64, String) {

    let (dataset_train, dataset_valid) = dataset.split_with_ratio(0.8);

    println!("Penalty? ");
    let mut penalty_string = String::new();
    std::io::stdin().read_line(&mut penalty_string).unwrap();
    let penalty: f64 = penalty_string.trim().parse().unwrap();

    println!("LASSO? ");
    let mut lasso_string = String::new();
    std::io::stdin().read_line(&mut lasso_string).unwrap();
    let lasso: f64 = lasso_string.trim().parse().unwrap();

    let settings = format!("## Settings:\n* Penalty: {}\n* LASSO: {}\n", penalty, lasso);

    let elastic_net = ElasticNet::params()
        .penalty(penalty)
        .l1_ratio(lasso)
        .fit(&dataset_train).unwrap();
    let elastic_net_evaluation = elastic_net.predict(&dataset_valid);
    let mae = elastic_net_evaluation.mean_absolute_error(&dataset_valid).unwrap();
    let prediction = elastic_net.predict(&test_features);
    return (prediction, mae, settings);
}


pub fn train_svm(dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>, test_features: ArrayBase<OwnedRepr<f64>, Ix2>) -> (Array1<f64>, f64, String) {

    let prediction = Array1::<f64>::zeros(4809);
    let mae = 0.0;
    let settings = format!("");

    let (dataset_train, dataset_valid) = dataset.split_with_ratio(0.8);

    println!("epsilon (type 0) or nu (type 1)? ");
    let mut epsilon_or_nu_string = String::new();
    std::io::stdin().read_line(&mut epsilon_or_nu_string).unwrap();
    let epsilon_or_nu: i32 = epsilon_or_nu_string.trim().parse().unwrap();

    if epsilon_or_nu == 0 {
        println!("Epsilon? ");
        let mut epsilon_string = String::new();
        std::io::stdin().read_line(&mut epsilon_string).unwrap();
        let epsilon: f64 = epsilon_string.trim().parse().unwrap();
        let settings = format!("## Settings:\n* epsilon: {}\n", epsilon);

        let svm = Svm::params()
            .gaussian_kernel(100.0)
            .eps(epsilon)
            .fit(&dataset_train).unwrap();

        let svm_evaluation = svm.predict(&dataset_valid);
        let mae = svm_evaluation.mean_absolute_error(&dataset_valid).unwrap();
        let prediction = svm.predict(&test_features);

        return (prediction, mae, settings);

    } else if epsilon_or_nu == 1 {
        println!("Nu? ");
        let mut nu_string = String::new();
        std::io::stdin().read_line(&mut nu_string).unwrap();
        let nu: f64 = nu_string.trim().parse().unwrap();
        let settings = format!("## Settings:\n* nu: {}\n", nu);

        let svm = Svm::params()
            .gaussian_kernel(100.0)
            .nu_weight(nu)
            .fit(&dataset_train).unwrap();

        let svm_evaluation = svm.predict(&dataset_valid);
        let mae = svm_evaluation.mean_absolute_error(&dataset_valid).unwrap();
        let prediction = svm.predict(&test_features);

        return (prediction, mae, settings);
    }
    return (prediction, mae, settings);
}


pub fn train_model(dataset: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>>, test_features: ArrayBase<OwnedRepr<f64>, Ix2>, model: &str) -> (Array1<f64>, f64, String) {
    let prediction = Array1::<f64>::zeros(4809);
    let mae = 0.0;
    let settings = format!("");

    if model == "Elastic Net" {
        return train_elastic_net(dataset, test_features);
    } else if model == "SVM" {
        return train_svm(dataset, test_features);
    }
    return (prediction, mae, settings);
}