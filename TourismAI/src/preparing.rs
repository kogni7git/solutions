use csv::Reader;
use std::fs::File;
use ndarray::{Array1, Array2, ArrayBase, Ix1, Ix2, OwnedRepr};


pub fn prepare_data(mut train: Reader<File>, mut train_helper: Reader<File>, mut test: Reader<File>) -> (ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<f64>, Ix1>, ArrayBase<OwnedRepr<f64>, Ix2>) {
    // We don't use id, travel_with, most_impressing, total_female, total_male (because of missing values).
    // We don't use country and age_group. They differ in Train.csv and Test.csv.
    // night_mainland, night_zanzibar and total_cost are numerical columns and will be just copied.
    // We label-encode the categorical columns.
    let mut purpose_helper = Vec::new();
    let mut main_activity_helper = Vec::new();
    let mut info_source_helper = Vec::new();
    let mut tour_arrangement_helper = Vec::new();
    let mut package_transport_int_helper = Vec::new();
    let mut package_accomodation_helper = Vec::new();
    let mut package_food_helper_helper = Vec::new();
    let mut package_transport_tz_helper = Vec::new();
    let mut package_sightseeing_helper = Vec::new();
    let mut package_guided_tour_helper = Vec::new();
    let mut package_insurance_helper = Vec::new();
    let mut payment_mode_helper = Vec::new();
    let mut first_trip_tz_helper = Vec::new();

    for row in train_helper.records() {
        let row_record = row.unwrap();
        purpose_helper.push(row_record.get(6).unwrap().to_string());
        main_activity_helper.push(row_record.get(7).unwrap().to_string());
        info_source_helper.push(row_record.get(8).unwrap().to_string());
        tour_arrangement_helper.push(row_record.get(9).unwrap().to_string());
        package_transport_int_helper.push(row_record.get(10).unwrap().to_string());
        package_accomodation_helper.push(row_record.get(11).unwrap().to_string());
        package_food_helper_helper.push(row_record.get(12).unwrap().to_string());
        package_transport_tz_helper.push(row_record.get(13).unwrap().to_string());
        package_sightseeing_helper.push(row_record.get(14).unwrap().to_string());
        package_guided_tour_helper.push(row_record.get(15).unwrap().to_string());
        package_insurance_helper.push(row_record.get(16).unwrap().to_string());
        payment_mode_helper.push(row_record.get(19).unwrap().to_string());
        first_trip_tz_helper.push(row_record.get(20).unwrap().to_string());
    }

    purpose_helper.sort_unstable();
    purpose_helper.dedup();

    main_activity_helper.sort_unstable();
    main_activity_helper.dedup();

    info_source_helper.sort_unstable();
    info_source_helper.dedup();

    tour_arrangement_helper.sort_unstable();
    tour_arrangement_helper.dedup();

    package_transport_int_helper.sort_unstable();
    package_transport_int_helper.dedup();

    package_accomodation_helper.sort_unstable();
    package_accomodation_helper.dedup();

    package_food_helper_helper.sort_unstable();
    package_food_helper_helper.dedup();

    package_transport_tz_helper.sort_unstable();
    package_transport_tz_helper.dedup();

    package_sightseeing_helper.sort_unstable();
    package_sightseeing_helper.dedup();

    package_guided_tour_helper.sort_unstable();
    package_guided_tour_helper.dedup();

    package_insurance_helper.sort_unstable();
    package_insurance_helper.dedup();

    payment_mode_helper.sort_unstable();
    payment_mode_helper.dedup();

    first_trip_tz_helper.sort_unstable();
    first_trip_tz_helper.dedup();

    let mut features = Array2::<f64>::zeros((4809, 17));
    let mut test_features = Array2::<f64>::zeros((1601, 17));
    let mut labels = Array1::<f64>::zeros(4809);
    let mut counter = 0;

    for row in train.records() {
        let row_record = row.unwrap();
        features[[counter, 2]] = purpose_helper.iter().position(|x| x == &row_record.get(6).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 3]] = main_activity_helper.iter().position(|x| x == &row_record.get(7).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 4]] = info_source_helper.iter().position(|x| x == &row_record.get(8).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 5]] = tour_arrangement_helper.iter().position(|x| x == &row_record.get(9).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 6]] = package_transport_int_helper.iter().position(|x| x == &row_record.get(10).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 7]] = package_accomodation_helper.iter().position(|x| x == &row_record.get(11).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 8]] = package_food_helper_helper.iter().position(|x| x == &row_record.get(12).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 9]] = package_transport_tz_helper.iter().position(|x| x == &row_record.get(13).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 10]] = package_sightseeing_helper.iter().position(|x| x == &row_record.get(14).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 11]] = package_guided_tour_helper.iter().position(|x| x == &row_record.get(15).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 12]] = package_insurance_helper.iter().position(|x| x == &row_record.get(16).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 13]] = row_record.get(17).unwrap().to_string().parse().unwrap();
        features[[counter, 14]] = row_record.get(18).unwrap().to_string().parse().unwrap();
        features[[counter, 15]] = payment_mode_helper.iter().position(|x| x == &row_record.get(19).unwrap()).unwrap().to_string().parse().unwrap();
        features[[counter, 16]] = first_trip_tz_helper.iter().position(|x| x == &row_record.get(20).unwrap()).unwrap().to_string().parse().unwrap();

        labels[counter] = row_record.get(22).unwrap().trim().parse().unwrap();
        counter += 1;
    }

    counter = 0;

    for row in test.records() {
        let row_record = row.unwrap();
        test_features[[counter, 2]] = purpose_helper.iter().position(|x| x == &row_record.get(6).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 3]] = main_activity_helper.iter().position(|x| x == &row_record.get(7).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 4]] = info_source_helper.iter().position(|x| x == &row_record.get(8).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 5]] = tour_arrangement_helper.iter().position(|x| x == &row_record.get(9).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 6]] = package_transport_int_helper.iter().position(|x| x == &row_record.get(10).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 7]] = package_accomodation_helper.iter().position(|x| x == &row_record.get(11).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 8]] = package_food_helper_helper.iter().position(|x| x == &row_record.get(12).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 9]] = package_transport_tz_helper.iter().position(|x| x == &row_record.get(13).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 10]] = package_sightseeing_helper.iter().position(|x| x == &row_record.get(14).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 11]] = package_guided_tour_helper.iter().position(|x| x == &row_record.get(15).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 12]] = package_insurance_helper.iter().position(|x| x == &row_record.get(16).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 13]] = row_record.get(17).unwrap().to_string().parse().unwrap();
        test_features[[counter, 14]] = row_record.get(18).unwrap().to_string().parse().unwrap();
        test_features[[counter, 15]] = payment_mode_helper.iter().position(|x| x == &row_record.get(19).unwrap()).unwrap().to_string().parse().unwrap();
        test_features[[counter, 16]] = first_trip_tz_helper.iter().position(|x| x == &row_record.get(20).unwrap()).unwrap().to_string().parse().unwrap();

        counter += 1;
    }

    return (features, labels, test_features);
}


pub fn prepare_submission(mut sample_submission: Reader<File>, prediction: Array1<f64>) -> String {
    let mut sample_submission_ids = Vec::new();
    for row in sample_submission.records() {
        let row_record = row.unwrap();
        sample_submission_ids.push(row_record.get(0).unwrap().to_string());
    }

    let mut submission: String = "ID,total_cost\n".to_owned();
    let mut counter = 0;
    for i in sample_submission_ids {
        let id: &str = &i;
        let comma: &str = &",";
        let total_cost: &str = &prediction[counter].to_string();
        let new_line: &str = &"\n";
        submission = submission.to_owned() + id + comma + total_cost + new_line;
        counter += 1;
    }

    return submission;
}