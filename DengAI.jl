# This is DengAI.
# author: Andreas Schmitt
# date: April/May/July 2020

# Seed
import Random
Random.seed!(42)

ACTUAL_VERSION = 7

# Loading the data
using DataFrames, CSV
X = DataFrame!(CSV.File("./Data/dengue_features_train.csv"))
X_test = DataFrame!(CSV.File("./Data/dengue_features_test.csv"))
y = DataFrame!(CSV.File("./Data/dengue_labels_train.csv"))
y_test = DataFrame!(CSV.File("./Data/submission_format.csv"))

# PREPROCESSING
# Filter data by city
X_sj = filter(:city => city -> city == "sj", X)
X_iq = filter(:city => city -> city == "iq", X)
X_test_sj = filter(:city => city -> city == "sj", X_test)
X_test_iq = filter(:city => city -> city == "iq", X_test)
y_sj = filter(:city => city -> city == "sj", y)
y_iq = filter(:city => city -> city == "iq", y)

# Remove unnecessary columns
X_sj = select!(X_sj, Not([ :city, :year, :weekofyear, :week_start_date ]))
X_iq = select!(X_iq, Not([ :city, :year, :weekofyear, :week_start_date ]))
X_test_sj = select!(X_test_sj, Not([ :city, :year, :weekofyear, :week_start_date ]))
X_test_iq = select!(X_test_iq, Not([ :city, :year, :weekofyear, :week_start_date ]))
y_sj = select!(y_sj, Not([ :city, :year, :weekofyear ]))
y_iq = select!(y_iq, Not([ :city, :year, :weekofyear ]))

# Impute missing values
using MLJ

@load FillImputer
imputer_sj = machine(FillImputer(), X_sj)
fit!(imputer_sj)
X_sj = MLJ.transform(imputer_sj, X_sj)

imputer_iq = machine(FillImputer(), X_iq)
fit!(imputer_iq)
X_iq = MLJ.transform(imputer_iq, X_iq)

imputer_test_sj = machine(FillImputer(), X_test_sj)
fit!(imputer_test_sj)
X_test_sj = MLJ.transform(imputer_test_sj, X_test_sj)

imputer_test_iq = machine(FillImputer(), X_test_iq)
fit!(imputer_test_iq)
X_test_iq = MLJ.transform(imputer_test_iq, X_test_iq)

# Cross-Validation for time series
function time_series_cv(window_train, window_test, fixed, length_X)
    last_index = window_train + window_test
    indices = []

    if fixed == false
        while last_index <= length_X
            indices = cat(indices, (1:window_train, window_train+1:window_train+window_test), dims=1)
            window_train += 10
            last_index += 10
        end
    else
        first_index = 1
        while last_index <= length_X
            indices = cat(indices, (first_index:window_train, window_train+1:window_train+window_test), dims=1)
            first_index += 10
            window_train += 10
            last_index += 10
        end
    end

    indices = convert(Array{Tuple{UnitRange{Int64},UnitRange{Int64}},1}, indices)

    return indices
end

# Preparation for Training
coerce!(y_sj, Count => Continuous)
coerce!(y_iq, Count => Continuous)

y_sj = convert(Vector, y_sj[!, "total_cases"])
y_iq = convert(Vector, y_iq[!, "total_cases"])

# Find appropriate models
possible_models = models() do model
    matching(model, X_sj, y_sj) &&
    model.prediction_type == :deterministic &&
    model.is_supervised &&
    model.is_pure_julia
end

for m in possible_models
    println(m[:name], " in ", m[:package_name])
end

# Set CV
indices_sj = time_series_cv(50, 10, true, length(X_sj[!, 1]))
indices_iq = time_series_cv(50, 10, true, length(X_iq[!, 1]))

# Procedure for Training
function training(model_sj, model_iq, name, cv_indices_sj, cv_indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, n)
    if r2_sj === nothing
        tune_sj = TunedModel(model=model_sj, tuning=RandomSearch(rng=42), resampling=cv_indices_sj, range=[r1_sj], measure=mae, n=n);
        tune_iq = TunedModel(model=model_iq, tuning=RandomSearch(rng=42), resampling=cv_indices_iq, range=[r1_iq], measure=mae, n=n);
    else
        tune_sj = TunedModel(model=model_sj, tuning=RandomSearch(rng=42), resampling=cv_indices_sj, range=[r1_sj, r2_sj], measure=mae, n=n);
        tune_iq = TunedModel(model=model_iq, tuning=RandomSearch(rng=42), resampling=cv_indices_iq, range=[r1_iq, r2_iq], measure=mae, n=n);
    end

    mach_sj = machine(tune_sj, X_sj, y_sj)
    mach_iq = machine(tune_iq, X_iq, y_iq)

    MLJ.fit!(mach_sj)
    MLJ.fit!(mach_iq)

    MLJ.save(string("./Submission/Version ", ACTUAL_VERSION, "/mach_sj_", name, ".jlso"), mach_sj)
    MLJ.save(string("./Submission/Version ", ACTUAL_VERSION, "/mach_iq_", name, ".jlso"), mach_iq)
    # For loading: load all used packages, then mach = machine(".jlso")

    return mach_sj, mach_iq
end

# Procedure for Prediction
function prediction(mach_sj, mach_iq)
    pred_sj = MLJ.predict(mach_sj, X_test_sj)
    pred_iq = MLJ.predict(mach_iq, X_test_iq)

    y_test[!, "total_cases"] = vcat(convert(Array{Int}, [floor(i) for i in pred_sj]),
                                    convert(Array{Int}, [floor(i) for i in pred_iq]))

    CSV.write(string("./Submission/Version ", ACTUAL_VERSION, "/output.csv"), y_test)
end

# Procedure for Comparison
function comparison(models_dict_sj, models_dict_iq)
    measurements_dict_sj = Dict()
    measurements_dict_iq = Dict()

    for (name, model) in models_dict_sj
        measurements_dict_sj[name] = round(model.report[2][2][1], digits=2)
    end

    for (name, model) in models_dict_iq
        measurements_dict_iq[name] = round(model.report[2][2][1], digits=2)
    end

    measurements_dict_sj = sort(collect(measurements_dict_sj), by=x->x[2])
    measurements_dict_iq = sort(collect(measurements_dict_iq), by=x->x[2])

    io = open(string("./Submission/Version ", ACTUAL_VERSION, "/comparison.txt"), "w");

    println("\nCOMPARISON\n")
    write(io, "COMPARISON\n")
    println("sj\n")
    write(io, "sj\n")

    for (name, measurement) in measurements_dict_sj
        println(name, ": ", measurement)
        write(io, string(name, ": ", measurement, "\n"))
    end

    println("\niq\n")
    write(io, "\niq\n")

    for (name, measurement) in measurements_dict_iq
        println(name, ": ", measurement)
        write(io, string(name, ": ", measurement, "\n"))
    end

    close(io)

    # best models
    name_sj = findmin(Dict(measurements_dict_sj))[2]
    name_iq = findmin(Dict(measurements_dict_iq))[2]
    mach_sj_best = models_dict_sj[name_sj]
    mach_iq_best = models_dict_iq[name_iq]

    MLJ.save(string("./Submission/Version ", ACTUAL_VERSION, "/mach_sj_best.jlso"), mach_sj_best)
    MLJ.save(string("./Submission/Version ", ACTUAL_VERSION, "/mach_iq_best.jlso"), mach_iq_best)

    return mach_sj_best, mach_iq_best
end

# TRAINING
# Packages
using DecisionTree
using MLJModels
using MLJLinearModels
using EvoTrees
using NearestNeighbors
using MLJFlux
using MultivariateStats

regression_models_sj = Dict()
regression_models_iq = Dict()

println()

# 1. DecisionTreeRegressor in DecisionTree
println("1. DecisionTreeRegressor")
model_sj_decisiontree = @load DecisionTreeRegressor pkg=DecisionTree
model_iq_decisiontree = @load DecisionTreeRegressor pkg=DecisionTree

r1_sj = range(model_sj_decisiontree, :(max_depth), lower=2, upper=50);
r2_sj = range(model_sj_decisiontree, :(n_subfeatures), lower=2, upper=20);
r1_iq = range(model_iq_decisiontree, :(max_depth), lower=2, upper=50);
r2_iq = range(model_iq_decisiontree, :(n_subfeatures), lower=2, upper=20);

model_sj_decisiontree, model_iq_decisiontree = training(model_sj_decisiontree, model_iq_decisiontree, "decisiontree", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 50)

regression_models_sj["decisiontree"] = model_sj_decisiontree
regression_models_iq["decisiontree"] = model_iq_decisiontree

println()

# 2. HuberRegressor in MLJLinearModels
println("2. HuberRegressor")
model_sj_huber = @load HuberRegressor pkg=MLJLinearModels
model_iq_huber = @load HuberRegressor pkg=MLJLinearModels

r1_sj = range(model_sj_huber, :(lambda), lower=1e-5, upper=0.1);
r2_sj = range(model_sj_huber, :(gamma), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_huber, :(lambda), lower=1e-5, upper=0.1);
r2_iq = range(model_iq_huber, :(gamma), lower=1e-5, upper=0.1);

model_sj_huber, model_iq_huber = training(model_sj_huber, model_iq_huber, "huber", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 20)

regression_models_sj["huber"] = model_sj_huber
regression_models_iq["huber"] = model_iq_huber

println()

# 3. KNNRegressor in NearestNeighbors
println("3. KNNRegressor")
model_sj_knn = @load KNNRegressor pkg=NearestNeighbors
model_iq_knn = @load KNNRegressor pkg=NearestNeighbors

r1_sj = range(model_sj_knn, :(K), lower=2, upper=30);
r2_sj = range(model_sj_knn, :(leafsize), lower=2, upper=10);
r1_iq = range(model_iq_knn, :(K), lower=2, upper=30);
r2_iq = range(model_iq_knn, :(leafsize), lower=2, upper=10);

model_sj_knn, model_iq_knn = training(model_sj_knn, model_iq_knn, "knn", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 100)

regression_models_sj["knn"] = model_sj_knn
regression_models_iq["knn"] = model_iq_knn

println()

# 4. LADRegressor in MLJLinearModels
println("4. LADRegressor")
model_sj_lad = @load LADRegressor pkg=MLJLinearModels
model_iq_lad = @load LADRegressor pkg=MLJLinearModels

r1_sj = range(model_sj_lad, :(lambda), lower=1e-5, upper=0.1);
r2_sj = range(model_sj_lad, :(gamma), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_lad, :(lambda), lower=1e-5, upper=0.1);
r2_iq = range(model_iq_lad, :(gamma), lower=1e-5, upper=0.1);

model_sj_lad, model_iq_lad = training(model_sj_lad, model_iq_lad, "lad", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 100)

regression_models_sj["lad"] = model_sj_lad
regression_models_iq["lad"] = model_iq_lad

println()

# 5. QuantileRegressor in MLJLinearModels
println("5. QuantileRegressor")
model_sj_quantile = @load QuantileRegressor pkg=MLJLinearModels
model_iq_quantile = @load QuantileRegressor pkg=MLJLinearModels

r1_sj = range(model_sj_quantile, :(lambda), lower=1e-5, upper=0.1);
r2_sj = range(model_sj_quantile, :(gamma), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_quantile, :(lambda), lower=1e-5, upper=0.1);
r2_iq = range(model_iq_quantile, :(gamma), lower=1e-5, upper=0.1);

model_sj_quantile, model_iq_quantile = training(model_sj_quantile, model_iq_quantile, "quantile", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 100)

regression_models_sj["quantile"] = model_sj_quantile
regression_models_iq["quantile"] = model_iq_quantile

println()

# 6. RandomForestRegressor in DecisionTree
println("6. RandomForestRegressor")
model_sj_randomforest = @load RandomForestRegressor pkg=DecisionTree
model_iq_randomforest = @load RandomForestRegressor pkg=DecisionTree

r1_sj = range(model_sj_randomforest, :(max_depth), lower=2, upper=50);
r2_sj = range(model_iq_randomforest, :(n_subfeatures), lower=2, upper=20);
r1_iq = range(model_sj_randomforest, :(max_depth), lower=2, upper=50);
r2_iq = range(model_iq_randomforest, :(n_subfeatures), lower=2, upper=20);

model_sj_randomforest, model_iq_randomforest = training(model_sj_randomforest, model_iq_randomforest, "randomforest", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 50)

regression_models_sj["randomforest"] = model_sj_randomforest
regression_models_iq["randomforest"] = model_iq_randomforest

println()

# 7. RidgeRegressor in MLJLinearModels
println("7. RidgeRegressor")
model_sj_ridge1 = @load RidgeRegressor pkg=MLJLinearModels
model_iq_ridge1 = @load RidgeRegressor pkg=MLJLinearModels

r1_sj = range(model_sj_ridge1, :(lambda), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_ridge1, :(lambda), lower=1e-5, upper=0.1);

model_sj_ridge1, model_iq_ridge1 = training(model_sj_ridge1, model_iq_ridge1, "ridge1", indices_sj, indices_iq, r1_sj, r1_iq, nothing, nothing, 100)

regression_models_sj["ridge1"] = model_sj_ridge1
regression_models_iq["ridge1"] = model_iq_ridge1

println()

# 8. RidgeRegressor in MultivariateStats
println("8. RidgeRegressor")
model_sj_ridge2 = @load RidgeRegressor pkg=MultivariateStats
model_iq_ridge2 = @load RidgeRegressor pkg=MultivariateStats

r1_sj = range(model_sj_ridge2, :(lambda), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_ridge2, :(lambda), lower=1e-5, upper=0.1);

model_sj_ridge2, model_iq_ridge2 = training(model_sj_ridge2, model_iq_ridge2, "ridge2", indices_sj, indices_iq, r1_sj, r1_iq, nothing, nothing, 100)

regression_models_sj["ridge2"] = model_sj_ridge2
regression_models_iq["ridge2"] = model_iq_ridge2

println()

# 9. RobustRegressor in MLJLinearModels
println("9. RobustRegressor")
model_sj_robust = @load RobustRegressor pkg=MLJLinearModels
model_iq_robust = @load RobustRegressor pkg=MLJLinearModels

r1_sj = range(model_sj_robust, :(lambda), lower=1e-5, upper=0.1);
r2_sj = range(model_sj_robust, :(gamma), lower=1e-5, upper=0.1);
r1_iq = range(model_iq_robust, :(lambda), lower=1e-5, upper=0.1);
r2_iq = range(model_iq_robust, :(gamma), lower=1e-5, upper=0.1);

model_sj_robust, model_iq_robust = training(model_sj_robust, model_iq_robust, "robust", indices_sj, indices_iq, r1_sj, r1_iq, r2_sj, r2_iq, 150)

regression_models_sj["robust"] = model_sj_robust
regression_models_iq["robust"] = model_iq_robust

println()

# COMPARISON
mach_sj_best, mach_iq_best = comparison(regression_models_sj, regression_models_iq)

# PREDICTION
prediction(mach_sj_best, mach_iq_best)
