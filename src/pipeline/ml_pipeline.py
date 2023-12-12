import sys
import os

import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(parent_dir)

from ml.src.pipeline.ml_helper import assess_similarity, get_trained_model, init_aeid, load_config, get_assay_df, get_fingerprint_df, \
    merge_assay_and_fingerprint_df, split_data, partition_data, handle_oversampling, grid_search_cv, build_pipeline, get_label_counts, report_exception, save_model, \
    build_preprocessing_pipeline, preprocess_all_sets, load_model, init_estimator_pipeline, \
    get_feature_importance_if_applicable, init_preprocessing_pipeline, init_target_variable, init_validation_set, \
    init_ml_algo, predict_and_report, get_total_elapsed_time, add_status_file

from datetime import datetime
import traceback
from ml.src.utils.helper import get_subset_aeids

if __name__ == '__main__':
    
    # Get ML configuration
    CONFIG, START_TIME, LOGGER, TARGET_RUN_FOLDER = load_config()

    # Get fingerprint data
    fps_df = get_fingerprint_df()

    # Get assay endpoint ids from subset considered for ML
    aeids_target_assays = get_subset_aeids()['aeid']

    # Iterate through target variables (hitcall, hitcall_c)
    for target_variable in CONFIG['target_variables']:
        LOGGER.info(f"Start ML pipeline for target variable: {target_variable}")
        LOGGER.info("#" * 60)
        TARGET_VARIABLE = init_target_variable(target_variable)

        # Iterate through ML algorithms (binary classification, regression)
        for ml_algorithm in CONFIG['ml_algorithms']:
            LOGGER.info(f"Start ML pipeline for ML algorithm: {ml_algorithm}")
            LOGGER.info("#" * 60)
            ML_ALGORITHM, CONFIG_ESTIMATORS = init_ml_algo(ml_algorithm)

            # Iterate through aeids_target_assays and launch each iteration in a separate process
            for aeid in aeids_target_assays[:]:  # [97]: #
                try:
                    # Init aeid
                    init_aeid(aeid)
                    LOGGER.info(f"Start ML pipeline for assay ID: {aeid}")
                    LOGGER.info("#" * 60)

                    # Get assay data
                    assay_df = get_assay_df()

                    # Merge chemical ids in both dataframes
                    df = merge_assay_and_fingerprint_df(assay_df, fps_df)

                    # Partition data into X and y and respective massbank validation set (massbank validation set evaluates generalization to unseen data from spectral data)
                    feature_names, X, y, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val = partition_data(df)

                    # Calculate the similarity between the two massbank validation sets, true and predicted fingerprints
                    assess_similarity(X_massbank_val_from_structure, X_massbank_val_from_sirius)

                    # Split ML data into train test set (test set evaluates generalization to unseen data)
                    X_train, y_train, X_test, y_test = split_data(X, y)

                    preprocessing_pipelines = build_preprocessing_pipeline()

                    for preprocessing_pipeline in preprocessing_pipelines:
                        init_preprocessing_pipeline(preprocessing_pipeline)
                        
                        feature_names, X_train, y_train, X_test, y_test, X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val = \
                            preprocess_all_sets(preprocessing_pipeline, feature_names, X_train, y_train, X_test, y_test,
                                                X_massbank_val_from_structure, X_massbank_val_from_sirius, y_massbank_val)

                        # Apply oversampling if configured
                        X_train, y_train = handle_oversampling(X_train, y_train)

                        # Get the label counts
                        get_label_counts(y, y_train, y_test, y_massbank_val)

                        LOGGER.info(f"Run pipeline for all estimaors:\n")

                        # Build for each estimator a pipeline according to the configurations in the config file
                        for estimator in CONFIG_ESTIMATORS['estimators']:
                            start_time = datetime.now()

                            # Init a new folder for this estimator
                            estimator_name = estimator['name']
                            ESTIMATOR_PIPELINE = init_estimator_pipeline(estimator_name)
                            LOGGER.info(f"Apply {ESTIMATOR_PIPELINE}")

                            # Training #
                            if not CONFIG['apply']['only_predict']:
                                # Build the pipeline for the current estimator with the specified parameter grid
                                pipeline = build_pipeline(estimator)

                                # Perform grid search (Note: CV on TRAINING set with RepeatedStratifiedKFold)
                                LOGGER.info("Start Grid Search Cross-Validation..")
                                grid_search = grid_search_cv(X_train, y_train, estimator, pipeline)
                                LOGGER.info("Training Done.\n")

                                # Save best estimator (estimator with best performing parameters from grid search)
                                best_estimator = grid_search.best_estimator_
                            else:
                                best_estimator_path = get_trained_model("train")
                                best_estimator = load_model(best_estimator_path, "train")

                                best_estimator_train_test_path = get_trained_model("train_test")
                                best_estimator_train_test = load_model(best_estimator_train_test_path, "train_test")

                                best_estimator_full_data_path = get_trained_model("full_data")
                                best_estimator_full_data = load_model(best_estimator_full_data_path, "full_data")


                            save_model(best_estimator, "train")

                            # Validation #
                            # Predict on the test set with the best estimator (X_test, y_test is unseen)
                            LOGGER.info("Start Internal Validation..")
                            VALIDATION_SET = init_validation_set("val")
                            predict_and_report(X_test, y_test, best_estimator)
                            LOGGER.info("Internal Validation Done.\n")

                            # Retrain the best estimator from GridSearchCV with train+test set for Massbank validation (unseen)
                            init_estimator_pipeline(estimator_name)  # re-init estimator's DUMP_FOLDER
                            if not CONFIG['apply']['only_predict']:
                                LOGGER.info(f"Retrain on train+test set..\n")
                                X_combined = np.vstack((X_train, X_test))
                                y_combined = np.concatenate((y_train, y_test))
                                best_estimator.fit(X_combined, y_combined)
                                save_model(best_estimator, "train_test")
                            else:
                                best_estimator = best_estimator_train_test

                            save_model(best_estimator, "train_test")                           

                            # Predict on the 1. "true Massbank" and 2. "SIRIUS predicted" validation set
                            LOGGER.info("Start MassBank Validation")
                            validation_set_names = ["structure", "sirius"]
                            validation_sets = [X_massbank_val_from_structure, X_massbank_val_from_sirius]
                            for name, data in zip(validation_set_names, validation_sets):
                                init_validation_set(f"mb_val_{name}")
                                predict_and_report(data, y_massbank_val, best_estimator)

                            LOGGER.info("MassBank Validation Done.\n")

                            # Retrain the estimator on full data for future predictions
                            init_estimator_pipeline(estimator_name)  # re-init estimator's DUMP_FOLDER
                            if not CONFIG['apply']['only_predict']:
                                LOGGER.info(f"Retrain on full data..\n")
                                X_all = np.vstack((X_combined, X_massbank_val_from_structure))
                                y_all = np.concatenate((y_combined, y_massbank_val))
                                best_estimator.fit(X_all, y_all)
                            else:
                                best_estimator = best_estimator_full_data
                                
                            save_model(best_estimator, "full_data")

                            # Get feature importances (only implemented for XGB and RandomForest)
                            feature_importances = get_feature_importance_if_applicable(best_estimator, feature_names)

                            # Time elapsed
                            elapsed = round((datetime.now() - start_time).total_seconds(), 2)
                            LOGGER.info(f"Done {estimator['name']} >> {elapsed} seconds.\n\n\n")

                except Exception as e:
                    traceback_info = traceback.format_exc()
                    report_exception(e, traceback_info, aeid)

                    # Write a failed flag file to the aeid folder
                    init_aeid(aeid)  # re-init aeid's DUMP_FOLDER
                    add_status_file("failed")

    # Calculate the total elapsed time
    LOGGER.info(f"Finished all: Total time >> { get_total_elapsed_time()}\n")
