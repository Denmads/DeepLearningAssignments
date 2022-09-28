from functools import partial
from Framework.pipeline import Pipeline
from Framework.tasks.loadtask import LoadNpzTask
from Framework.tasks.split_data_task import SplitDataTask
from Framework.tasks.visualizetask import VisualizeSamplesTask
from Framework.tasks.checkpointtask import DataCheckpointTask
from Framework.tasks.subsampletask import SubSampleTask
from Framework.tasks.reshapesamplestask import ConvertSamplesToRowsTask
from Framework.tasks.trainmodeltask import TrainModelTask
from Framework.tasks.assignment1.visualize_knn_distances_task import VisualizeKnnDistancesTask
from Framework.tasks.modelpredicttask import ModelPredictTask
from Framework.tasks.calculteaccuracytask import CalculateAccuracyTask
from Framework.tasks.customtask import CustomTask
from Framework.tasks.flow.forlooptask import ForLoopTask
from Framework.tasks.flow.foreachlooptask import ForEachLoopTask
from Framework.tasks.kfoldsplittask import KfoldSplitTask
from Framework.tasks.shufflesamplestask import ShuffleSamplesTask
from Framework.tasks.addtooutputtask import AddToOutputTask
from Framework.tasks.preprocesstask import CalculateMeanTask, SubtractMeanTask, AppendBiasTask

from external.linear_svm import svm_loss_naive, svm_loss_vectorized
from external.gradient_check import grad_check_sparse

import matplotlib.pyplot as plt
import numpy as np
import time

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'

############################# VARIABLES #######################################################

####### DATA #######
RAW_TRAIN = "raw_train"
RAW_VAL = "raw_val"
RAW_TEST = "raw_test"

TRAIN_SAMPLES = "train_samples"
TRAIN_LABELS = "train_labels"
TRAIN_META = "train_meta"

VAL_SAMPLES = "test_samples"
VAL_LABELS = "test_labels"
VAL_META = "test_meta"

TEST_SAMPLES = "test_samples"
TEST_LABELS = "test_labels"
TEST_META = "test_meta"

####### PREPROCESS #######
MEAN = "mean"



TRAIN_FOLDS_TRAIN = "train_folds_train"
TRAIN_FOLDS_LABELS = "train_folds_labels"

CROSS_TRAIN = "cross_train"
CROSS_TRAIN_LABELS = "cross_train_labels"
CROSS_VAL = "cross_val"
CROSS_VAL_LABELS = "cross_val_labels"
CROSS_PREDICTED = "cross_predicted"

ACCURACY_OUT = "accuracy_out"
ACCURACIES = "accuracies"

KNN_CLASSIFIER = "knn"
PREDICTED_LABELS_1 = "predicition"
PREDICTED_LABELS_5 = "prediction_5"

LOOP_VAR = "i"
LOOP_VAR2 = "k"

K_VALUES = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]
K_MAP = "k_map"
PARAMS = {}




############################# FUNCTIONS #######################################################

def convert_labels_to_int(labels_var, data_store, create_resource_path):
    labels = data_store[labels_var]
    labels = np.array([int(i) for i in labels])
    data_store[labels_var] = labels

def visualize_mean(mean_var, data_store, create_resource_path, save_file: str = None):
    plt.clf()
    mean = data_store[mean_var]
    plt.figure(figsize=(4,4))
    plt.imshow(mean.reshape((48,48,3)).astype('uint8')) # visualize the mean image

    if save_file is None:
        plt.show()
    else:
        plt.savefig(create_resource_path(save_file), bbox_inches="tight")

def eval_loss_func(loss_func, samples_var, labels_var, regularization, data_store, create_resource_path):
    samples = data_store[samples_var]
    labels = data_store[labels_var]

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(6913, 3) * 0.0001 

    loss, grad = loss_func(W, samples, labels, regularization)
    print('Loss: %f' % (loss, ))

def eval_loss_func_against_numeric(loss_func, samples_var, labels_var, data_store, create_resource_path):
    samples = data_store[samples_var]
    labels = data_store[labels_var]

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(6913, 3) * 0.0001 

    # Analytical gradient
    loss, grad = loss_func(W, samples, labels, 0.0)
    
    # Numerical gradient
    f = lambda w: svm_loss_naive(w, samples, labels, 0.0)[0]
    grad_numerical = grad_check_sparse(f, W, grad)

    # With regularization
    loss, grad = svm_loss_naive(W, samples, labels, 5e1)
    f = lambda w: svm_loss_naive(w, samples, labels, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad)

def eval_loss_func_performance(samples_var, labels_var, data_store, create_resource_path):
    samples = data_store[samples_var]
    labels = data_store[labels_var]

    W = np.random.randn(6913, 3) * 0.0001 

    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, samples, labels, 0.000005)
    toc = time.time()
    print('Naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, _ = svm_loss_vectorized(W, samples, labels, 0.000005)
    toc = time.time()
    print('Vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.
    print('difference: %f' % (loss_naive - loss_vectorized))

# def create_validate_sets(loop_var, folds_var, cross_train_var, cross_val_var, data_store, create_resource_path):
#     index = data_store[loop_var]
#     folds = data_store[folds_var]
#     cross_train_sets = list(map(lambda x: x[1], filter(lambda x: x[0] != index, enumerate(folds))))
#     cross_train_set = np.concatenate(cross_train_sets)
#     cross_val_set = folds[index]

#     data_store[cross_train_var] = cross_train_set
#     data_store[cross_val_var] = cross_val_set


# def create_prediction_param_dict(param_dict, data_store, create_resource_path):
#     param_dict["k"] = data_store[LOOP_VAR2]

# def create_and_append_accuracies(acc_var, result_var, data_store, create_resource_path):
#     if result_var not in data_store:
#         accs = []
#         data_store[result_var] = accs
#     else:
#         accs = data_store[result_var]
    
#     accuracy = data_store[acc_var]
#     accs.append(accuracy)


# def calculate_avg_accuracy(accuracies_var, k_var, acc_map_var, data_store, create_resource_path):
#     accs = data_store[accuracies_var]
#     sm = 0
#     for ac in accs:
#         sm += ac
#     acc = sm / len(accs)

#     k = data_store[k_var]
#     print(f"k({k}) - {acc:.4%}")
    
#     if acc_map_var not in data_store:
#         acc_map = {}
#         data_store[acc_map_var] = acc_map
#     else:
#         acc_map = data_store[acc_map_var]
    
#     acc_map[k] = accs
#     del data_store[accuracies_var]

# def visualize_accuracies_map(acc_map_var, data_store, create_resource_path, save_file: str = None):
#     plt.clf()
#     k_to_accuracies = data_store[acc_map_var]

#     for k in K_VALUES:
#         accuracies = k_to_accuracies[k]
#         plt.scatter([k] * len(accuracies), accuracies)

#     # plot the trend line with error bars that correspond to standard deviation
#     accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
#     accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
#     plt.errorbar(K_VALUES, accuracies_mean, yerr=accuracies_std)
#     plt.title('Cross-validation on k')
#     plt.xlabel('k')
#     plt.ylabel('Cross-validation accuracy')

#     print("Visualized Accuracies")
#     if save_file is None:
#         plt.show()
#     else:
#         plt.savefig(create_resource_path(save_file), bbox_inches="tight")

############################# PIPELINE #######################################################

pipeline = Pipeline(
    ############ LOAD DATA ####################
    LoadNpzTask(path_to_file="data/assignment1/HAM_train.npz", output_name=RAW_TRAIN),
    LoadNpzTask(path_to_file="data/assignment1/HAM_val.npz", output_name=RAW_VAL),
    LoadNpzTask(path_to_file="data/assignment1/HAM_test.npz", output_name=RAW_TEST),

    ############ PREPROCESS ####################
    SplitDataTask(RAW_TRAIN, TRAIN_SAMPLES, TRAIN_LABELS, TRAIN_META),
    SplitDataTask(RAW_VAL, VAL_SAMPLES, VAL_LABELS, VAL_META),
    SplitDataTask(RAW_TEST, TEST_SAMPLES, TEST_LABELS, TEST_META),
    ShuffleSamplesTask(TRAIN_SAMPLES, TRAIN_LABELS),
    ShuffleSamplesTask(VAL_SAMPLES, VAL_LABELS),
    ShuffleSamplesTask(TEST_SAMPLES, TEST_LABELS),
    CustomTask(partial(convert_labels_to_int, TRAIN_LABELS)),
    CustomTask(partial(convert_labels_to_int, VAL_LABELS)),
    CustomTask(partial(convert_labels_to_int, TEST_LABELS)),
    DataCheckpointTask(),
    #VisualizeSamplesTask(TRAIN_META, TRAIN_SAMPLES, 
    #    {'bkl':'benign keratosis-like lesions', 
    #        'vasc':'vascular lesions', 
    #        'nv':'melanocytic nevi'}, 7,
    #    "samples.png"
    #),
    SubSampleTask(TRAIN_SAMPLES, TRAIN_SAMPLES, 5000),
    SubSampleTask(TRAIN_LABELS, TRAIN_LABELS, 5000),
    SubSampleTask(VAL_SAMPLES, VAL_SAMPLES, 500),
    SubSampleTask(VAL_LABELS, VAL_LABELS, 500),
    ConvertSamplesToRowsTask(TRAIN_SAMPLES, TRAIN_SAMPLES),
    ConvertSamplesToRowsTask(VAL_SAMPLES, VAL_SAMPLES),
    ConvertSamplesToRowsTask(TEST_SAMPLES, TEST_SAMPLES),

    ############ Preprocess ####################
    CalculateMeanTask(TRAIN_SAMPLES, MEAN),
    CustomTask(partial(visualize_mean, MEAN, save_file="mean.png")),
    SubtractMeanTask(MEAN, [TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES]),
    AppendBiasTask([TRAIN_SAMPLES, VAL_SAMPLES, TEST_SAMPLES]),

    ############ Loss Func Eval ####################
    CustomTask(partial(eval_loss_func, svm_loss_naive, TRAIN_SAMPLES, TRAIN_LABELS, 0.000005)),
    CustomTask(partial(eval_loss_func_against_numeric, svm_loss_naive, TRAIN_SAMPLES, TRAIN_LABELS)),
    CustomTask(partial(eval_loss_func_performance, TRAIN_SAMPLES, TRAIN_LABELS)),

    ############ Train KNN ####################
    # TrainModelTask(KnnClassifier(), KNN_CLASSIFIER, TRAIN_SAMPLES, TRAIN_LABELS),
    # DataCheckpointTask(),
    # VisualizeKnnDistancesTask(KNN_CLASSIFIER, VAL_SAMPLES, "distances.png"),

    # ############ KNN Prediction ####################
    # ModelPredictTask(KNN_CLASSIFIER, VAL_SAMPLES, PREDICTED_LABELS_1, {"k": 1}),
    # DataCheckpointTask(),
    # ModelPredictTask(KNN_CLASSIFIER, VAL_SAMPLES, PREDICTED_LABELS_5, {"k": 5}),
    # DataCheckpointTask(),
    # CalculateAccuracyTask(PREDICTED_LABELS_1, VAL_LABELS, ACCURACY_OUT),
    # AddToOutputTask("k1_accuracy", ACCURACY_OUT),
    # CalculateAccuracyTask(PREDICTED_LABELS_5, VAL_LABELS, ACCURACY_OUT),
    # AddToOutputTask("k5_accuracy", ACCURACY_OUT),

    # ############ CROSS VALIDATION ####################
    # KfoldSplitTask(TRAIN_SAMPLES, TRAIN_FOLDS_TRAIN, 5),
    # KfoldSplitTask(TRAIN_LABELS, TRAIN_FOLDS_LABELS, 5),
    # ForEachLoopTask([
    #     CustomTask(partial(create_prediction_param_dict, PARAMS)),
    #     ForLoopTask([
    #         CustomTask(partial(create_validate_sets, 
    #             LOOP_VAR, 
    #             TRAIN_FOLDS_TRAIN, 
    #             CROSS_TRAIN, 
    #             CROSS_VAL)
    #         ),
    #         CustomTask(partial(create_validate_sets, 
    #             LOOP_VAR, 
    #             TRAIN_FOLDS_LABELS, 
    #             CROSS_TRAIN_LABELS, 
    #             CROSS_VAL_LABELS)
    #         ),
    #         TrainModelTask(KnnClassifier(), KNN_CLASSIFIER, CROSS_TRAIN, CROSS_TRAIN_LABELS),
    #         ModelPredictTask(KNN_CLASSIFIER, CROSS_VAL, CROSS_PREDICTED, PARAMS),
    #         CalculateAccuracyTask(CROSS_PREDICTED, CROSS_VAL_LABELS, ACCURACY_OUT),
    #         CustomTask(partial(create_and_append_accuracies, ACCURACY_OUT, ACCURACIES))
    #     ], LOOP_VAR, max_value=5),
    #     CustomTask(partial(calculate_avg_accuracy, ACCURACIES, LOOP_VAR2, K_MAP))
    # ], LOOP_VAR2, K_VALUES),
    # AddToOutputTask("cross_validation", K_MAP),
    # CustomTask(partial(visualize_accuracies_map, K_MAP, save_file="cross-validation.png")),
    pipeline_path="results/assign1/svm"
)

pipeline.run()
