import light_format_feature_loader
import network
import glob
import os

result_log_file = '../data/tmp_result_log_file.txt'
log_fp = open(result_log_file, 'w')

rainfall_min_value = 0
rainfall_max_value = 100
n_rainfall_indexes = 100 # for use in vectorized form of rainfall label (number of neurons for output layer)
size_hidden_layer = 30 # Number of neurons for hidden layer

# training_data_dir = "/home/awips/sample_data/svmdata.oneHourPrecipAsLabel/svm_case1_merged"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case1_merged_all"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case1_merged"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case1_merged_typhoon"

# training_data_dir = "/home/awips/sample_data/svmdata.oneHourPrecipAsLabel/svm_case2_merged"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case2_merged_all"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case2_merged"
# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case2_merged_typhoon"

# training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case3_merged_all"
training_data_dir = "/home/awips/sample_data/svmdata.10minPrecipAsLabel/svm_case3_merged_all_middle_value_only"

filepath_list = glob.glob(training_data_dir + "/svm_traindata.txt.3a.*")

filename_list = []
print "==== Training file list ===="
for filepath in filepath_list:
    if os.path.basename(filepath).find("log") == -1:
        filename_list.append(os.path.basename(filepath))

# Manual setting of training files
#filename_list = []
#filename_list.append("svm_traindata.txt.1a.Z_0.5_0.5")
#filename_list.append("svm_traindata.txt.1a.V_0.5_0.5")
#filename_list.append("svm_traindata.txt.2b.9999.0_-9999.0")
log_fp.write("===== Training file list =====\n")
for filename in filename_list:
    print filename
    log_fp.write(filename + "\n")
print ""

"""
Hyper-parameters
"""
epoches = 30
mini_batch_size = 10
training_rate = 2.0

log_fp.write("training_data_dir: {0}\n".format(training_data_dir))
log_fp.write("rainfall_min_value: {0}\n".format(rainfall_min_value))
log_fp.write("rainfall_max_value: {0}\n".format(rainfall_max_value))
log_fp.write("n_rainfall_indexes(Number of neurons for output-layer): {0}\n".format(n_rainfall_indexes))
log_fp.write("size_hidden_layer(Number of neurons for hidden-layer): {0}\n".format(size_hidden_layer))
log_fp.write("kernel type: Sigmoid\n")
log_fp.write("epoches: {0}\n".format(epoches))
log_fp.write("mini_batch_size: {0}\n".format(mini_batch_size))
log_fp.write("training_rate: {0}\n".format(training_rate))

"""
Train and evaluate the network for the given list of training files
"""
for filename in filename_list:
    training_file = training_data_dir + "/" + filename
    training_data, validation_data, test_data, dimension = \
    light_format_feature_loader.load_data_wrapper(training_file, rainfall_min_value, rainfall_max_value, n_rainfall_indexes)

    if training_data == None:
        print "[ERROR] Training data in {0} is None, skipped training task".format(training_file)
        continue

    log_fp.write("===== {0} =====\n".format(filename));
    log_fp.write("dimension(Number of neurons for input layer) : {0}\n".format(dimension))
    log_fp.write("number of training examples: {0}  test examples: {1}\n".format(len(training_data), len(test_data)))

    """ 
    Build a three-layer neural network given the number of neurons for each layer
    Parameters:
    [number of neurons for input layer], [number of neurons for hidden layer], [number of neurons for output layer]
    """
    net = network.Network([dimension, size_hidden_layer, n_rainfall_indexes])

    """
    Perform stochastic gradient descent algorithm given necessary hyper-parameters
    Parameters:
    [training_data], [number of epoches], [mini-batch size], [training rate], [test_data]
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    """
    print "===== Start training  for {0} =====".format(filename)
    net.SGD(training_data, epoches, mini_batch_size, training_rate, log_file_pointer=log_fp, test_data=test_data)
    correctly_classified = net.evaluate(test_data)
    print "Result after {0} epoches: {1} / {2}  ({3}%)\n".format(
                    epoches, correctly_classified, len(test_data), 100*correctly_classified/len(test_data))
    log_fp.write("Result after {0} epoches: {1} / {2}  ({3}%)\n".format(
                    epoches, correctly_classified, len(test_data), 100*correctly_classified/len(test_data)))
    log_fp.write("\n")

log_fp.close()

