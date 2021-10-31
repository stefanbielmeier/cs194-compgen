#!/usr/bin/env python3
#
# This code has been produced by a free evaluation version of Brainome(tm).
# Portions of this code copyright (c) 2019-2021 by Brainome, Inc. All Rights Reserved.
# Brainome, Inc grants an exclusive (subject to our continuing rights to use and modify models),
# worldwide, non-sublicensable, and non-transferable limited license to use and modify this
# predictor produced through the input of your data:
# (i) for users accessing the service through a free evaluation account, solely for your
# own non-commercial purposes, including for the purpose of evaluating this service, and
# (ii) for users accessing the service through a paid, commercial use account, for your
# own internal  and commercial purposes.
# Please contact support@brainome.ai with any questions.
# Use of predictions results at your own risk.
#
# Output of Brainome v1.006-19-prod.
# Invocation: brainome dataset.csv -headerless -f NN
# Total compiler execution time: 0:01:35.54. Finished on: Oct-31-2021 13:47:27.
# This source code requires Python 3.
#
"""

[01;1mPredictor:[0m                        a.py
    Classifier Type:              Neural Network
    System Type:                  Binary classifier
    Training / Validation Split:  60% : 40%
    Accuracy:
      Best-guess accuracy:        54.17%
      Training accuracy:          67.37% (18167/26964 correct)
      Validation Accuracy:        66.88% (12024/17978 correct)
      Combined Model Accuracy:    67.17% (30191/44942 correct)

    Model Capacity (MEC):         16    bits

    Generalization Ratio:       1129.72 bits/bit
    Percent of Data Memorized:     0.18%
    Resilience to Noise:          -3.06 dB


    Training Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   9352   5255 
                   1 |   3542   8815 

    Validation Confusion Matrix:
              Actual | Predicted
              ------ | ---------
                   0 |   6163   3576 
                   1 |   2378   5861 

    Training Accuracy by Class:
              target |     TP     FP     TN     FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ------ ------ ------ ------ -------- -------- -------- -------- -------- --------
                   0 |   9352   3542   8815   5255   64.02%   71.34%   72.53%   62.65%   68.01%   51.53%
                   1 |   8815   5255   9352   3542   71.34%   64.02%   62.65%   72.53%   66.71%   50.05%

    Validation Accuracy by Class:
              target |     TP     FP     TN     FN     TPR      TNR      PPV      NPV       F1       TS 
              ------ | ------ ------ ------ ------ -------- -------- -------- -------- -------- --------
                   0 |   6163   2378   5861   3576   63.28%   71.14%   72.16%   62.11%   67.43%   50.86%
                   1 |   5861   3576   6163   2378   71.14%   63.28%   62.11%   72.16%   66.32%   49.61%




"""

import sys
import math
import os
import argparse
import tempfile
import csv
import binascii
import faulthandler
import json
from io import StringIO
try:
    import numpy as np # For numpy see: http://numpy.org
    from numpy import array
except:
    print("This predictor requires the Numpy library. Please run 'python3 -m pip install numpy'.")
    sys.exit(1)
try:
    from scipy.sparse import coo_matrix
    report_cmat = True
except:
    print("Note: If you install scipy (https://www.scipy.org) this predictor generates a confusion matrix. Try 'python3 -m pip install scipy'.")
    report_cmat = False

IOBUF = 100000000
sys.setrecursionlimit(1000000)
TRAINFILE = ['dataset.csv']
mapping = {'0': 0, '1': 1}
ignorelabels = []
ignorecolumns = []
target = '' 
target_column = 3
important_idxs = [0, 1, 2]
ignore_idxs = []
classifier_type = 'NN'
num_attr = 3
n_classes = 2
model_cap = 16
w_h = np.array([[-0.14179708063602448, 0.06099817901849747, 14.083345413208008], [-0.7359390258789062, -0.3851543664932251, 0.26815736293792725], [0.2221897840499878, 0.08819884061813354, -13.726582527160645]])
b_h = np.array([-0.11618394404649734, -0.06432671844959259, -0.07004344463348389])
w_o = np.array([[0.05429321900010109, -0.3701339364051819, -0.03716045990586281]])
b_o = np.array(-0.36965423822402954)


def __convert(cell):
    value = str(cell)
    try:
        result = int(value)
        return result
    except ValueError:
        try:
            result=float(value)
            if math.isnan(result):
                print('NaN value found. Aborting.')
                sys.exit(1)
            return result
        except ValueError:
            result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
            return result
        except Exception as e:
            print(f"An exception of type {type(e).__name__} was encountered. Aborting.")
            sys.exit(1)


def __get_key(val, dictionary):
    if dictionary == {}:
        return val
    for key, value in dictionary.items(): 
        if val == value:
            return key
    if val not in dictionary.values:
        print("Label key does not exist")
        sys.exit(1)


def __convertclassid(cell, classlist=[]):

    value = str(cell)
    
    if value == '':
        print('Empty value encountered for a class label. Aborting.')
        sys.exit(1)
    
    if mapping != {}:
        result = -1
        try:
            result = mapping[cell]
        except KeyError:
            print(f"The class label {value} does not exist in the class mapping. Aborting.")
            sys.exit(1)
        except Exception as e:
            print(f"An exception of type {type(e).__name__} was encountered. Aborting.")
            sys.exit(1)
        if result != int(result):
            print(f"The label {value} is mapped to {result} but class labels must be mapped to integers. Aborting.")
            sys.exit(1)
        if str(result) not in classlist:
            classlist.append(str(result))
        return result
    
    try:
        result = float(cell)
        if str(result) not in classlist:
            classlist.append(str(result))
    except:
        result = (binascii.crc32(value.encode('utf8')) % (1 << 32))
        if result in classlist:
            result = classlist.index(result)
        else:
            classlist.append(str(result))
            result = classlist.index(result)
        if result != int(result):
            print(f"The label {value} is mapped to {result} but class labels must be mapped to integers. Aborting.")
            sys.exit(1)
    finally:
        if result < 0:
            print(f"The label {value} is mapped to {result} but class labels must be mapped to non-negative integers. Aborting.")
            sys.exit(1)

    return result


def __clean(filename, outfile, headerless=False, testfile=False, trim=False):
    classlist = []
    outbuf = []
    remove_bad_chars = lambda x: x.replace('"', '').replace(',', '').replace('(', '').replace(')', '')
    
    with open(filename, encoding='utf-8') as csv_file, open(outfile, "w+", encoding='utf-8') as f:
        
        reader = csv.reader(csv_file)
        if not headerless:
            next(reader, None)
        
        for i, row in enumerate(reader):

            if row == []:
                continue

            
            expected_row_length = len(important_idxs)
            if not trim:
                expected_row_length += len(ignorecolumns)
            if not testfile:
                expected_row_length += 1
            actual_row_length = len(row)

            if testfile and actual_row_length == expected_row_length + 1:
                error_str = f"We found {actual_row_length} columns but expected {expected_row_length} columns at row {i}. "
                error_str += f"Please check that the CSV contains no target column otherwise use -validate. Aborting."
                print(error_str)
                sys.exit(1)
            
            if actual_row_length != expected_row_length:
                print(f"We found {actual_row_length} columns but expected {expected_row_length} columns.")
                sys.exit(1)            

            if testfile:
                if len(row) == 1:
                    converted_row = [str(__convert(remove_bad_chars(row[0])))]
                else:
                    converted_row = [str(__convert(remove_bad_chars(element))) + "," for element in row[:-1]] + [str(__convert(remove_bad_chars(row[-1])))]         
            else:
                converted_row = [str(__convert(remove_bad_chars(element))) + "," for element in row[:-1]] + [str(__convertclassid(row[-1], classlist))]
            outbuf.extend(converted_row)

            if len(outbuf) < IOBUF:
                outbuf.append(os.linesep)
            else:
                print(''.join(outbuf), file=f)
                outbuf = []
        
        print(''.join(outbuf), end="", file=f)

    n_classes_found = len(classlist)
    if not testfile and n_classes_found < 2:
        print(f"Only {n_classes_found} classes were found. Aborting.")
        sys.exit(1)


def __confusion_matrix(y_true, y_pred, json, labels=None, sample_weight=None, normalize=None):
    stats = {}
    if labels is None:
        labels = np.array(list(set(list(y_true.astype('int')))))
    else:
        labels = np.asarray(labels)
        if np.all([l not in y_true for l in labels]):
            raise ValueError("At least one label specified must be in y_true")
    n_labels = labels.size

    for class_i in range(n_labels):
        stats[class_i] = {'TP':{},'FP':{},'FN':{},'TN':{}}
        class_i_indices = np.argwhere(y_true==class_i)
        not_class_i_indices = np.argwhere(y_true!=class_i)
        stats[int(class_i)]['TP'] = int(np.sum(y_pred[class_i_indices] == class_i))
        stats[int(class_i)]['FN'] = int(np.sum(y_pred[class_i_indices] != class_i))
        stats[int(class_i)]['TN'] = int(np.sum(y_pred[not_class_i_indices] != class_i))
        stats[int(class_i)]['FP'] = int(np.sum(y_pred[not_class_i_indices] == class_i))

    if not report_cmat:
        if json:
            return np.array([]), stats
        else:
            sys.exit(0)

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0], dtype=np.int64)
    else:
        sample_weight = np.asarray(sample_weight)
    if y_true.shape[0]!=y_pred.shape[0]:
        raise ValueError("y_true and y_pred must be of the same length")

    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', 'all', None}")


    label_to_ind = {y: x for x, y in enumerate(labels)}
    y_pred = np.array([label_to_ind.get(x, n_labels + 1) for x in y_pred])
    y_true = np.array([label_to_ind.get(x, n_labels + 1) for x in y_true])
    ind = np.logical_and(y_pred < n_labels, y_true < n_labels)
    y_pred = y_pred[ind]
    y_true = y_true[ind]

    sample_weight = sample_weight[ind]
    if sample_weight.dtype.kind in {'i', 'u', 'b'}:
        dtype = np.int64
    else:
        dtype = np.float64
    cm = coo_matrix((sample_weight, (y_true, y_pred)), shape=(n_labels, n_labels), dtype=dtype,).toarray()

    with np.errstate(all='ignore'):
        if normalize == 'true':
            cm = cm / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm / cm.sum()
        cm = np.nan_to_num(cm)
    return cm, stats


def __predict(arr, headerless, csvfile, trim=False):
    with open(csvfile, 'r', encoding='utf-8') as csvinput:
        reader = csv.reader(csvinput)
        if not headerless:
            if trim:
                header = ','.join([x for i, x in enumerate(next(reader, None)) if i in important_idxs] + ['Prediction'])
            else:
                header = ','.join(next(reader, None) + ['Prediction'])
            print(header)
        outputs = __classify(arr)
        for i, row in enumerate(reader):
            pred = str(__get_key(int(outputs[i]), mapping))
            if trim:
                row = ['"' + field + '"' if ',' in field else field for i, field in enumerate(row) if i in important_idxs]
            else:
                row = ['"' + field + '"' if ',' in field else field for field in row]            
            row.append(pred)
            print(','.join(row))


def __preprocess_and_clean_in_memory(arr):
    if not isinstance(arr, list) and not isinstance(arr, np.ndarray):
        print(f'The input to \'predict\' must be a list or np.ndarray but an input of type {type(arr).__name__} was found.')
        sys.exit(1)
    clean_arr = np.zeros((len(arr), len(important_idxs)))
    for i, row in enumerate(arr):
        try:
            row_used_cols_only = [row[i] for i in important_idxs]
        except IndexError:
            error_str = f"The input has shape ({len(arr)}, {len(row)}) but the expected shape is (*, {num_attr})."
            if len(arr) == num_attr and len(arr[0]) != num_attr:
                error_str += "\n\nNote: You may have passed an input directly to 'preprocess_and_clean_in_memory' or 'predict_in_memory' "
                error_str += "rather than as an element of a list. Make sure that even single instances "
                error_str += "are enclosed in a list. Example: predict_in_memory(0) is invalid but "
                error_str += "predict_in_memory([0]) is valid."
            print(error_str)
            sys.exit(1)
        clean_arr[i] = [float(__convert(field)) for field in row_used_cols_only]
    return clean_arr


def __classify(arr, return_probabilities=False):
    h = np.dot(arr, w_h.T) + b_h
    relu = np.maximum(h, np.zeros_like(h))
    out = np.dot(relu, w_o.T) + b_o
    if return_probabilities:
        exp_o = np.zeros((out.shape[0],))
        idxs_negative = np.argwhere(out < 0.).reshape(-1)
        if idxs_negative.shape[0] > 0:
            exp_o[idxs_negative] = 1. - 1. / (1. + np.exp(out[idxs_negative])).reshape(-1)
        idxs_positive = np.argwhere(out >= 0.).reshape(-1)
        if idxs_positive.shape[0] > 0:
            exp_o[idxs_positive] = 1. / (1. + np.exp(-out[idxs_positive])).reshape(-1)
        exp_o = exp_o.reshape(-1, 1)
        output = np.concatenate((1. - exp_o, exp_o), axis=1)
    else:
        output = (out >= 0).astype('int').reshape(-1)
    return output


def __validate_kwargs(kwargs):
    for key in kwargs:
        if key not in ['return_probabilities']:
        
            print(f'{key} is not a keyword argument for Brainome\'s {classifier_type} predictor. Please see the documentation.')
            sys.exit(1)


def predict(arr, remap=True, **kwargs):
    """
    Parameters
    ----------
    arr : list[list]
        An array of inputs to be cleaned by 'preprocess_and_clean_in_memory'.

    remap : bool
        If True and 'return_probs' is False, remaps the output to the original class
        label. If 'return_probs' is True this instead adds a header indicating which
        original class label each column of output corresponds to.
    
    **kwargs :
        return_probabilities : bool
            If true, return class membership probabilities instead of classifications.
        
    Returns
    -------
    output : np.ndarray
        A numpy array of

            1. Class predictions if 'return_probabilities' is False.
            2. Class probabilities if 'return_probabilities' is True.
        """
    kwargs = kwargs or {}
    __validate_kwargs(kwargs)
    remove_bad_chars = lambda x: str(x).replace('"', '').replace(',', '').replace('(', '').replace(')', '')
    arr = [[remove_bad_chars(field) for field in row] for row in arr]
    arr = __preprocess_and_clean_in_memory(arr)
    output = __classify(arr, **kwargs)
    if remap:
        if len(output.shape) > 1: # probabilities were returned
            header = np.array([__get_key(i, mapping) for i in range(output.shape[1])], dtype=str).reshape(1, -1)
            output = np.concatenate((header, output), axis=0)
        else:
            output = np.array([__get_key(prediction, mapping) for prediction in output])
    return output


def validate(cleanarr):
    """
    Parameters
    ----------
    cleanarr : np.ndarray
        An array of float values that has undergone each pre-
        prediction step.

    Returns
    -------
    count : int
        A count of the number of instances in cleanarr.

    correct_count : int
        A count of the number of correctly classified instances in
        cleanarr.

    numeachclass : dict
        A dictionary mapping each class to its number of instances.

    outputs : np.ndarray
        The output of the predictor's '__classify' method on cleanarr.
    """
    outputs = __classify(cleanarr[:, :-1])
    count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0 = 0, 0, 0, 0, 0, 0, 0, 0
    correct_count = int(np.sum(outputs.reshape(-1) == cleanarr[:, -1].reshape(-1)))
    count = outputs.shape[0]
    num_TP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 1)))
    num_TN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 0)))
    num_FN = int(np.sum(np.logical_and(outputs.reshape(-1) == 0, cleanarr[:, -1].reshape(-1) == 1)))
    num_FP = int(np.sum(np.logical_and(outputs.reshape(-1) == 1, cleanarr[:, -1].reshape(-1) == 0)))
    num_class_0 = int(np.sum(cleanarr[:, -1].reshape(-1) == 0))
    num_class_1 = int(np.sum(cleanarr[:, -1].reshape(-1) == 1))
    return count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, outputs
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predictor trained on ' + str(TRAINFILE))
    parser.add_argument('csvfile', type=str, help='CSV file containing test set (unlabeled).')
    parser.add_argument('-validate', action='store_true', help='Validation mode. csvfile must be labeled. Output is classification statistics rather than predictions.')
    parser.add_argument('-headerless', help='Do not treat the first line of csvfile as a header.', action='store_true')
    parser.add_argument('-json', action="store_true", default=False, help="report measurements as json")
    parser.add_argument('-trim', action="store_true", help="If true, the prediction will not output ignored columns.")
    args = parser.parse_args()
    faulthandler.enable()

    if args.validate:
        args.trim = True

    is_testfile = not args.validate
    
    cleanfile = tempfile.NamedTemporaryFile().name
    __clean(args.csvfile, cleanfile, args.headerless, is_testfile, trim=args.trim)
    cleanarr = np.loadtxt(cleanfile, delimiter=',', dtype='float64')
    if len(cleanarr.shape) == 1:
        if args.trim and len(important_idxs) == 1:
            cleanarr = cleanarr.reshape(-1, 1)
        elif len(open(cleanfile, 'r').read().splitlines()) == 1:
            cleanarr = cleanarr.reshape(1, -1)

    if not args.trim and ignorecolumns != []:
        cleanarr = cleanarr[:, important_idxs].reshape(-1, len(important_idxs))

    if not args.validate:
        __predict(cleanarr, args.headerless, args.csvfile, trim=args.trim)
    else:
        count, correct_count, num_TP, num_TN, num_FP, num_FN, num_class_1, num_class_0, preds = validate(cleanarr)
        
        true_labels = cleanarr[:, -1]
        classcounts = np.bincount(cleanarr[:, -1].astype('int32')).reshape(-1)
        classbalance = (classcounts[np.argwhere(classcounts > 0)] / cleanarr.shape[0]).reshape(-1).tolist()
        best_guess = round(100.0 * np.max(classbalance), 2)
        H = float(-1.0 * sum([classbalance[i] * math.log(classbalance[i]) / math.log(2) for i in range(len(classbalance))]))
        modelacc = int(float(correct_count * 10000) / count) / 100.0

        if args.json:
            FN = float(num_FN) * 100.0 / float(count)
            FP = float(num_FP) * 100.0 / float(count)
            TN = float(num_TN) * 100.0 / float(count)
            TP = float(num_TP) * 100.0 / float(count)

            if int(num_TP + num_FN) != 0:
                TPR = num_TP / (num_TP + num_FN)  # Sensitivity, Recall
            if int(num_TN + num_FP) != 0:
                TNR = num_TN / (num_TN + num_FP)  # Specificity
            if int(num_TP + num_FP) != 0:
                PPV = num_TP / (num_TP + num_FP)  # Recall
            if int(num_FN + num_TP) != 0:
                FNR = num_FN / (num_FN + num_TP)  # Miss rate
            if int(2 * num_TP + num_FP + num_FN) != 0:
                FONE = 2 * num_TP / (2 * num_TP + num_FP + num_FN)  # F1 Score
            if int(num_TP + num_FN + num_FP) != 0:
                TS = num_TP / (num_TP + num_FN + num_FP)  # Critical Success Index
            json_dict = {'instance_count': count,
                         'classifier_type': classifier_type,
                         'classes': n_classes,
                         'number_correct': correct_count,
                         'accuracy': {
                             'best_guess': best_guess,
                             'improvement': modelacc - best_guess,
                             'model_accuracy': modelacc,
                         },
                         'false_negative_instances': num_FN,
                         'false_positive_instances': num_FP,
                         'true_positive_instances': num_TP,
                         'true_negative_instances': num_TN,
                         'false_negatives': FN,
                         'false_positives': FP,
                         'true_negatives': TN,
                         'true_positives': TP,
                         'model_capacity': model_cap,
                         'generalization_ratio': int(float(correct_count * 100) / model_cap) / 100.0 * H,
                         'model_efficiency': int(100 * (modelacc - best_guess) / model_cap) / 100.0,
                         'shannon_entropy_of_labels': H,
                         'classbalance': classbalance} 
        else:
            print("Classifier Type:                    Neural Network")
            print(f"System Type:                        {n_classes}-way classifier")
            print()
            print("Accuracy:")
            print("    Best-guess accuracy:            {:.2f}%".format(best_guess))
            print("    Model accuracy:                 {:.2f}%".format(modelacc) + " (" + str(int(correct_count)) + "/" + str(count) + " correct)")
            print("    Improvement over best guess:    {:.2f}%".format(modelacc - best_guess) + " (of possible " + str(round(100 - best_guess, 2)) + "%)")
            print()
            print("Model capacity (MEC):               {:.0f} bits".format(model_cap))
            if classifier_type == '\'NN\'':
                print("Model Capacity Utilized:            {:.0f} bits".format(cap_utilized))  # noqa
            print("Generalization ratio:               {:.2f}".format(int(float(correct_count * 100) / model_cap) / 100.0 * H) + " bits/bit")

        mtrx, stats = __confusion_matrix(np.array(true_labels).reshape(-1), np.array(preds).reshape(-1), args.json)

        if args.json:
            json_dict['confusion_matrix'] = mtrx.tolist()
            json_dict['multiclass_stats'] = stats
            print(json.dumps(json_dict))
        else:
            mtrx = mtrx.astype('str')
            labels = np.array(list(mapping.keys())).reshape(-1, 1)
            mtrx = np.concatenate((labels, mtrx), axis=1).astype('str')
            max_TP_len, max_FP_len, max_TN_len, max_FN_len = 0, 0, 0, 0
            max_class_name_len = len('target') + 2
            for classs in mapping.keys():
                max_class_name_len = max(max_class_name_len, len(classs))
            for key in stats.keys():
                class_stats = stats[key]
                max_TP_len, max_FP_len, max_TN_len, max_FN_len = max(max_TP_len, len(str(class_stats['TP']))), max(max_FP_len, len(str(class_stats['FP']))), max(
                    max_TN_len, len(str(class_stats['TN']))), max(max_FN_len, len(str(class_stats['FN'])))
            print()
            print("Confusion Matrix:")
            print()
            max_len_value = int(np.max(np.vectorize(len)(mtrx)))
            max_pred_len = (int(mtrx.shape[1]) - 1) * max_len_value

            print(" " * 4 + "{:>{}} |{:^{}}".format("Actual", max_class_name_len, "Predicted", max_pred_len))
            print(" " * 4 + "-" * (max_class_name_len + max_pred_len + mtrx.shape[1] + 1))
            for row in mtrx:
                print(str(" " * 4 + "{:>{}}".format(row[0], max_class_name_len)) + " |" + "{:^{}}".format(
                    (' '.join([str('{:>{}}'.format(i, max_len_value)) for i in row[1:]])), max_pred_len))
            print()
            print("Accuracy by Class:")
            print()
            print(" " * 4 + "{:>{}} | {:>{}} {:>{}} {:>{}} {:>{}} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format('target',
                                                                                                              max_class_name_len,
                                                                                                              'TP', max_TP_len,
                                                                                                              'FP', max_FP_len,
                                                                                                              'TN', max_TN_len,
                                                                                                              'FN', max_FN_len,
                                                                                                              'TPR', 'TNR',
                                                                                                              'PPV', 'NPV',
                                                                                                              'F1', 'TS'))
            print(" " * 4 + "-" * max_class_name_len + ' | ' + "-" * (
                max_TP_len) + ' ' + "-" * max_FP_len + ' ' + "-" * max_TN_len + ' ' + "-" * max_FN_len + (' ' + 7 * "-") * 6)
            for raw_class in mapping.keys():
                class_stats = stats[int(mapping[raw_class])]
                TPR = class_stats['TP'] / (class_stats['TP'] + class_stats['FN']) if int(
                    class_stats['TP'] + class_stats['FN']) != 0 else 0
                TNR = class_stats['TN'] / (class_stats['TN'] + class_stats['FP']) if int(
                    class_stats['TN'] + class_stats['FP']) != 0 else 0
                PPV = class_stats['TP'] / (class_stats['TP'] + class_stats['FP']) if int(
                    class_stats['TP'] + class_stats['FP']) != 0 else 0
                NPV = class_stats['TN'] / (class_stats['TN'] + class_stats['FN']) if int(
                    class_stats['TN'] + class_stats['FN']) != 0 else 0
                F1 = 2 * class_stats['TP'] / (2 * class_stats['TP'] + class_stats['FP'] + class_stats['FN']) if int(
                    (2 * class_stats['TP'] + class_stats['FP'] + class_stats['FN'])) != 0 else 0
                TS = class_stats['TP'] / (class_stats['TP'] + class_stats['FP'] + class_stats['FN']) if int(
                    (class_stats['TP'] + class_stats['FP'] + class_stats['FN'])) != 0 else 0
                print(" " * 4 + "{:>{}} | {:>{}} {:>{}} {:>{}} {:>{}} {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}".format(raw_class,
                                                                                                                  max_class_name_len,
                                                                                                                  class_stats['TP'],
                                                                                                                  max_TP_len,
                                                                                                                  class_stats['FP'],
                                                                                                                  max_FP_len,
                                                                                                                  class_stats['TN'],
                                                                                                                  max_TN_len,
                                                                                                                  class_stats['FN'],
                                                                                                                  max_FN_len,
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * TPR, 2)),
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * TNR, 2)),
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * PPV, 2)),
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * NPV, 2)),
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * F1, 2)),
                                                                                                                  "{:0.2f}%".format(
                                                                                                                      round(100.0 * TS, 2))))
            
    os.remove(cleanfile)
    
