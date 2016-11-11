import numpy, nltk


def make_kfold(data, k, total_fold):
    """
    returns two lists, the k-th fold of k total folds and the rest of the data
    """
    fold_length = int(len(data)/total_fold)
    fold_start = k * fold_length
    fold_end = (k + 1) * fold_length
    # returns TEST, TRAIN
    return data[fold_start:fold_end], data[:fold_start] + data[fold_end:] 

def cross_validation(data, folds, verbose = True):
    """
    Run a k-fold cross validation on the 'data'
    We split the data up into 'folds' chunks, train on all but 1 chunk and then
    test on the remaining chunk

    we return the average and the variance for the accuracy of the runs
    """
    runs = []
    if verbose:
        print("Doing a", str(folds) + "-fold cross validation on", len(data), \
            "data.")
    for k in range(folds):    
    #split into test and train sets
        test, train = make_kfold(data,k,folds)        
        
        b_class = nltk.NaiveBayesClassifier.train(train)
        runs.append(nltk.classify.accuracy(b_class, test))
        if verbose:
            print("\tFold", k, " :: ", runs[-1])
    if verbose:
        print("Testing complete:\n\tMean:", numpy.mean(runs), "\tVariance:", \
            numpy.var(runs))
    return numpy.mean(runs), numpy.var(runs)