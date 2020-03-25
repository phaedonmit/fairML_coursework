
#ignore deprection warnings:
import warnings
warnings.simplefilter("ignore", category=FutureWarning) 

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.metrics import ClassificationMetric
from aif360.metrics import BinaryLabelDatasetMetric

# set random seed for repeatable results
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
np.random.seed(0)


def fit_classifier(classifier, weights, lambda_values, X_train, y_train, X_test, y_test, test_pred):
    '''
    Function to fit classifiers for range of Lambda values
    
    Args:
        classifier: SVM or Logistic regression
        weights: weights for each sample
        lambda_values: range of lambda values to assess
        X_train: training data
        y_train: training lables
        X_test: test data
        y_test: test labels
        test_pred: prepared format to store predictions

    Returns: 
        accuracy_list: test accuracy for each model
        equal_opp_list: Equal Opportunity difference for each model
        stat_parity_list: Statistical Parity difference for each model
    '''
    accuracy_list = []
    equal_opp_list = []
    stat_parity_list = []

    for l in lambda_values:
        print("-------- \n", 'Lambda: ', "{0:.2f}".format(l))
        if classifier == "Logistic Regression":
            learner = LogisticRegression(solver='liblinear', random_state=1, penalty='l2', C=1/l)  
        else:
            learner = svm.SVC(C=1/l)  
        learner.fit(X_train,y_train, sample_weight=weights)
        test_pred.labels = learner.predict(X_test)
        metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        print("Equal opportunity:", "{0:.3f}".format(metric.equal_opportunity_difference()))
        print("Statistical parity:", "{0:.3f}".format(metric.statistical_parity_difference()))
        print("Accuracy:", "{0:.3f}".format(metric.accuracy()))
        accuracy_list.append(metric.accuracy())
        equal_opp_list.append(metric.equal_opportunity_difference())
        stat_parity_list.append(metric.statistical_parity_difference())

    return accuracy_list, equal_opp_list, stat_parity_list


def plot_analysis(filename, lambda_values, accuracy_list, fairness_metric_list, fairness_metric):
    '''
    Function to plot graph with performance of different classifiers
    
    Args: 
        filename: name of file to store the plot
        accuracy_list: test accuracy for each model
        fairness_metric_list: Fairness metric list for each model
        fairness_metric: Fairness metric label
    '''    
    plt.xscale("log")
    ax = sns.lineplot(x=lambda_values, y=fairness_metric_list, label=fairness_metric)
    ax2 = ax.twinx()
    sns.lineplot(x=lambda_values, y=accuracy_list, color="g", ax=ax2, label="Accuracy")
    ax.figure.legend()
    ax.set_xlabel(r'Lambda ($\lambda$)')
    ax.set_ylabel(fairness_metric)
    ax2.set_ylabel(r'Accuracy')
    ax.legend().remove()
    ax2.legend().remove()
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], 8))
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], 8))
    ax.set_yticklabels(['{:.3f}'.format(float(t)) for t in ax.get_yticks()])
    ax2.set_yticklabels(['{:.3f}'.format(float(t)) for t in ax2.get_yticks()])
    plt.savefig(f'{filename}.png')    
    plt.show()


def k_fold_statistics(k_folds, classifier, lambda_value, dataset, unprivileged_groups, privileged_groups):
    '''
    Function to fit classifier to k number of random train/test splits
    
    Args:
        k_folds: number of folds of statistics
        classifier: SVM or Logistic regression
        weights: weights for each sample
        lambda_value: selected level of regularisation
        dataset: dataset to be used

    Returns: 
        accuracy_list: test accuracy for each model
        equal_opp_list: Equal Opportunity difference for each model
        stat_parity_list: Statistical Parity difference for each model
    '''
    accuracy_list = []
    equal_opp_list = []
    stat_parity_list = []

    for k in range(k_folds):
        train, test = dataset_orig.split([0.7], shuffle=True)
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(train.features)
        y_train = train.labels.ravel()
        X_test = scale_orig.transform(test.features)
        y_test = test.labels.ravel()
        test_pred = test.copy() 

        RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
        train = RW.fit_transform(train)

        if classifier == "Logistic Regression":
            learner = LogisticRegression(solver='liblinear', random_state=1, penalty='l2', C=1/lambda_value)  
        else:
            learner = svm.SVC(C=1/lambda_value)  
        learner.fit(X_train,y_train, sample_weight=train.instance_weights)
        test_pred.labels = learner.predict(X_test)
        metric = ClassificationMetric(test, test_pred, unprivileged_groups=unprivileged_groups,
                                        privileged_groups=privileged_groups)
        print("----------------")
        print(f'Split {k}/{k_folds}')
        print("Equal opportunity:", "{0:.3f}".format(metric.equal_opportunity_difference()))
        print("Statistical parity:", "{0:.3f}".format(metric.statistical_parity_difference()))
        print("Accuracy:", "{0:.3f}".format(metric.accuracy()))
        accuracy_list.append(metric.accuracy())
        equal_opp_list.append(metric.equal_opportunity_difference())
        stat_parity_list.append(metric.statistical_parity_difference())

    accuracy_list = np.array(accuracy_list)
    equal_opp_list = np.array(equal_opp_list)
    stat_parity_list = np.array(stat_parity_list)
    print(f'The mean accuracy for lambda={lambda_value:.3f} is: \n Mean Accuracy: {np.mean(accuracy_list):.3f}, Std: {np.std(accuracy_list):.3f} \n Mean Equal Opportunity: {np.mean(equal_opp_list):.3f}, Std: {np.std(equal_opp_list):.3f}')
    
    return accuracy_list, equal_opp_list, stat_parity_list


if __name__ == "__main__":

    #*******************    
    # Step 1 - Get data
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]
    dataset_orig = load_preproc_data_adult(['sex'])
    dataset_orig = load_preproc_data_compas(['sex'])
    print(dataset_orig)

    #*******************
    # Step 2 - Split into train and test and normalise
    train, test = dataset_orig.split([0.7], shuffle=True)
    print(train.feature_names)
    # print(train.labels)
    scale_orig = StandardScaler()
    X_train = scale_orig.fit_transform(train.features)
    y_train = train.labels.ravel()
    X_test = scale_orig.transform(test.features)
    y_test = test.labels.ravel()
    test_pred = test.copy()

    #*******************
    # Step 3 - Train machine Learning classifier and plot results
    # lambda_values = np.logspace(0,10, num=50)
    lambda_values = np.logspace(0,2, num=10)
    # accuracy_list, equal_opp_list, stat_parity_list = fit_classifier("SVM", train.instance_weights, lambda_values, 
    #                                                 X_train, y_train, X_test, y_test, test_pred)
    # plot_analysis("unweighted_SVM_equal", lambda_values, accuracy_list, equal_opp_list, "Equal Opport. Difference")
    # plot_analysis("unweighted_SVM_stat", lambda_values, accuracy_list, stat_parity_list, "Statistical Parity")

    #*******************
    # Step 5 - Perform Reweighing, fit classifiers and plot results
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    train = RW.fit_transform(train)
    # accuracy_list, equal_opp_list, stat_parity_list = fit_classifier("SVM", train.instance_weights, lambda_values, 
    #                                                 X_train, y_train, X_test, y_test, test_pred)
    # plot_analysis("weighted_SVM_equal", lambda_values, accuracy_list, equal_opp_list, "Equal Opport. Difference")
    # plot_analysis("weighted_SVM_stat", lambda_values, accuracy_list, stat_parity_list, "Statistical Parity")
    # Plot weight values after re-weighting samples
    # ax = sns.scatterplot(x=np.arange(len(train.instance_weights)),y=train.instance_weights,hue=y_train,
    #                     s=25)
    # ax = sns.distplot(train.instance_weights, kde=False)
    ax = sns.distplot(train.features.ravel(), kde=False)
    ax.set_xlabel(r'Range of Weight')
    ax.set_ylabel('Frequency')
    plt.savefig('reweighted.png')    
    plt.show()
    custom_weights = []
    for sample in X_train:
        print(sample)
        break
    exit()

    #*******************
    # Step 6 - Perform 5 random  train/test splits and report results
    accuracy_list, equal_opp_list, stat_parity_list = k_fold_statistics(10, "SVM", 10, dataset_orig, unprivileged_groups, privileged_groups)
    ax = sns.distplot(equal_opp_list, bins=40)
    ax.set_xlabel(r'Equality of Opportunity Difference')
    ax.set_ylabel('Frequency')
    plt.savefig('kfold.png')    
    plt.show()
