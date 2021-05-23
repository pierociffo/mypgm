import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def action_table(variables):
    #  create the table of all possible actions given evidences
    iterables = []
    columns = []
    
    for v in variables:
        iterables.append(v.labels)
        columns.append(str(v.name))

    rows = []
    for t in itertools.product(*iterables):
        rows.append(t)
    
    return pd.DataFrame(data=np.array(rows), columns=columns)

def best_actions(best_policy):
    #  create the table of optimal actions given the policy
    vrs = best_policy.scope[1:]
    d = best_policy.scope[0]
    
    iterables = []
    columns = []
    
    for v in vrs:
        iterables.append(v.labels)
        columns.append(str(v.name))
        
    rows = []
    for t in itertools.product(*iterables):
        evidence = []
        for v, assg in zip(vrs, list(t)):
            evidence.append((v, assg))
            
        dec = best_policy.reduce(evidence).scope[0].label_dict[int(np.argwhere(best_policy.reduce(evidence).values==1))]
        rows.append([*t, dec])
        
    columns.append(str(d.name))
    
    return pd.DataFrame(data=np.array(rows), columns=columns)

def add_label(df, optimal):
    # find action that are optimal adding a label
    to_check = df.to_dict('records')
    _df = []
    optimal_scope = set()

    for dec in optimal:
        _df.append(best_actions(dec))
        for var in dec.scope:
            optimal_scope.add(var)

    optimal_df = pd.merge(_df[0], _df[1])

    for f in _df[2:]:
        optimal_df = pd.merge(optimal_df, f)

    truth = optimal_df.to_dict('records')
    
    new_col = list(np.zeros((len(to_check))))
    for i in range(len(to_check)):
        filtered = { your_key.name: to_check[i][your_key.name] for your_key in list(optimal_scope) }
        for j in range(len(truth)):
            if truth[j] == filtered:
                new_col[i] = 1
                break
                

    df['Optimal'] = pd.Series(new_col)
    
    return df

from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                             confusion_matrix)

def get_auc_scores(clf, X_train, X_test, y_train, y_test):

 
    y_train_score = clf.predict_proba(X_train)[:, 1]
    y_test_score = clf.predict_proba(X_test)[:, 1]
    auc_train = roc_auc_score(y_train, y_train_score)
    auc_test = roc_auc_score(y_test, y_test_score)
    print(f'Training AUC: {auc_train}')
    print(f'Testing AUC: {auc_test}')
 
    return y_test_score

def plot_roc_curve(y_test, y_test_score_l, clfs): 
    plt.figure(figsize=(10,9))
    for y_test_score, clf in zip(y_test_score_l, clfs):
        fpr, tpr, _ = roc_curve(y_test, y_test_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, '--', label='ROC curve, area = {}, {}'.format(roc_auc, clf))

    plt.plot([0, 1], [0, 1], '-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc='lower right')
    plt.show()
    
    
def show_cm(y_true, y_pred, class_names=None, model_name=None):
 
    cf = confusion_matrix(y_true, y_pred)
    plt.imshow(cf, cmap=plt.cm.Blues)
    if model_name:
        plt.title('Confusion Matrix: {}'.format(model_name))
    else:
        plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    else:
        class_names = set(y_true)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
    thresh = cf.max() / 2.0
    for i, j in itertools.product(range(cf.shape[0]),
                               range(cf.shape[1])):
        plt.text(j, i, cf[i, j],
            horizontalalignment='center',
            color='white' if cf[i, j] > thresh else 'black',
            )
    plt.colorbar()