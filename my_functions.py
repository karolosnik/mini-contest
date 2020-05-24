#!/usr/bin/env python
# coding: utf-8

# # Confusion Matrix plot

# In[25]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, fig, ax, allClasses,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = allClasses[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #print(cm)

    #fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# In[26]:


import re
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for i, topic in enumerate(model.components_):
        print("Topic #{}:".format(i+1))
        print(' '.join([words[i] for i in topic.argsort()[:-n_top_words - 1:-1]]),'\n')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
# Helper function
def plot_10_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]

    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:10]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15/1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


# # EM GMM

# In[27]:


from matplotlib.patches import Ellipse
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))

def plot_gmm(gmm, X, label=True, ax=None):
    ax = ax or plt.gca()
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)
    ax.axis('equal')

    w_factor = 0.2 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, alpha=w * w_factor)


# ### Graphical methods for GMM (silhouette, BIC)

# In[28]:


def elbow_method_GMM(X, clusters=3, save=False):
    if clusters > 0:
        bic = []
        sil = []
        plt.figure()
        rng = range(2, clusters+1)
        for i in rng:
            gmm = GaussianMixture(n_components=i, covariance_type='full').fit(X)
            bic.append(gmm.bic(X))
            labels = gmm.predict(X)
            sil.append(metrics.silhouette_score(X, labels))
        plt.figure(figsize=(12, 4), dpi= 80, facecolor='w', edgecolor='k')
        # plot 1
        plt.subplot(121)
        plt.plot(rng, bic)
        plt.title('Bic method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Bic')
        plt.xticks(rng)
        # plot 2
        plt.subplot(122)
        plt.plot(rng, sil)
        plt.title('Silhouette coefficient method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette')
        plt.xticks(rng)
        if save:
            plt.savefig('GMM_elbow_plots.png')
        else:
            return plt
    else:
        raise ValueError("Wrong number of clusters. Need smth above 0")


# ### Elbow method kmeans

# In[29]:


#method to compute the optimal number of clusters(graphically)
from sklearn.cluster import KMeans
def elbow_method_kmeans(X, clusters=3, save=False):
    if clusters > 0:
        wcss = []
        plt.figure()
        rng = range(1,clusters+1)
        for i in rng:
            kmeans = KMeans(n_clusters = i, init = 'k-means++')
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, clusters+1), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.xticks(rng)
        if save:
            plt.savefig('kmeans_elbow_plot.png')
        else:
            return plt
    else:
        raise ValueError("Wrong number of clusters. Need smth above 0")


# # 2D-3D plots

# In[ ]:


import matplotlib.pyplot as plt
def PCA_plots(X_3d, labels, title='Kmeans'):
    #2D
    plt.figure()
    ax1 = plt.axes()
    for g in np.unique(labels):
        ix = np.where(labels == g)
        xs = X_3d[ix, 0]
        ys = X_3d[ix, 1]
        ax1.scatter(xs, ys, cmap="viridis", label=g+1 , s=9, alpha=0.7)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(loc = 'upper left')
    plt.title(title + " - 2D projection")
    plt1 = plt
    # 3D
    plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.axes(projection='3d')
    for g in np.unique(labels):
        ix = np.where(labels == g)
        xs = X_3d[ix, 0]
        ys = X_3d[ix, 1]
        zs = X_3d[ix, 2]
        ax.scatter(xs, ys, zs, cmap="viridis", label=g+1 , s=9, alpha=0.6)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(loc = 'lower left')
    plt.title(title + " - 3D projection")
    return plt1, plt


# # Word Counter

# In[21]:


def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# # Classifier evaluation

# In[22]:


def evaluate_classifiers(X_train, X_test, y_train, y_test, models, metric='Accuracy', per_class=False):
    """"
    - Choose metric amongst ["Accuracy", "Recall", "Precision", "F1", "Support"]

    - Models example:
    models = {'linear_svm': SVC(C = 1, kernel='linear', gamma='scale', probability=True),
              'rbf': SVC(C = 1, kernel='rbf', gamma='scale', probability=True),
              'knn': KNeighborsClassifier(n_neighbors=9),
              'NB': GaussianNB(),
              'RF': RandomForestClassifier(max_depth=8, min_samples_leaf=3, min_samples_split=3, n_estimators=600),
              'Logistic': LogisticRegression(),
              'XGB': XGBClassifier(),
              'LGBM': LGBMClassifier()}
    """
    # Create scores dictionary for each algorithm
    scores=[]
    mdl=[]
    results=[]
    type(models)
    for model in models.keys():
        est = models[model]
        est.fit(X_train,  y_train)
        mdl.append(model)
        y_pred = est.predict(X_test)
        results.append((est.score(X_test, y_test), y_pred))
        #print(metrics.classification_report(y_test, y_pred, digits=3))
    results = [dict(zip(['Accuracy','y_pred'], i)) for i in results]

    # At this point "scores" only contains accuracy and y_pred for each one of the best models chosen for each algorithm
    scores = dict(zip(mdl, results))

    # Enrich scores dictionary with the extra metrics
    minRe={}
    maxRe={}
    from sklearn.metrics import precision_recall_fscore_support as classmetrics
    for alg in scores.keys():
        print (alg)
        precision, recall, fscore, support = classmetrics(y_test, scores[alg]['y_pred'])
        scores[alg]['Precision'] = precision
        scores[alg]['Recall'] = recall
        scores[alg]['F1'] = fscore
        scores[alg]['Support']  = support
        print ('{}: {}'.format(metric, scores[alg][metric]))
    if per_class:
        minRe[alg] = np.argmin(scores[alg][metric])
        maxRe[alg] = np.argmax(scores[alg][metric])
        print("\nWorst performance class for each classifier based on {}:".format(metric))
        print(maxRe)
        print("Best performance class for each classifier based on {}:".format(metric))
        print(minRe)
    return scores


# # Gridsearch

# In[23]:


from sklearn.model_selection import GridSearchCV
from sklearn import metrics

def grid_tuning(X_train, y_train, algorithms, param_grid, X_test=None, y_test=None, folds=10):
    cv_scores = []
    test_scores = []
    params = []
    # Gridsearch loop for all classifiers
    for (name, a, t_p) in list(zip(algorithms.keys(), algorithms.values(), param_grid)):
        print("Tuning hyper-parameters, based on accuracy for: {}\nwith parameter choice:\n{}".format(name, t_p))
        clf = GridSearchCV(a, t_p, cv = folds, scoring = 'accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        print("Best parameter set found on training set:")
        print(clf.best_params_)
        cv_score = max(clf.cv_results_["mean_test_score"])
        print("CV results of best model: {:.3f}".format(cv_score))
        #store round scores
        cv_scores.append(clf.cv_results_["mean_test_score"])
        params.append(clf.best_params_)
        if (X_test and y_test):
            #evaluate and store scores of estimators of each category on test set
            test_score = clf.score(X_test, y_test)
            print("Test score:", test_score)
            test_scores.append(test_score)
        print("===============================\n")
    cv_scores = dict(zip(algorithms.keys(), cv_scores))
    if (X_test and y_test):
        test_scores = dict(zip(algorithms.keys(), test_scores))
    print("Finished!")
    return cv_scores, test_scores


# # Voting

# In[24]:


from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
"""If n_folds==True the results are computed on the folds of the training using CV. Test set is ignored
scores is used if we already have the old scores of the classifiers in order to compare with voting results"""

def voting3(X_train, y_train, models, alg1, alg2, alg3, X_test=None, y_test=None,
            voting='hard', scores=None, n_folds=None):
    #cross validation case:
    # Training is automatically split to training and validation couples
    print("{} voting:".format(voting))
    if n_folds:
        print("Cross validation process... (test set will not be used):")
        vcl = VotingClassifier(estimators=[(alg1, models[alg1]),
                                           (alg2, models[alg2]),
                                           (alg3, models[alg3])], voting=voting)
        # cross validation results
        cv = KFold(n_folds)
        acc = np.mean(cross_val_score(clf, X_train, y_train, cv=cv))
        vcl.fit(X_train, y_train)
        y_pred = []
        # print(metrics.classification_report(y_test, y_pred, digits=3))
        print("{} voting accuracy (CV): {:.3f}".format(voting, acc))
    elif not((X_test is None) or (y_test is None)):
        vcl = VotingClassifier(estimators=[(alg1, models[alg1]), (alg2, models[alg2]),
                                           (alg3, models[alg3])], voting=voting)
        vcl.fit(X_train, y_train)
        acc = vcl.score(X_test,y_test)
        y_pred = vcl.predict(X_test)
        print(classification_report(y_test, y_pred, digits=3))
        if scores:
            print("Accuracy scores (Test set):\n{}: {:.3f}, {}: {:.3f}, {}: {:.3f}, {} ensemble: {:.3f}"
                  .format(alg1, scores[alg1]['Accuracy'], alg2, scores[alg2]['Accuracy'],
                          alg3, scores[alg3]['Accuracy'], voting, acc))
        else:
            print("{} voting accuracy (Test set):\n{:.3f}".format(voting, acc))
    print("Finished voting!\n=====================================================")
    return scores, y_pred
