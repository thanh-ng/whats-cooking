__author__ = 'thanh'

# Loading used packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.decomposition import SparsePCA
from utils import*

def main():
    start_time = time.time()
    print '####################################################'
    print ' # Loading data'
    data_train_frame, target_transform = loadJS(trainpath)
    data_train = data_train_frame['ingredients']
    target_train = data_train_frame['cuisine']

    data_test_frame, nonsense = loadJS(testpath)
    data_test = data_test_frame['ingredients']
    id_test = data_test_frame['id']

    print '####################################################'

    print ' # Pipelining tfidf, chi selection, and svm together'
    pipe_1 = Pipeline([('tfidf', TfidfVectorizer()),
                        ('chi_selector', SelectPercentile(chi2)),
                        ('svm',SVC(kernel='rbf'))
                        ])
    """
    pipe_1 = Pipeline([('tfidf', TfidfVectorizer()),
                        ('chi_selector', SelectPercentile(chi2)),
                       ('sparse_pca',SparsePCA(verbose=True)),
                        ('svm',SVC(kernel='rbf'))
                        ])
    """
    print ' # Tuning hyperparameters and return best estimator'
    param_grid_1 = dict(chi_selector__percentile=[60],
                      svm__C=[3],
                      svm__gamma=[1.103,1.104,1.105]
                      )
    print '     ## param_grid:', param_grid_1
    print '     ## Fitting data'
    estimator_1 = GridSearchCV(pipe_1, param_grid=param_grid_1, verbose=10,n_jobs=3).fit(data_train, target_train)

    print '     ## Best model =:',estimator_1.best_params_
    print '     ## Best SCORE: ', estimator_1.best_score_
    print '####################################################'

    print ' # Predicting'
    target_test_predicted = list(target_transform.inverse_transform(estimator_1.predict(data_test)))

    print ' # Writing predicted results to csv'
    write2CSV(returnpath, zip(list(id_test),target_test_predicted))

    print '####################################################'

    mins = (time.time() - start_time) / 60
    print("--- %s minutes ---" % mins)


if __name__ == "__main__":
    main()
