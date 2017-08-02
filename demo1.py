import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, cross_validation, discriminant_analysis, datasets

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target,test_size=0.25, random_state=0)

def test_LinearRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s,intercept:%.2f'%(regr.coef_, regr.intercept_))
    print('Residual sum of squares:%.2f'%np.mean((regr.predict(X_test) - y_test)**2))
    print('Score:%.2f'%regr.score(X_test, y_test))

def test_Ridge(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.Ridge()
    regr.fit(X_train, y_train)
    print('Coefficients:%s,intercept:%.2f'%(regr.coef_, regr.intercept_))
    print('Residual sum of squares:%.2f'%np.mean((regr.predict(X_test) - y_test)**2))
    print('Score:%.2f'%regr.score(X_test, y_test))

def test_Lasso_alpha(*data):
    X_train, X_test, y_train, y_test = load_data()
    alphas = [0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000]
    score = []
    for alpha in alphas:
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(X_train, y_train)
        score.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(alphas, score)
    ax.set_xlabel('alpha')
    ax.set_xscale('log')
    ax.set_ylabel('score')
    ax.set_title('Ridge')
    plt.show()

def load_iris_data():
    iris = datasets.load_iris()
    return cross_validation.train_test_split(iris.data, iris.target, test_size=0.25, \
                                             stratify=iris.target, random_state=0)

def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression()
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%s'%(regr.coef_, regr.intercept_))
    print('score:%s'%regr.score(X_test, y_test))

def test_LogisticRegression_multinomial(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    regr.fit(X_train, y_train)
    print('Coefficients:%s, intercept:%s'%(regr.coef_, regr.intercept_))
    print('Score:%s'%regr.score(X_test, y_test))

X_train, X_test, y_train, y_test = load_iris_data()
test_LogisticRegression_multinomial(X_train, X_test, y_train, y_test)