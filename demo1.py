import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, cross_validation, discriminant_analysis, datasets, decomposition, manifold

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

def test_LogisticRegression_C(*data):
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2, 4,num=100)
    Scores = []
    for c in Cs:
        regr = linear_model.LogisticRegression(C=c)
        regr.fit(X_train, y_train)
        Scores.append(regr.score(X_test, y_test))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Cs, Scores, 'r-')
    ax.set_xlabel(r'Cs')
    ax.set_ylabel(r'Scores')
    ax.set_xscale('log')
    ax.set_title('LogisticRegression')
    plt.show()

def load_data_PCA():
    iris = datasets.load_iris()
    return iris.data, iris.target

def plot_KPCA_poly(*data):
    X, y = data
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1))
    Params = [(3,1,1),(3,10,1),(3,1,10),(3,10,10),(10,1,1),(10,10,1),(10,1,10),(10,10,10)]
    for i,(p,gamma,r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components=2, kernel='poly',
                                       gamma=gamma, degree=p,coef0=r)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2,4,i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(X_r[position, 0], X_r[position, 1],label='target=%s'%label,color=color)
        ax.set_xlabel(r'X[0]')
        ax.set_ylabel(r'y[1]')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(r'$(%s(x \cdot z + 1)+%s)^{%s}$'%(gamma,r,p))
        ax.legend(loc = 'best')
    plt.suptitle(r'KPCA-poly')
    plt.show()

def plot_KPCA_rbf(*data):
    X, y = data
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1))
    Gammas = [0.4,1,4,10]
    for i,gamma in enumerate(Gammas):
        kpca = decomposition.KernelPCA(n_components=2,kernel='rbf',gamma=gamma)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2,2,i+1)
        for label, color in zip(np.unique(y), colors):
            position = y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label='target=%d'%label,color = color)
        ax.set_xlabel(r'X[0]')
        ax.set_ylabel(r'y[1]')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc = 'best')
        ax.set_title(r'$exp(-%s||x-z||^2)$'%gamma)
    plt.suptitle('KPCA-rbf')
    plt.show()

def test_MDS(*data):
    X,y = data
    for n in [4,3,2,1]:
        mds = manifold.MDS(n_components=n)
        mds.fit(X)
        print('stress(n_components=%d):%s'%(n, str(mds.stress_)))

def plot_MDS(*data):
    X,y = data
    mds = manifold.MDS(n_components=2)
    X_r = mds.fit_transform(X)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ((1,0,0),(0,1,0),(0,0,1))
    for label,color in zip(np.unique(y), colors):
        position = y == label
        ax.scatter(X_r[position, 0], X_r[position, 1], label='target=%s'%label, color=color)
    ax.set_xlabel(r'X[0]')
    ax.set_ylabel(r'y[0]')
    ax.set_title('MDS')
    ax.legend(loc = 'best')
    plt.show()
X, y = load_data_PCA()
plot_MDS(X,y)