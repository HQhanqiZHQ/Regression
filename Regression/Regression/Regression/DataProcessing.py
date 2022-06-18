import pandas as pd
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures, scale, StandardScaler
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import time

#################################  function part  ######################################
# The functions we gonna use
def w(dis):
    """
        Args:
            dis: distance
        
        Func:
            Weight Function
    """
    dis = dis / 0.3
    if dis < 0:
        return 0
    elif dis <= 0.5:
        return 2/3 - 4 * dis**2 + 4 * dis**3
    elif dis <= 1:
        return 4/3 - 4 * dis + 4 * dis**2 - 4/3 * dis**3
    else:
        return 0


def mls(x, y, x_i):
    """
        Moving Least Square
    """
    sumxx = sumx = sumxf = sumf = sumw = 0
    for (a, b) in zip(x, y):
        weight = w(abs(x_i - a))
        sumw += weight
        sumx += a * weight
        sumxx += a * a * weight
        sumf += b * weight
        sumxf += a * b * weight
    A = np.array([[sumw, sumx],
                  [sumx, sumxx]])#A
    B = np.array([sumf, sumxf])#B
    ans = np.linalg.solve(A, B)
    print("%f+%f*x,{x|%f<x<%f}"%(ans[0],ans[1],x_i,x_i+step))
    return ans[0] + ans[1] * x_i


def stdError_func(y_test, y):
    """
        Func:
            To compute the standard error
    """
    return np.sqrt(np.mean((y_test - y) ** 2))
    

def select_vif(X):
    """
        Func:
            To compute the VIFs and auto delete variables with multicolinearity (highest vif)
    """
    A = np.linalg.inv(X.T @ X)
    vif = np.array([variance_inflation_factor(X.values, X.columns.get_loc(i)) for i in X.columns])
    drop_index = []
    for i in range(A.shape[1]):
        #vif.append(A[i,i])
        if vif[i]>10:
            drop_index.append(i)

    drop_index = [X.columns[i] for i in drop_index]
    if epoch<2:
        vifs = ["%.2f"%v for v in vif]
        zipped = list(zip(drop_index,vifs))
        df_data = np.matrix(zipped)
        print("\n")
        print(pd.DataFrame(data=df_data, columns=["Col_name", "VIF"]))
    print("\nTotal Num of variables can try to delete: %d"%(len(drop_index)))
    
    try_drop = X.columns[vif==vif.max()]
    print("\n\nTry to drop column '%s', whose vif is %.3f...\n"%(try_drop[0], vif.max()))
    
    return X.drop(columns = try_drop)


def poly_fit(x_train, y_train, x_test, y_test, degree=0):
    """
        Args:
            x_train: the training data
            y_train: the training data
            x_test: the test data
            y_test:the test data
            degree: default to be 0, which means simple multiple linear
                    regression. When comes to 2, meaning we'll do a 
                    Cubic polynomial regression.
        
        Func:
            to do multi-degree polynomial fitting
    """
    print("**"*30+"\n\nHere is our poly-fit info: \n")
    
    if degree>0:
        poly_reg =PolynomialFeatures(degree=degree)
        x_train =poly_reg.fit_transform(x_train)
        x_test =poly_reg.fit_transform(x_test)
    
    cft = linear_model.LinearRegression()
    cft.fit(x_train, y_train)
    predict_y = cft.predict(x_test)
    strError = mean_squared_error(predict_y, y_test)
    score = cft.score(x_test, y_test)
    print("coefficients", cft.coef_)
    print("intercept", cft.intercept_)
    print('degree={}: strError={:.2f}, clf.score={:.2f}'.format(3, strError, score))
    print("\n\n"+"**"*30)
    return predict_y


def draw_scatter(X, col_name, xlabels, save=True, whether_scale=True):
    """
        Args:
            X: the data without NA
            col_name: each columns' name
            xlabels: a list that contains the explaination of each column's name
            save: whether wanna auto-save the image. default to be true.
            
        Func:
            to draw a beautiful scatter graph
    """
    if(whether_scale):
        X_scared = pd.DataFrame(scale(X, axis=1), columns=col_name)
    else:
        X_scared = X
    for i in range(len(col_name)):
        x = X[col_name[i]]
        plt.scatter(x, y, c=y, s=size, alpha=0.5, cmap='viridis')
        plt.colorbar()

        plt.tick_params(labelsize=16)
        # plt.ylabel(u"自有住房的中位数价值(以1000美元计)", size=12)
        plt.ylabel(u"Median value of owner-occupied homes in $1000'ss", size=12)  # English version
        plt.xlabel(xlabels[i], size=12)
        
        if save:
            plt.savefig('./img/%i_en.png'%i) # to auto save the image
        plt.show()
        print("\nThe %dth diagram:"%(i+1))


def kMeans(X, y, n_clusters=3):
    """
        Args:
            n_clusters: how many clusters do we have
    """
    X = np.matrix(X)
    Y = np.matrix(y).T @ np.matrix(np.ones(13))
    X_pro = np.array([X, Y])
    fig = plt.figure(figsize=(8,3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    colors_ori = ["#C9B8B1","#BD9A87","#D32801","#901F02","#033F5C","#4D7F8B","#B1ACB3","#53565C","#FFB6C1", "#C71585","#9370DB","#90EE90","#F5DEB3"]
    
    # plot ori-data
    ax = fig.add_subplot(1,2,1)
    _, _, features = np.shape(X_pro)
    p = [0 for i in range(13)]
    for j in range(features):
        p[j] = ax.scatter(X_pro[0, :, j], X_pro[1, :, j], c=colors_ori[j], s=2, alpha=0.5, label=col_name[j])
    legend = ax.legend(p, col_name)
    legend = ax.legend(loc='best', shadow=False, fontsize=10, bbox_to_anchor=(1.2, 1))
    frame = legend.get_frame()  
    frame.set_alpha(0.8)  
    frame.set_facecolor('none')
    ax.tick_params(labelsize=12)
       
    
    X_pro = np.array([X.reshape(-1,1), Y.reshape(-1,1)]).reshape(2,-1).T
    k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    t0 = time.time()
    k_means.fit(X_pro)
    t_batch = time.time() - t0
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X_pro, k_means_cluster_centers)
    ax = fig.add_subplot(1, 2, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k		# my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X_pro[my_members, 0], X_pro[my_members, 1], 'w',
                markerfacecolor=col, marker='.', alpha=0.4)	# 将同一类的点表示出来
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', marker='o', alpha=0.4)	# 将聚类中心单独表示出来
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
    plt.show()
    

####################################################################################################


#####################################  Begin computing  ############################################

# Draw contour
x = np.linspace(-3,3,1000)
y = np.linspace(-3,3,1000)
X,Y = np.meshgrid(x,y)
Z = (X-0.7)**2 + (Y-1.8)**2
a = np.arange(-2*np.pi,2*np.pi,0.00001)
r_x = r_y = 0
b = np.sqrt(np.power(1,2)-np.power((a-r_x),2))+r_y # r_x是圆心横坐标；r_y是圆心纵坐标

plt.figure(figsize=(6,6))
# plt.contourf(X,Y,Z) # to fill the contour
plt.contour(X,Y,Z,[0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1], cmap="hot")
plt.scatter(0.7, 1.82, c="r", s=14)
#plt.clabel(contour,fontsize=10,colors='k')
plt.plot([0, 1.414, 0, -1.414, 0], [1.414, 0, -1.414, 0, 1.414], "b--", label="L1 reg")
plt.plot(a, b, "g--", label="L2 reg")
plt.plot(a, -b, "g--")
plt.legend(fontsize=12)
plt.show()


# Initialize the data
d = pd.read_csv("BostonHousing.csv",delim_whitespace=True)
# d.to_csv("BostonHousing_comma.csv")
d_nona = d.dropna(axis=0, how='any')
print(d_nona.head(7))

# initialize the variables
y = d_nona["MEDV"]
col_name = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"]
xlabels_en = [u"per capital crime rate by town", u"proportion of residential land zoned for lots over 25,000 sq.ft.", u"proportion of non-retail business acres per town", u"Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)", u" nitric oxides concentration (parts per 10 million)", u"average number of rooms per dwelling", u"proportion of owner-occupied units built prior to 1940", u"weighted distances to five Boston employment centers", u"index of accessibility to radial highways", u"full-value property-tax rate per 10,000 USD", u"pupil-teacher ratio by town", u"Black 1000(Bk — 0.63)² where Bk is the proportion of blacks by town", "lower status of the population"]

xlabels = [u"城镇人均犯罪率", u"占地超过25,000平方英尺的住宅用地比例", u"每个城镇非零售业务英亩的比例", u"查尔斯河虚拟变量(如果束缚河流，则为1；否则为0)", u"一氧化氮浓度(百万分之几)", u"每个住宅的平均房间数", u"1940年之前建造的自有住房的年龄比例", u"加权到五个波士顿就业中心的距离", u"径向公路可及性指数", u"每10,000美元的全值房产税税率", u"按镇划分的师生比例", u"黑人1000（Bk — 0.63）²，\n其中Bk是按城镇划分的黑人比例", "人口地位降低的百分比"]


# figure out the config to draw graph in Chinese
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号


# Prepare our original data
size = np.array(y)
X = d_nona.iloc[:,:-1]
X_scared = pd.DataFrame(scale(X, axis=1), columns=col_name)  # standardize each column
X_scared.to_csv("Standardized.csv")

# The following scripts are prepareing for Machine Learning
rows = X.shape[0]
index = np.arange(rows)
np.random.shuffle(index)
x_train = X_scared.iloc[index[:round(rows*0.6)], :]
y_train = y.iloc[index[:round(rows*0.6)]]
x_test = X_scared.iloc[index[round(rows*0.6):], :]
y_test = y.iloc[index[round(rows*0.6):]]
print("\nHere is the shape of training data and test data:")
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# Using k-means
kMeans(X, y)


# Using PCA to select the principal components
pca = decomposition.PCA()#n_components=0.99)
# pca.fit(X)
pca.fit(X_scared)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.components_)


# To plot the PCA, so that we can get some intuition
plt.figure()
plt.bar(np.arange(len(pca.explained_variance_)), pca.explained_variance_ratio_, alpha=0.4)
plt.plot(pca.explained_variance_ratio_, 'b', alpha=0.8, linewidth=2)
plt.xlabel('Components', fontsize=16)
plt.ylabel('explained_variance_', fontsize=16)

for _x, _y in zip(np.arange(3), pca.explained_variance_ratio_):
    plt.text(_x, _y+0.01, '%.3f' % _y, ha='center', va='bottom', size=10)

plt.show()


# polynomial fitting
y_predict = poly_fit(x_train, y_train, x_test, y_test, degree=0)


# After PCA doing poly-fit
#X_pca = pca.fit_transform(X)
X_pca = pca.fit_transform(X_scared)
pd.DataFrame(X_pca).to_csv("BostonHousing_pca.csv")
poly_fit(X_pca, y, X_pca, y)

"""
# Trying Moving Least Square Method
step = 0.1
xx = np.arange(np.array(X_scared).min(), np.array(X_scared).max()+step, step)
yy = [mls(X_scared, y, xi) for xi in xx]
plt.plot(xx, yy)
plt.scatter(X_scared, y, c='r', alpha=0.5)
plt.show()
"""

# Try to do Ridge Regression
std_y = StandardScaler()
std_y.fit_transform(np.array(y_train).reshape(-1,1))
std_y.transform(np.array(y_test).reshape(-1,1))
ri = Ridge(alpha=1.0)
ri.fit(x_train, y_train)
print("coefficients", ri.coef_)
print("intercept", ri.intercept_)
# predict the price of houses
ri_y_predict = std_y.inverse_transform(ri.predict(x_test))
print("梯度下降每个房子的预测价格:", ri_y_predict)
print("正规方程均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))
print("岭回归误差：", mean_squared_error(std_y.inverse_transform(y_test), ri_y_predict))



# to draw all the graph with different x
draw_scatter(d_nona, col_name, xlabels, save=True, whether_scale=True)

# To check if its epsilons fits Normal distribution
"""
x_check = []
x_check_mean = []
epsilon_check = []

for col in col_name:
    x_check.append(d_nona[col].value_counts().index[0])
    x_check_mean.append(y[d_nona[col]==x_check[-1]].mean())
    epsilon_check.append([])
    for i in y[d_nona[col]==x_check[-1]]:
        epsilon_check[-1].append(i - x_check_mean[-1])


# Print out the eplisons to check
plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False      # 用来正常显示负号

for i in range(len(col_name)):
    x = epsilon_check[i]
    eps_max = np.array(epsilon_check[i]).max()
    y = [-(abs(i))**2+(eps_max)**2 for i in np.array(epsilon_check[i])]
    #y = np.zeros(len(x))
    #plt.scatter(x, y, alpha=0.5)
    plt.bar(x, y, alpha=0.5)

    plt.tick_params(labelsize=16)
    # plt.ylabel("Median value of owner-occupied homes in $1000's", size=12)
    # plt.ylabel(u"自有住房的中位数价值(以1000美元计)", size=12)
    plt.ylabel(u"Median value of owner-occupied homes in $1000'ss", size=12)  # English version
# plt.xlabel("per capita crime rate by town", size=12)
# plt.xlabel("proportion of residential land zoned for lots over 25,000 sq.ft.", size=12)
    plt.xlabel(xlabels_en[i], size=12)
    
    plt.savefig('./img/epsilon_%i_en.png'%i)
    plt.show()
    print("\nThe %dth diagram:"%(i+1))

"""

# To check multicolinearity

"""
# Using conditional number
X = d_nona.iloc[:,:-1]
epoch = 1
# X_1 = X.drop(columns=["NOX","RM", "PTRATIO", "TAX", "AGE", "B"])
A = np.linalg.inv(X.T @ X)
# A1 = np.linalg.inv(X_1.T @ X_1)
w, v = LA.eig(A)
# w1, v1 = LA.eig(A1)
con_num = v.max()/np.abs(v).min()
# con_num1 = v1.max()/np.abs(v1).min()
print("Conditional Number: "+str(abs(con_num)))
# print("Conditional Number: "+str(abs(con_num1)))
#if con_num>10:
print("\nWe consider CN>10^4 is having multicolinearity.\n")
while con_num>10**4:
    print("\nIt has multicolinearity. Let's do something to reduce its multicolinearity.")
    print("\n\n")
    print(("Epoch %d"%epoch).center(60, "*"))
    print("\n\nRunning...\nPlease wait a moment...\n\n")
    X = select_vif(X)
    epoch += 1
    A = np.linalg.inv(X.T @ X)
    w, v = LA.eig(A)
    con_num = v.max()/np.abs(v).min()
    if len(X.columns)<13:
        print("Conditional Number: "+str(abs(con_num)))
    print("\n"+"**"*30+"\n")
    
print("\nCongratulate! Your data don't have multicolinearity!\n\n")
    
"""


