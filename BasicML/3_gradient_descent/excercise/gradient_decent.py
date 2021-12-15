import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd 
import math
from sklearn.linear_model import LinearRegression

def predict_using_sklearn():
    df = pd.read_csv("test_scores.csv")
    r = LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr=0
    b_curr=0
    learning_rate = 0.0002
    iterations = 1000000
    prev_cost = 0
    n = len(x)
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, prev_cost, rel_tol=1e-20):
            break
        #print(cost_previous = cost}, iteration {}".format(m_curr,b_curr,abs(cost-prev_cost),i))
        prev_cost = cost
        
    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv('test_scores.csv')
    x = np.array(df['math'])
    y = np.array(df['cs'])
    
    m,b = gradient_descent(x,y)
    print("Using gradient descent coeffiecient={} and Intercept={}".format(m,b))
    
    msk,bsk = predict_using_sklearn()
    print("using sklearn descent coeffiecinet = {} and Intercept = {}".format(msk,bsk))

