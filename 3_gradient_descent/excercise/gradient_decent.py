import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd 
import math

def gradient_descent(x,y):
    m_curr=b_curr=0
    learning_rate = 0.0002
    iterations = 100000
    prev_cost = 0
    n = len(x)
    for i in range(iterations):
        y_predicted = m_curr*x-b_curr
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        md = (-2/n)*sum(x*(y-y_predicted))
        bd = (-2/n)*sum(y-y_predicted)
        m_curr = m_curr-learning_rate*md
        b_curr = b_curr - learning_rate*bd
        if(math.isclose(cost,prev_cost,abs_tol=1e-20)):
            break
        print("m {}, b {}, cost {}, iteration {}".format(m_curr,b_curr,abs(cost-prev_cost),i))
        prev_cost = cost



df = pd.read_csv('test_scores.csv')
x = np.array(df['math'])
y = np.array(df['cs'])

gradient_descent(x,y)