import numpy as np

def gradient_descent(x,y):
    iterations = 10000
    m_curr=b_curr=0
    learning_rate = 0.00001
    n = len(x)

    for i in range(0,iterations):
        y_predicted = m_curr*x - b_curr
        bd = -(2/n) * sum(y - y_predicted)
        md = -(2/n) * sum(x * (y - y_predicted))
        b_curr = b_curr - learning_rate * bd
        m_curr = m_curr - learning_rate * md
        print()


x = [1,2,3,4,5]
y = [5,7,9,11,13]
