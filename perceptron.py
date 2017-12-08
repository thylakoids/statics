w = [0, 0]
b = 0
import numpy as np 
def createDataSet():
    """
    create dataset for test
    """
    return [[(3, 3), 1], [(4, 3), 1], [(1, 1), -1]]

def update(item):
    """
    update with stochastic gradient descent
    """
    global w, b
    w[0] += item[1] * item[0][0]
    w[1] += item[1] * item[0][1]
    b += item[1]

def cal(item):
    # """
    # calculate the functional distance between 'item' an the dicision surface. output yi(w*xi+b).
    # """
    # res = 0
    # for i in range(len(item[0])):
    #     res += item[0][i] * w[i]
    # res += b
    # res *= item[1]
    # return res

    return (np.dot(item[0],w)+b)*item[1]


def check():
    """
    check if the hyperplane can classify the examples correctly
    """
    flag = False
    for item in training_set:
        if cal(item) <= 0:
            flag = True
            update(item)
    if not flag:
        print "RESULT: w: " + str(w) + " b: " + str(b)
    return flag

if __name__ == "__main__":
    training_set = createDataSet()
    while check():
        pass