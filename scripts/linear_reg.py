#!/usr/bin/env python3
"""
Linear regression learning algorithm
"""

class LinearRegression:

    def train(self, x, y):
        """
        Trains a linear model from the data set

        Args:
            x -> independent variable matrix
            y -> dependent variable column vector
        
        Return:
            List of learn co-efficients
        """
        #Augmenting idv with constant intercept value 
        for xi in x:
            xi.insert(0, 1)
        
        no_idv = len(x[0])

        #Todo: make it a hyper param
        alpha = 0.001

        #Todo: randomization
        theta = [0] * no_idv
        theta_temp = [t for t in theta]
        
        #Convergence criteria is -> iterating 100000 times, this is for the sake of impl
        for j in range(0, 100000):
            for i in range(0, no_idv):
                theta[i] = theta_temp[i] - alpha * self._partial_diff(theta_temp, x, y, i)
            theta_temp = [t for t in theta]
        
        return theta
            
            

    def _partial_diff(self, theta, x, y, index):
        """
        Computes the paritial differentiation of htheta

        Args:
            theta -> coefficients row vector
            x -> independent variable matrix
            y -> dependent variable column vector
            index -> coefficient term being differentiated

        Return:
            updated coefficient at given 'index'
        """        
        no_idv = len(x[0])        
        m = len(x)
        sigma = 0.0
        for i in range(0, m):
            htheta = self._htheta(theta, x, i)
            sigma = sigma + (htheta - y[i][0]) * x[i][index]
        
        return sigma / m

    def _htheta(self, theta, x, index):
        """
        htheta function computation

        Args:
            theta -> coefficient row vector
            x -> independent variable matrix
            index -> row index of x
        """
        no_idv = len(x[0])
        htheta = 0.0
        for i in range(0, no_idv):
            htheta = htheta + theta[i] * x[index][i]
        
        return htheta


def main():
    regression = LinearRegression()
    #y = 2 + x
    x = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11]]
    y = [[3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13]]
    thetas = regression.train(x, y)

    print(thetas)

    for xi in x:
        htheta = thetas[0] * 1 + thetas[1] * xi[1]
        print("{xi} -> {htheta}".format(xi=xi[1], htheta=htheta))


if __name__ == "__main__":
    main()