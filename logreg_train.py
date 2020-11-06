import math
from math import e
import sys
from describe import Math_calculat
from describe import DataSet
import decimal
import numpy as np
import argparse
import pandas as pd


class LogisticRegression:
    """
        - Logistic Regression allow to find probability assigment x to four classes
            Hogwarts House = [Ravenclaw, Slytherin, Gryffindor, Hufflepuff]
        - Parameters:
            :param file: It's path to file with features and target
            :param x: It's massiv with float, x = [[1,1, ... 1], [2,2, ... 2], ...] where x.shape[1] == len(x_columns)
        - Example to run:

        >>  lg = LogisticRegression(file, y_true)
        >>  lg.fit()
        >>  print(lg.predict(x))
        >>  print(lg.predict_prob(x))

    """
    def __init__(self, file='datasets/dataset_train.csv', y_true='Slytherin', x_columns=[], size=10, lr=0.15,
                 num_iter=2000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.verbose = verbose
        self.fit_intercept = fit_intercept
        self.y_true = y_true
        self.size = size
        self.x_columns = x_columns
        self.file = file
        self.theta = []

        if not (y_true in ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']):
            print("Error: bad parameter y_true")
            sys.exit()
        if num_iter < 1 or lr < 0 or lr > 1:
            print("Error: bad parameter num_iter or lr")
            sys.exit()
        if not (type(x_columns) == list):
            print("Error: x_columns must will be list int")
            sys.exit()

    def __add_intercept(self, X):
        """

        :param X: It's massiv with float, x = [[1,1, ... 1], [2,2, ... 2], ...] where x.shape[1] == len(x_columns)
        :return:
        """
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        """

        :param z: It's float which use function sigmoid
        :return: value sigmoid
        """
        return 1. / (1. + np.exp(-z))

    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def preprocessing(self, ds, i):
        col = ds.get_float_col(i)[:self.size]
        mc = Math_calculat(col)
        mean = mc.Mean()
        std = mc.Std()
        # min = mc.Quartile(0)
        # max = mc.Quartile(1)
        # return  ([(i - min) / (max - min) for i in col])
        return ([(i - mean) / std for i in col])

    def get_x_y(self, ds, return_y=True):
        mas_columns = []
        x = []
        y = []
        if self.x_columns:
            for i in self.x_columns:
                if i in ds.numeric_columns:
                    mas_columns.append(i)
        if not mas_columns:
            mas_columns = ds.numeric_columns
        for i in mas_columns:
            x.append(self.preprocessing(ds, i))
        x_new = []
        for i in range(len(x[0])):
            new = []
            for j in range(len(mas_columns)):
                new.append(x[j][i])
            x_new.append(new)
        if not return_y:
            return np.array(x_new)
        for i in ds.get_col(1)[:self.size]:
            y.append(1) if i == self.y_true else y.append(0)
        return np.array(x_new), np.array(y)

    def fit(self):
        ds = DataSet(filename=self.file)
        ds.find_numeric_label()
        X, y = self.get_x_y(ds)
        if self.fit_intercept:
            X = self.__add_intercept(X)

        self.theta = np.random.randn(X.shape[1])
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            if (self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')

    def predict_file(self, theta=np.array([]), theta_exit=0):
        df = DataSet(filename=self.file)
        df.find_numeric_label()
        X = self.get_x_y(df, return_y=False)
        X = self.__add_intercept(X)
        self.theta = np.array(theta)
        if not theta and theta_exit:
            print("Error: Have not theta")
            sys.exit()
        if not theta:
            self.theta = np.ones(X.shape[1])
        if self.theta.shape[0] != X.shape[1]:
            print('Error: bad theta or X')
            sys.exit()
        return [self.predict(X), self.predict_prob(X)]

    def predict_prob(self, X, fit_intercept=0):
        """

        :param X: It's massiv with float, x = [[1,1, ... 1], [2,2, ... 2], ...] where x.shape[1] == len(x_columns)
        :param fit_intercept: It's add 1 to X samples else not
        :return: predict probability
        """
        if fit_intercept:
            X = self.__add_intercept(X)
        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5, fit_intercept=0):
        """

        :param X: It's massiv with float, x = [[1,1, ... 1], [2,2, ... 2], ...] where x.shape[1] == len(x_columns)
        :param threshold: It's threshold before 0 after 1
        :param fit_intercept: It's add 1 to X samples else not
        :return: predict 0 or 1
        """
        return self.predict_prob(X, fit_intercept=fit_intercept) >= threshold


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--file', nargs=1,
                            help="train csv file is dataset with feature and target, default is datasets/dataset_train.csv",
                            default=['datasets/dataset_train.csv'], required=False)
        parser.add_argument('--x_columns', nargs='+',
                            help='''Input index x columns through space, default use 8, 9, 10, 11, 12, 17, 18 columns 
                                 (Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, Charms, Flying)''',
                            default=[8, 9, 10, 11, 12, 17, 18], type=int, required=False)
        parser.add_argument('--size', nargs=1,
                            help="Input size samples, default value is 1600 (all rows)",
                            default=[1600], type=int, required=False)
        parser.add_argument('--lr', nargs=1,
                            help="Input value learning rate, default value is 0.15",
                            default=[0.15], type=float, required=False)
        parser.add_argument('--num_iter', nargs=1,
                            help="Input count iter, default value is 100",
                            default=[100], type=int, required=False)

        pars = parser.parse_args()
        lr = {'Ravenclaw': [], 'Slytherin': [], 'Gryffindor': [], 'Hufflepuff': []}
        for i in lr:
            lr[i].append(LogisticRegression(file=pars.file[0], y_true=i, x_columns=pars.x_columns,
                                            lr=pars.lr[0], num_iter=pars.num_iter[0], size=pars.size[0]))
            lr[i][0].fit()
            lr[i].append(lr[i][0].theta)

        with open('model.txt', 'w') as f:
            for i in lr:
                for j in lr[i][1]:
                    f.write(str(j) + ';')
                f.write('\n')
    except Exception:
        print('Error: Bad input parameters')
