import math
from math import e
import sys
from describe import Math_calculat
from describe import DataSet
import decimal
import numpy as np
import argparse
from logreg_train import LogisticRegression

if __name__ == '__main__':
    try:

        parser = argparse.ArgumentParser()
        parser.add_argument('--file', nargs=1,
                            help="test csv file is dataset with features, default is datasets/dataset_test.csv",
                            default=['datasets/dataset_test.csv'], required=False)
        parser.add_argument('--x_columns', nargs='+',
                            help='''It's columns which wil be use for predict, default use 8, 9, 10, 11, 12, 17, 18 columns 
                                 (Herbology, Defense Against the Dark Arts, Divination, Muggle Studies, Ancient Runes, Charms, Flying)''',
                            default=[8, 9, 10, 11, 12, 17, 18], type=int, required=False)
        parser.add_argument('--x', nargs='+',
                            help="Input x, default value is None",
                            default=[], type=int, required=False)
        parser.add_argument('--fit_intercept', nargs=1,
                            help="Input False if don't want add 1 to x else True, default value is 0",
                            default=[0], type=int, required=False)
        parser.add_argument('--size', nargs=1,
                            help="Input size predict dataset, default all rows",
                            default=[400], type=int, required=False)
        parser.add_argument('--theta_exit', nargs=1,
                            help="flag if model.txt dont't exists input Error else theta equal 0, default 0",
                            default=[0], type=int, required=False)

        theta = []
        try:
            with open('model.txt', 'r') as f:
                cat = f.read()
            mas = []
            for i in cat.strip().split('\n'):
                s = []
                for j in i.strip().split(';'):
                    if j.strip() != '':
                        s.append(float(j))
                theta.append(s)
        except Exception:
            theta = [[], [], [], []]

        if len(theta) != 4:
            print('Error: bad model.txt')
            sys.exit()

        lr = {'Ravenclaw': [], 'Slytherin': [], 'Gryffindor': [], 'Hufflepuff': []}
        if parser.parse_args().x:
            max_prob = -1
            max_key = 'Ravenclaw'
            k = 0
            for i in lr:
                model = LogisticRegression()
                model.theta = theta[k]
                k += 1
                prob = model.predict_prob(X=np.array([parser.parse_args().x]), fit_intercept=parser.parse_args().fit_intercept[0])[0]
                key = model.predict(X=np.array([parser.parse_args().x]), fit_intercept=parser.parse_args().fit_intercept[0])[0]
                if prob >= max_prob and key:
                    max_prob = prob
                    max_key = i
            with open('houses.csv', 'w') as f:
                f.write('Index,Hogwarts House\n')
                f.write('0,' + str(max_key) + '\n')
            sys.exit()

        k = 0
        for i in lr:
            lr[i].append(LogisticRegression(file=parser.parse_args().file[0], y_true=i, x_columns=parser.parse_args().x_columns, size=parser.parse_args().size[0]))
            ms = lr[i][0].predict_file(theta[k], theta_exit=parser.parse_args().theta_exit[0])
            k += 1
            lr[i].append(ms[0])
            lr[i].append(ms[1])

        with open('houses.csv', 'w') as f:
            f.write('Index,Hogwarts House\n')
            for i in range(len(lr['Ravenclaw'][1])):
                f.write(str(i) + ',')
                max_prob = -1
                key = 'Ravenclaw'
                for j in lr:
                    if lr[j][1][i] and lr[j][2][i] >= max_prob:
                        max_prob = lr[j][2][i]
                        key = j
                f.write(str(key) + '\n')
    except Exception:
        print('Error: bad parameters')