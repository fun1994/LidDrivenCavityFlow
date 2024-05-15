# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 09:42:44 2024

@author: HFKJ059
"""

import numpy as np
from matplotlib import pyplot as plt


def read_1d(path, filename):
    with open("./data/" + path + "/" + filename + ".txt", "r") as file:
        data = file.read()
    data = data.split()
    for i in range(len(data)):
        data[i] = float(data[i])
    data = np.array(data)
    return data

def read_2d(path, filename):
    data = []
    with open("./data/" + path + "/" + filename + ".txt", "r") as file:
        while True:
            line = file.readline()
            if not line:
                break
            data_temp = line.split()
            data.append(data_temp)
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = float(data[i][j])
    data = np.array(data)
    return data

def read(grid, Re):
    x_p = read_1d(grid + "/Re=" + Re, "x_p")
    y_p = read_1d(grid + "/Re=" + Re, "y_p")
    x_u = read_1d(grid + "/Re=" + Re, "x_u")
    y_u = read_1d(grid + "/Re=" + Re, "y_u")
    x_v = read_1d(grid + "/Re=" + Re, "x_v")
    y_v = read_1d(grid + "/Re=" + Re, "y_v")
    p = read_2d(grid + "/Re=" + Re, "p")
    u = read_2d(grid + "/Re=" + Re, "u")
    v = read_2d(grid + "/Re=" + Re, "v")
    return x_p, y_p, x_u, y_u, x_v, y_v, p, u, v

def show(x, y, u, v, grid, Re):
    X, Y = np.meshgrid(x, y)
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, u.T, v.T, density=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("grid=" + grid + ", Re=" + Re)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_aspect("equal")
    plt.show()

def run(grid, Re):
    x_p, y_p, x_u, y_u, x_v, y_v, p, u, v = read(grid, Re)
    if grid == "staggered":
        u = (u[:-1, :] + u[1:, :]) / 2
        v = (v[:, :-1] + v[:, 1:]) / 2
    show(x_v, y_u, u, v, grid, Re)

def main():
    run("staggered", "100")
    run("staggered", "400")
    run("staggered", "1000")
    run("collocated", "100")
    run("collocated", "400")
    run("collocated", "1000")


main()
