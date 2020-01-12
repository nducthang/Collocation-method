"""
Phương pháp trọng số giải bài toán Laplace với biên dirichlet
Miền hình vuông
"""
import numpy as np
import time
import math
from multiprocessing import Process, Pipe
from numpy.core._multiarray_umath import ndarray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse as sp
import scipy.sparse.linalg as spla

""" ----- CÁC THAM SỐ ĐẦU VÀO ----- """
"""BÀI TOÁN 1"""
# Các hàm tương ứng trên các biên
g1 = lambda x1, x2: x1 ** 2 - 9
g2 = lambda x1, x2: 9 - x2 ** 2
g3 = lambda x1, x2: x1 ** 2 - 9
g4 = lambda x1, x2: 9 - x2 ** 2
# Toạ độ miền
a1, a2, b1, b2 = -3, 3, -3, 3
# Hàm kết quả
u = lambda x1, x2: x1 ** 2 - x2 ** 2
"""BÀI TOÁN 2"""
# # Các hàm tương ứng trên các biên
# g1 = lambda x1, x2: x1 ** 3 - 75 * x1 + 100
# g2 = lambda x1, x2: 225 - 15 * (x2 ** 2)
# g3 = lambda x1, x2: x1 ** 3 - 75 * x1 + 100
# g4 = lambda x1, x2: -25 + 15 * (x2 ** 2)
# # Toạ độ miền
# a1, a2, b1, b2 = -5, 5, -5, 5
# # Hàm kết quả
# u = lambda x1, x2: x1 ** 3 - 3 * x1 * (x2 ** 2) + 100
"""BÀI TOÁN 3"""
# # Các hàm tương ứng trên các biên
# g1 = lambda x1, x2: -np.exp(-3 * x1)
# g2 = lambda x1, x2: np.exp(-3 * math.pi) * np.cos(3 * x2)
# g3 = lambda x1, x2: -np.exp(-3 * x1)
# g4 = lambda x1, x2: np.exp(3 * math.pi) * np.cos(3 * x2)
# # Toạ độ miền
# a1, a2, b1, b2 = -math.pi, math.pi, -math.pi, math.pi
# # Hàm kết quả
# u = lambda x1, x2: np.exp(-3 * x1) * np.cos(3 * x2)

# Số phần tử chia trên mỗi cạnh
N = 32
# Số tiến trình
num_process = 8
""" ----- KẾT THÚC THAM SỐ ĐẦU VÀO ----- """

""" ----- CÁC TUỲ CHỌN ----- """
# Ma trận V-phi kích cỡ 4Nx4N
V = np.zeros((4 * N, 4 * N))
# vector xichma kích cỡ 4Nx1
xichma = []
# vecto g = V.xichma kích cỡ 4Nx1
g = np.zeros((4 * N, 1))
# Lưu trữ toạ độ trung điểm của từng đoạn phần tử hữu hạn trên biên
X = []


# Định nghĩa các nguyên hàm
def I1(x1, x2, y1):
    if x2 != b1:
        return (-1) * ((y1 - x1) * np.log((x1 - y1) ** 2 + (x2 - b1) ** 2) - 2 * y1 + 2 * ((x2 - b1) ** 2) * np.arctan(
            (y1 - x1) / np.fabs(x2 - b1)) / np.fabs(x2 - b1)) / (4 * np.pi)
    else:
        return (-1) * ((y1 - x1) * np.log((x1 - y1) ** 2 + (x2 - b1) ** 2) - 2 * y1 + 2 * ((x2 - b1) ** 2) * (-1) / (
                y1 - x1)) / (4 * np.pi)


def I2(x1, x2, y2):
    if x1 != a2:
        return (-1) * ((y2 - x2) * np.log((x2 - y2) ** 2 + (x1 - a2) ** 2) - 2 * y2 + 2 * ((x1 - a2) ** 2) * np.arctan(
            (y2 - x2) / np.fabs(x1 - a2)) / np.fabs(x1 - a2)) / (4 * np.pi)
    else:
        return (-1) * ((y2 - x2) * np.log((x2 - y2) ** 2 + (x1 - a2) ** 2) - 2 * y2 + 2 * ((x1 - a2) ** 2) * (-1) / (
                y2 - x2)) / (4 * np.pi)


def I3(x1, x2, y1):
    if x2 != b2:
        return (-1) * ((y1 - x1) * np.log((x1 - y1) ** 2 + (x2 - b2) ** 2) - 2 * y1 + 2 * ((x2 - b2) ** 2) * np.arctan(
            (y1 - x1) / np.fabs(x2 - b2)) / np.fabs(x2 - b2)) / (4 * np.pi)
    else:
        return (-1) * ((y1 - x1) * np.log((x1 - y1) ** 2 + (x2 - b2) ** 2) - 2 * y1 + 2 * ((x2 - b2) ** 2) * (-1) / (
                y1 - x1)) / (4 * np.pi)


def I4(x1, x2, y2):
    if x1 != a1:
        return (-1) * ((y2 - x2) * np.log((x2 - y2) ** 2 + (x1 - a1) ** 2) - 2 * y2 + 2 * ((x1 - a1) ** 2) * np.arctan(
            (y2 - x2) / np.fabs(x1 - a1)) / np.fabs(x1 - a1)) / (4 * np.pi)
    else:
        return (-1) * ((y2 - x2) * np.log((x2 - y2) ** 2 + (x1 - a1) ** 2) - 2 * y2 + 2 * ((x1 - a1) ** 2) * (-1) / (
                y2 - x2)) / (4 * np.pi)


# Sử dụng để tính gauss 2 chiều
epxilon: ndarray = np.zeros(7, dtype=object)
w = np.zeros(7)
# Ma trận chứa toạ độ các điểm chia trên miền
chia_mien = 100
matrix_point = np.zeros((chia_mien + 1, chia_mien + 1), dtype=object)


# ----- KẾT THÚC TUỲ CHỌN -----

def init_point():
    """ Khởi tạo các trung điểm của từng đoạn phần tử hữu hạn trên biên """
    global X
    X += [((a1 + (k - 1) * (a2 - a1) / N + a1 + k * (a2 - a1) / N) / 2, b1) for k in range(1, N + 1)]
    X += [(a2, (b1 + (k - 1) * (b2 - b1) / N + b1 + k * (b2 - b1) / N) / 2) for k in range(1, N + 1)]
    X += [((a1 + (k - 1) * (a2 - a1) / N + a1 + k * (a2 - a1) / N) / 2, b2) for k in range(1, N + 1)]
    X += [(a1, (b1 + (k - 1) * (b2 - b1) / N + b1 + k * (b2 - b1) / N) / 2) for k in range(1, N + 1)]


def init_g():
    """ Khởi tạo vecto g """
    for k in range(N):
        g[k] = g1(X[k][0], X[k][1])
        g[N + k] = g2(X[N + k][0], X[N + k][1])
        g[2 * N + k] = g3(X[2 * N + k][0], X[2 * N + k][1])
        g[3 * N + k] = g4(X[3 * N + k][0], X[3 * N + k][1])


def init_V():
    """ Khởi tạo ma trận V """
    for k in range(N):
        for i in range(4 * N):
            V[i][k] = I1(X[i][0], X[i][1], a1 + (k + 1) * (a2 - a1) / N) - I1(X[i][0], X[i][1], a1 + k * (a2 - a1) / N)
    for k in range(N):
        for i in range(4 * N):
            V[i][k + N] = I2(X[i][0], X[i][1], b1 + (k + 1) * (b2 - b1) / N) - I2(X[i][0], X[i][1],
                                                                                  b1 + k * (b2 - b1) / N)
    for k in range(N):
        for i in range(4 * N):
            V[i][k + 2 * N] = I3(X[i][0], X[i][1], a1 + (k + 1) * (a2 - a1) / N) - I3(X[i][0], X[i][1],
                                                                                      a1 + k * (a2 - a1) / N)
    for k in range(N):
        for i in range(4 * N):
            V[i][k + 3 * N] = I4(X[i][0], X[i][1], b1 + (k + 1) * (b2 - b1) / N) - I4(X[i][0], X[i][1],
                                                                                      b1 + k * (b2 - b1) / N)


def init_gauss_2d():
    """ Khởi tạo các điểm gauss để tính tích phân 2 chiều """
    epxilon[0] = (0.3333333333333333, 0.3333333333333333)
    epxilon[1] = (0.10128650732345633, 0.10128650732345633)
    epxilon[2] = (0.7974269853530872, 0.10128650732345633)
    epxilon[3] = (0.10128650732345633, 0.7974269853530872)
    epxilon[4] = (0.47014206410511505, 0.47014206410511505)
    epxilon[5] = (0.47014206410511505, 0.05971587178976981)
    epxilon[6] = (0.05971587178976981, 0.47014206410511505)
    w[0] = 0.225
    w[1] = 0.12593918054482717
    w[2] = 0.12593918054482717
    w[3] = 0.12593918054482717
    w[4] = 0.13239415278850616
    w[5] = 0.13239415278850616
    w[6] = 0.13239415278850616


def init_maxtrix_point():
    """ Khởi tạo ma trận chứa các điểm chia trên toàn miền """
    index_i, index_j = chia_mien, chia_mien
    i = a1
    while index_i >= 0:
        j = b1
        index_j = chia_mien
        while index_j >= 0:
            matrix_point[chia_mien - index_j][chia_mien - index_i] = (i, j)
            j += (b2 - b1) / chia_mien
            index_j -= 1
        i += (a2 - a1) / chia_mien
        index_i -= 1


def u_predict(x1, x2, xichma):
    """ Tính giá trị hàm u_predict tại điểm (x1,x2) """
    result = 0
    for k in range(N):
        result += np.float(np.array(xichma[k])) * (
                I1(x1, x2, a1 + (k + 1) * (a2 - a1) / N) - I1(x1, x2, a1 + k * (a2 - a1) / N))
        result += np.float(np.array(xichma[N + k])) * (
                I2(x1, x2, b1 + (k + 1) * (b2 - b1) / N) - I2(x1, x2, b1 + k * (b2 - b1) / N))
        result += np.float(np.array(xichma[2 * N + k])) * (
                I3(x1, x2, a1 + (k + 1) * (a2 - a1) / N) - I3(x1, x2, a1 + k * (a2 - a1) / N))
        result += np.float(np.array(xichma[3 * N + k])) * (
                I4(x1, x2, b1 + (k + 1) * (b2 - b1) / N) - I4(x1, x2, b1 + k * (b2 - b1) / N))
    return result


def triangle(x1, x2, x3, xichma, epxilon, w):
    """Tính sai số trong 1 miền tam giác với 3 đỉnh x1, x2, x3"""

    # Tính khoảng cách giữa 2 điểm x1 và x2
    def distance(x1, x2):
        return np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)

    result = 0
    point = np.zeros(2, dtype=object)
    for i in range(7):
        point[0] = x1[0] + epxilon[i][0] * (x2[0] - x1[0]) + epxilon[i][1] * (x3[0] - x1[0])
        point[1] = x1[1] + epxilon[i][0] * (x2[1] - x1[1]) + epxilon[i][1] * (x3[1] - x1[1])
        result += w[i] * (u(point[0], point[1]) - u_predict(point[0], point[1], xichma)) ** 2
    a, b, c = distance(x2, x3), distance(x1, x3), distance(x1, x2)
    p = (a + b + c) / 2
    S = math.sqrt(p * (p - a) * (p - b) * (p - c))
    return result * S


def cal(num_process, k, conn, matrix_point, xichma, epxilon, w):
    """ Xử lý song song
    num_process: số processer chia
    k: process thứ k (k=1,2,..)
    conn: pipeline để send dữ liệu về main
    """
    result = 0
    step = chia_mien // num_process
    if k < num_process:
        for i in range(1 + (k - 1) * step, 1 + k * step):
            for j in range(1, chia_mien + 1):
                result += triangle(matrix_point[i - 1][j - 1], matrix_point[i - 1][j], matrix_point[i][j], xichma,
                                   epxilon, w) + triangle(matrix_point[i - 1][j - 1], matrix_point[i][j - 1],
                                                          matrix_point[i][j], xichma, epxilon, w)
    else:
        for i in range(1 + (num_process - 1) * step, chia_mien + 1):
            for j in range(1, chia_mien + 1):
                result += triangle(matrix_point[i - 1][j - 1], matrix_point[i - 1][j], matrix_point[i][j], xichma,
                                   epxilon, w) + triangle(matrix_point[i - 1][j - 1], matrix_point[i][j - 1],
                                                          matrix_point[i][j], xichma, epxilon, w)
    conn.send(result)


def error(matrix_point, xichma, epxilon, w):
    """ Tính sai số L2 trên toàn miền """
    result = 0
    parent_conn, child_conn = Pipe()
    p = []
    for i in range(1, num_process + 1):
        pi = Process(target=cal, args=(num_process, i, child_conn, matrix_point, xichma, epxilon, w))
        p.append(pi)
        pi.start()
    for pi in p:
        pi.join()
        result += parent_conn.recv()
    return math.sqrt(result)


def plot_u(num_element=100):
    x1 = np.linspace(a1, a2, num=num_element)
    x2 = np.linspace(b1, b2, num=num_element)
    X, Y = np.meshgrid(x1, x2)
    Z = u(X, Y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('u')
    Z_predict = np.zeros((num_element, num_element))

    for i in range(num_element):
        for j in range(num_element):
            Z_predict[i][j] = u_predict(X[i][j], Y[i][j], xichma)

    #ax.plot_wireframe(X, Y, Z, color='green')
    ax.plot_wireframe(X, Y, Z_predict, color='blue')
    plt.show()


if __name__ == '__main__':
    init_point()
    init_g()
    init_V()
    start = time.perf_counter()
    xichma = np.linalg.solve(V, g)
    # xichma = np.dot(np.linalg.inv(V), g)
    end = time.perf_counter()
    time1 = end - start
    print("Thời gian giải hệ:", time1)

    #plot_u()

    init_gauss_2d()
    init_maxtrix_point()

    start = time.perf_counter()
    err = error(matrix_point, xichma, epxilon, w)
    end = time.perf_counter()
    time2 = end - start
    print("Sai số L2:", err)
    print("Thời gian tính sai số L2:", time2)

    print("Sai số tại điểm (4,1) là:", u_predict(4, 1, xichma))
    print("Tổng thời gian:", time1 + time2)
