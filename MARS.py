# if you would like to use the code, please cite
# Xinglong Ju, and Victoria C. P. Chen. "A MARS Python version using truncated linear function.", 2019. 
# The DOI can be found here at https://github.com/JuXinglong/WFLOP_Python
# from statistics import mean
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from sklearn import preprocessing
import matplotlib.cm as cm


class MARS:
    SST = None
    # SST_D_n_1 = None  # SST / (n-1) to calculate r_square_adjust
    SSE = None
    LOF = None
    r_square = 0
    r_square_adjust = 0
    static_knots = {}  # {0:[possible values for variable 1],1:[-1.61,-1.5,0,1.2]}
    knot_index_step = []
    n_basis_fn = 0  # number of basis functions
    basis_fns = []
    coefficients = None
    auto_stop = False
    y_bar = None
    x_middles = None
    x_half_ranges = None
    y_mean = None
    y_scale = None

    x_original = None
    y_original = None

    def __init__(self, n_variables=None, n_points=None, x=None, y=None, n_candidate_knots=[], n_max_basis_functions=0,
                 n_max_interactions=2,
                 difference=0.000002):
        self.n_variables = n_variables
        self.n_points = n_points
        if x is not None:
            self.x_original = x
            self.y_original = y

            x_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(x)
            self.x_middles = (x_scaler.data_max_ + x_scaler.data_min_) * 0.5
            self.x_half_ranges = x_scaler.data_range_ * 0.5

            self.x = x_scaler.transform(x)

        if y is not None:
            self.y = y
            self.y_mean = [0]
            self.y_scale = [1]


        self.n_candidate_knots = n_candidate_knots
        self.n_max_basis_functions = n_max_basis_functions
        if self.n_max_basis_functions == 0:
            self.auto_stop = True
        self.n_max_interactions = n_max_interactions
        self.difference = difference



    def X_inverse_scale(self, x_scaled):
        n = len(x_scaled)

        x_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        x_scaler.scale_ = np.zeros((self.n_variables), np.float64)
        for i in range(self.n_variables):
            x_scaler.scale_[i] = 1.0 / self.x_half_ranges[i]
        x_scaler.min_ = np.zeros((self.n_variables), np.float64)
        for i in range(self.n_variables):
            x_scaler.min_[i] = -self.x_middles[i] / self.x_half_ranges[i]
        return x_scaler.inverse_transform(x_scaled)

    def predict(self, x_new):
        n = len(x_new)
        # x_scaler = preprocessing.StandardScaler()
        x_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        # x_scaler.mean_ = self.x_means
        # x_scaler.scale_ = self.x_scales
        x_scaler.scale_ = np.zeros((self.n_variables), np.float64)
        for i in range(self.n_variables):
            x_scaler.scale_[i] = 1.0 / self.x_half_ranges[i]
        x_scaler.min_ = np.zeros((self.n_variables), np.float64)
        for i in range(self.n_variables):
            x_scaler.min_[i] = -self.x_middles[i] / self.x_half_ranges[i]
        x_new = x_scaler.transform(x_new)
        x_matrix = np.zeros((n, 1 + self.n_basis_fn), np.float64)
        for ind_bf in range(1 + self.n_basis_fn):
            for ind_x in range(n):
                x_matrix[ind_x, ind_bf] = self.basis_fns[ind_bf].cal_bf_value(x_new[ind_x])

        y = np.matmul(x_matrix, self.coefficients)

        return y

    def save_mars_model_to_file(self):

        self.SST = None
        self.SSE = None
        self.LOF = None
        self.r_square = None
        self.r_square_adjust = None
        self.static_knots = None
        self.knot_index_step = None
        self.auto_stop = None
        self.y_bar = None
        self.x_original = None
        self.y_original = None
        return



    def MARS_regress(self):
        self.y_bar = np.mean(self.y[:, 0])
        self.cal_SST()
        self.init_LOF_SSE_R_square()
        self.get_static_knots()
        self.cal_knot_index_step()

        bf = BasisFunction()
        self.basis_fns.append(bf)

        m_size = 100  # matrix size, columns of X matrix

        x_matrix = np.ones((self.n_points, m_size), np.float64)
        xTy_matrix = np.zeros((m_size, 1), np.float64)
        xTy_matrix[0, 0] = self.n_points * self.y_bar

        LTb_matrix = np.zeros((m_size, 1), np.float64)
        LTb_matrix[0, 0] = xTy_matrix[0, 0] / math.sqrt(self.n_points)

        b_matrix = np.zeros((m_size, 1), np.float64)
        b_matrix[0, 0] = self.y_bar

        xTx_matrix = np.zeros((m_size, m_size), np.float64)
        xTx_matrix[0, 0] = self.n_points
        L_matrix = np.zeros((m_size, m_size), np.float64)
        L_matrix[0, 0] = math.sqrt(xTx_matrix[0, 0])

        addon_size = 100
        while self.n_basis_fn < self.n_max_basis_functions or self.auto_stop:
            if m_size - self.n_basis_fn - 1 < 2:
                x_m_addon = np.array([[1] * addon_size] * self.n_points, np.float64)
                x_matrix = np.append(x_matrix, x_m_addon, axis=1)

                xTy_matrix_addon = np.array([[0]] * addon_size, np.float64)
                xTy_matrix = np.append(xTy_matrix, xTy_matrix_addon, axis=0)

                LTb_matrix_addon = np.array([[0]] * addon_size, np.float64)
                LTb_matrix = np.append(LTb_matrix, LTb_matrix_addon, axis=0)

                b_matrix_addon = np.array([[0]] * addon_size, np.float64)
                b_matrix = np.append(b_matrix, b_matrix_addon, axis=0)

                xTx_matrix_addon_column = np.array([[0] * addon_size] * m_size, np.float64)
                xTx_matrix = np.append(xTx_matrix, xTx_matrix_addon_column, axis=1)
                xTx_matrix_addon_row = np.array([[0] * (addon_size + m_size)] * addon_size, np.float64)
                xTx_matrix = np.append(xTx_matrix, xTx_matrix_addon_row, axis=0)

                L_matrix_addon_column = np.array([[0] * addon_size] * m_size, np.float64)
                L_matrix = np.append(L_matrix, L_matrix_addon_column, axis=1)
                L_matrix_addon_row = np.array([[0] * (addon_size + m_size)] * addon_size, np.float64)
                L_matrix = np.append(L_matrix, L_matrix_addon_row, axis=0)

                m_size += addon_size

            is_nbf_valid = False  # is new basis function valid (nbf new basis function)
            bs_split_bf_index = -1  # best selected split function index (parent function index) (bs best selected)
            bs_variable_index = -1  # best selected variable index
            bs_knot_value = 0  # best selected knot value
            bs_knot_index = -1  # best selected knot index
            bs_positive_valid = False
            bs_negative_valid = False
            bs_LOF = self.LOF  # best selected lack of fit
            new_LOF = self.LOF  #
            for ind_bf in range(self.n_basis_fn + 1):
                if self.basis_fns[ind_bf].order >= self.n_max_interactions:
                    continue
                selected_x_column = np.zeros((self.n_points, 1), np.float64)
                selected_x_column[:, 0] = x_matrix[:, ind_bf]
                for ind_variable in range(self.n_variables):
                    ind_max = int((len(self.static_knots[ind_variable]) - 1) / self.knot_index_step[ind_variable]) + 1
                    for ind in range(ind_max):
                        ind_knot = int(ind * self.knot_index_step[ind_variable])
                        if not self.is_knot_valid(self.basis_fns[ind_bf], ind_variable,
                                                  self.static_knots[ind_variable][ind_knot]):
                            continue
                        positive_knot_item = KnotItem(self.static_knots[ind_variable][ind_knot], ind_variable, +1)
                        negative_knot_item = KnotItem(self.static_knots[ind_variable][ind_knot], ind_variable, -1)
                        positive_x_column = np.zeros((self.n_points, 1), np.float64)
                        negative_x_column = np.zeros((self.n_points, 1), np.float64)
                        self.cal_new_x_column_val(selected_x_column, positive_x_column, positive_knot_item)
                        self.cal_new_x_column_val(selected_x_column, negative_x_column, negative_knot_item)
                        is_positive_x_column_valid = self.is_new_x_col_valid(positive_x_column)
                        is_negative_x_column_valid = self.is_new_x_col_valid(negative_x_column)
                        # print(self.static_knots[ind_variable][ind_knot])
                        if is_positive_x_column_valid and is_negative_x_column_valid:
                            x_matrix[:, self.n_basis_fn + 1] = positive_x_column[:, 0]
                            x_matrix[:, self.n_basis_fn + 2] = negative_x_column[:, 0]
                            self.update_xTx_matrix(x_matrix, xTx_matrix, 2)
                            self.update_xTy_matrix(x_matrix, xTy_matrix, 2)
                            if not self.update_L_matrix(xTx_matrix, L_matrix, 2):
                                continue
                            new_LOF = self.cal_new_LOF(L_matrix, LTb_matrix, xTy_matrix, b_matrix, x_matrix, 2)
                            if new_LOF < bs_LOF:
                                is_nbf_valid = True  # is new basis function valid (nbf new basis function)
                                bs_split_bf_index = ind_bf  # best selected split function index (parent function index) (bs best selected)
                                bs_variable_index = ind_variable  # best selected variable index
                                bs_knot_value = self.static_knots[ind_variable][ind_knot]  # best selected knot value
                                bs_knot_index = ind_knot  # best selected knot index
                                bs_positive_valid = True
                                bs_negative_valid = True
                                bs_LOF = new_LOF  # best selected lack of fit
                        elif is_positive_x_column_valid:
                            x_matrix[:, self.n_basis_fn + 1] = positive_x_column[:, 0]
                            self.update_xTx_matrix(x_matrix, xTx_matrix, 1)
                            self.update_xTy_matrix(x_matrix, xTy_matrix, 1)
                            if not self.update_L_matrix(xTx_matrix, L_matrix, 1):
                                continue
                            new_LOF = self.cal_new_LOF(L_matrix, LTb_matrix, xTy_matrix, b_matrix, x_matrix, 1)
                            if new_LOF < bs_LOF:
                                is_nbf_valid = True  # is new basis function valid (nbf new basis function)
                                bs_split_bf_index = ind_bf  # best selected split function index (parent function index) (bs best selected)
                                bs_variable_index = ind_variable  # best selected variable index
                                bs_knot_value = self.static_knots[ind_variable][ind_knot]  # best selected knot value
                                bs_knot_index = ind_knot  # best selected knot index
                                bs_positive_valid = True
                                bs_negative_valid = False
                                bs_LOF = new_LOF  # best selected lack of fit
                            # print(xTy_matrix)
                            # print(L_matrix)
                            # print(xTx_matrix)
                            # print(x_matrix)
                        elif is_negative_x_column_valid:
                            x_matrix[:, self.n_basis_fn + 1] = negative_x_column[:, 0]
                            self.update_xTx_matrix(x_matrix, xTx_matrix, 1)
                            self.update_xTy_matrix(x_matrix, xTy_matrix, 1)
                            if not self.update_L_matrix(xTx_matrix, L_matrix, 1):
                                continue
                            new_LOF = self.cal_new_LOF(L_matrix, LTb_matrix, xTy_matrix, b_matrix, x_matrix, 1)
                            if new_LOF < bs_LOF:
                                is_nbf_valid = True  # is new basis function valid (nbf new basis function)
                                bs_split_bf_index = ind_bf  # best selected split function index (parent function index) (bs best selected)
                                bs_variable_index = ind_variable  # best selected variable index
                                bs_knot_value = self.static_knots[ind_variable][ind_knot]  # best selected knot value
                                bs_knot_index = ind_knot  # best selected knot index
                                bs_positive_valid = False
                                bs_negative_valid = True
                                bs_LOF = new_LOF  # best selected lack of fit
                            # print(xTy_matrix)
                            # print(L_matrix)
                        else:
                            pass
            if is_nbf_valid:

                if bs_positive_valid and bs_negative_valid:
                    new_r_square = 1 - bs_LOF / self.SST
                    new_r_square_adjust = 1 - (bs_LOF / self.SST) * (
                            (self.n_points - 1) / (self.n_points - self.n_basis_fn - 2 - 1))
                    if new_r_square - self.r_square < self.difference or new_r_square_adjust - self.r_square_adjust < 0:
                        print("AUTO STOP")
                        break

                    self.LOF = bs_LOF
                    self.SSE = bs_LOF
                    self.r_square = new_r_square
                    self.r_square_adjust = new_r_square_adjust

                    positive_knot_item = KnotItem(bs_knot_value, bs_variable_index, +1)
                    negative_knot_item = KnotItem(bs_knot_value, bs_variable_index, -1)
                    self.update_x_matrix(x_matrix, bs_split_bf_index, self.n_basis_fn + 1, positive_knot_item)
                    self.update_x_matrix(x_matrix, bs_split_bf_index, self.n_basis_fn + 2, negative_knot_item)
                    self.update_xTx_matrix(x_matrix, xTx_matrix, 2)
                    self.update_xTy_matrix(x_matrix, xTy_matrix, 2)
                    self.update_L_matrix(xTx_matrix, L_matrix, 2)
                    self.update_LTb_matrix(L_matrix, LTb_matrix, xTy_matrix, 2)
                    self.n_basis_fn += 2
                    new_p_bf = BasisFunction()
                    new_p_bf.copy_basis_function(self.basis_fns[bs_split_bf_index])
                    new_p_bf.add_knot_item(positive_knot_item)
                    self.basis_fns.append(new_p_bf)
                    new_n_bf = BasisFunction()
                    new_n_bf.copy_basis_function(self.basis_fns[bs_split_bf_index])
                    new_n_bf.add_knot_item(negative_knot_item)
                    self.basis_fns.append(new_n_bf)

                    print("PB:", bs_split_bf_index, "V:", bs_variable_index, "K:", bs_knot_value, "R2:", self.r_square,
                          "AR2:", self.r_square_adjust, "NB:", self.n_basis_fn, "Pair")

                elif bs_positive_valid:
                    new_r_square = 1 - bs_LOF / self.SST
                    new_r_square_adjust = 1 - (bs_LOF / self.SST) * (
                            (self.n_points - 1) / (self.n_points - self.n_basis_fn - 1 - 1))
                    if new_r_square - self.r_square < self.difference or new_r_square_adjust - self.r_square_adjust < 0:
                        print("AUTO STOP")
                        break

                    self.LOF = bs_LOF
                    self.SSE = bs_LOF
                    self.r_square = new_r_square
                    self.r_square_adjust = new_r_square_adjust

                    positive_knot_item = KnotItem(bs_knot_value, bs_variable_index, +1)
                    self.update_x_matrix(x_matrix, bs_split_bf_index, self.n_basis_fn + 1, positive_knot_item)
                    self.update_xTx_matrix(x_matrix, xTx_matrix, 1)
                    self.update_xTy_matrix(x_matrix, xTy_matrix, 1)
                    self.update_L_matrix(xTx_matrix, L_matrix, 1)
                    self.update_LTb_matrix(L_matrix, LTb_matrix, xTy_matrix, 1)
                    self.n_basis_fn += 1
                    new_p_bf = BasisFunction()
                    new_p_bf.copy_basis_function(self.basis_fns[bs_split_bf_index])
                    new_p_bf.add_knot_item(positive_knot_item)
                    self.basis_fns.append(new_p_bf)

                    print("PB:", bs_split_bf_index, "V:", bs_variable_index, "K:", bs_knot_value, "R2:", self.r_square,
                          "AR2:", self.r_square_adjust, "NB:", self.n_basis_fn, "Single")

                elif bs_negative_valid:
                    new_r_square = 1 - bs_LOF / self.SST
                    new_r_square_adjust = 1 - (bs_LOF / self.SST) * (
                            (self.n_points - 1) / (self.n_points - self.n_basis_fn - 1 - 1))
                    if new_r_square - self.r_square < self.difference or new_r_square_adjust - self.r_square_adjust < 0:
                        print("AUTO STOP")
                        break

                    self.LOF = bs_LOF
                    self.SSE = bs_LOF
                    self.r_square = new_r_square
                    self.r_square_adjust = new_r_square_adjust

                    negative_knot_item = KnotItem(bs_knot_value, bs_variable_index, -1)
                    self.update_x_matrix(x_matrix, bs_split_bf_index, self.n_basis_fn + 1, negative_knot_item)
                    self.update_xTx_matrix(x_matrix, xTx_matrix, 1)
                    self.update_xTy_matrix(x_matrix, xTy_matrix, 1)
                    self.update_L_matrix(xTx_matrix, L_matrix, 1)
                    self.update_LTb_matrix(L_matrix, LTb_matrix, xTy_matrix, 1)
                    self.n_basis_fn += 1
                    new_n_bf = BasisFunction()
                    new_n_bf.copy_basis_function(self.basis_fns[bs_split_bf_index])
                    new_n_bf.add_knot_item(negative_knot_item)
                    self.basis_fns.append(new_n_bf)

                    print("PB:", bs_split_bf_index, "V:", bs_variable_index, "K:", bs_knot_value, "R2:", self.r_square,
                          "AR2:", self.r_square_adjust, "NB:", self.n_basis_fn, "Single")

                else:
                    pass

            else:
                print("No new basis function found with lower lack of fit. Stop.")
                break
        # print(xTx_matrix[0:7, 0:7])
        self.cal_coefficients(L_matrix, LTb_matrix)
        # print(self.coefficients)
        return

    def cal_new_LOF(self, L_m, LTb_m, xTy_m, b_m, x_m, n_new_cols):
        self.update_LTb_matrix(L_m, LTb_m, xTy_m, n_new_cols)
        self.update_b_matrix(L_m, LTb_m, n_new_cols, b_m)
        return (np.sum((np.matmul(x_m[:, :1 + self.n_basis_fn + n_new_cols],
                                  b_m[:1 + self.n_basis_fn + n_new_cols, :]) - self.y) ** 2))
        # print(LTb_m)
        # print(b_m)
        return

    def cal_coefficients(self, L_m, LTb_m):
        self.coefficients = np.zeros((1 + self.n_basis_fn, 1), np.float64)
        for r_ind in range(1 + self.n_basis_fn):
            i = self.n_basis_fn - r_ind
            result = LTb_m[i, 0]
            for c_ind in range(r_ind):
                j = self.n_basis_fn - c_ind
                result -= self.coefficients[j, 0] * L_m[j, i]  # LT_matrix[i,j]=L_matrix[j,i] transpose
            self.coefficients[i, 0] = result / L_m[i, i]  # LT_matrix[i,i]=L_matrix[i,i] transpose
        return

    def update_b_matrix(self, L_m, LTb_m, n_new_cols, b_m):
        for r_ind in range(1 + self.n_basis_fn + n_new_cols):
            i = self.n_basis_fn + n_new_cols - r_ind
            result = LTb_m[i, 0]
            for c_ind in range(r_ind):
                j = self.n_basis_fn + n_new_cols - c_ind
                result -= b_m[j, 0] * L_m[j, i]  # LT_matrix[i,j]=L_matrix[j,i] transpose
            b_m[i, 0] = result / L_m[i, i]  # LT_matrix[i,i]=L_matrix[i,i] transpose
        return

    def update_LTb_matrix(self, L_m, LTb_m, xTy_m, n_new_cols):
        for i in range(self.n_basis_fn + 1, self.n_basis_fn + 1 + n_new_cols):
            result = xTy_m[i, 0]
            for j in range(i):
                result -= L_m[i, j] * LTb_m[j, 0]
            LTb_m[i, 0] = result / L_m[i, i]
        return

    def update_L_matrix(self, xTx_m, L_m, n_new_cols):
        for j in range(self.n_basis_fn + 1, self.n_basis_fn + n_new_cols + 1):
            L_m[j, 0] = xTx_m[j, 0] / L_m[0, 0]
            for i in range(1, j):
                result = xTx_m[j, i]
                for p in range(i):
                    result -= L_m[i, p] * L_m[j, p]
                L_m[j, i] = result / L_m[i, i]
            result = xTx_m[j, j]
            for p in range(j):
                result -= L_m[j, p] ** 2
            if result <= 1.0e-10:
                return False
            L_m[j, j] = math.sqrt(result)
        return True

    def update_xTy_matrix(self, x_m, xTy_m, n_new_cols):
        for i_new_col in range(n_new_cols):
            result = 0
            for ind in range(self.n_points):
                result += x_m[ind, self.n_basis_fn + i_new_col + 1] * self.y[ind, 0]
            xTy_m[self.n_basis_fn + i_new_col + 1, 0] = result
        return

    def update_xTx_matrix(self, x_m, xTx_m, n_new_cols):
        # n_new_cols 1 or 2
        for i_new_col in range(n_new_cols):
            for i_col in range(self.n_basis_fn + 1 + i_new_col + 1):
                if i_new_col == 1 and i_col == self.n_basis_fn + 1:
                    result = 0
                else:
                    result = 0
                    for i_row in range(self.n_points):
                        result += x_m[i_row, self.n_basis_fn + i_new_col + 1] * x_m[i_row, i_col]
                xTx_m[self.n_basis_fn + i_new_col + 1, i_col] = result
                xTx_m[i_col, self.n_basis_fn + i_new_col + 1] = result
        return

    def is_new_x_col_valid(self, new_x_col):
        for i in range(self.n_points):
            if new_x_col[i, 0] > 1.0e-10:
                return True
        return False

    def cal_new_x_column_val(self, p_bf_x_col, new_x_col, knot_item):
        for ind in range(self.n_points):
            new_x_col[ind, 0] = p_bf_x_col[ind, 0] * knot_item.cal_knot_item_value(self.x[ind])

    def update_x_matrix(self, x_m, p_bf_ind, new_x_col_ind, knot_item):
        for ind in range(self.n_points):
            x_m[ind, new_x_col_ind] = x_m[ind, p_bf_ind] * knot_item.cal_knot_item_value(self.x[ind])
        # print(new_x_col)

    def is_knot_valid(self, p_bf, variable_index, knot_value):
        # return True
        # p_bf parent basis function
        for bf in self.basis_fns:
            if bf.order == p_bf.order + 1:
                is_same = True
                for ki in p_bf.knot_items:
                    if not bf.is_knot_item_in(ki):
                        is_same = False
                        break
                if is_same:
                    if not bf.is_knot_in(knot_value, variable_index):
                        is_same = False
                if is_same:
                    return False
        return True

    def cal_knot_index_step(self):
        for i in range(self.n_variables):
            n = len(self.static_knots[i])
            if self.n_candidate_knots[i] == 0:
                step = 1
            else:
                step = (n - 1) / (self.n_candidate_knots[i] - 1)
                if step == 0:
                    step = 1
            self.knot_index_step.append(step)
        # print(self.n_candidate_knots)

    def init_LOF_SSE_R_square(self):
        self.LOF = self.SST
        self.SSE = self.SST
        self.r_square = 0.0
        self.r_square_adjust = 0.0
        return

    def cal_SST(self):
        self.SST = 0.0
        for i in range(self.n_points):
            self.SST += (self.y[i, 0] - self.y_bar) ** 2
        # self.SST_D_n_1 = self.SST / (self.n_points - 1)
        return

    # get all possible knots from the training data and sort from smallest to largest
    def get_static_knots(self):
        for i in range(self.n_variables):
            self.static_knots[i] = sorted(set(self.x[:, i]))
            # print(len(self.static_knots[i]))
            # print(self.static_knots[i])
            # self.get_static_knots_for_variable_i(self.x[:, i], self.static_knots[i])
        return



    def get_static_knots_for_variable_i(self, knot_values, sorted_knot_values):
        n_values = len(knot_values)
        for i in range(n_values):
            new_index = self.get_new_knot_index(sorted_knot_values, knot_values[i])
            if new_index == -1:
                pass
            elif new_index == len(sorted_knot_values):
                sorted_knot_values.append(knot_values[i])
            else:
                sorted_knot_values.insert(new_index, knot_values[i])
        return

    def get_new_knot_index(self, sorted_knot_values, new_knot_value):
        n_sorted = len(sorted_knot_values)
        start_index = 0
        end_index = n_sorted - 1
        middle_index = int((start_index + end_index) / 2)
        while start_index <= end_index:
            if new_knot_value - sorted_knot_values[middle_index] < 0:
                end_index = middle_index - 1
            elif new_knot_value - sorted_knot_values[middle_index] > 0:
                start_index = middle_index + 1
            else:
                return -1
            middle_index = int((start_index + end_index) / 2)
        return start_index


class KnotItem:
    def __init__(self, knot_value, index_of_variable, sign):
        self.knot_value = knot_value
        self.index_of_variable = index_of_variable
        self.sign = sign

    def cal_knot_item_value(self, x):
        result = 0
        if self.sign == 1:
            result = x[self.index_of_variable] - self.knot_value
        else:
            result = self.knot_value - x[self.index_of_variable]

        if result > 0:
            return result
        else:
            return 0


class BasisFunction:
    def __init__(self):
        self.order = 0
        self.knot_items = []

    def is_knot_item_in(self, t_ki):
        # ki knot item t_ki test knot item
        for ki in self.knot_items:
            if ki.knot_value == t_ki.knot_value and ki.index_of_variable == t_ki.index_of_variable:
                return True
        return False

    def is_knot_in(self, k_value, var_ind):
        #
        for ki in self.knot_items:
            if ki.knot_value == k_value and ki.index_of_variable == var_ind:
                return True
        return False

    def add_knot_item(self, new_knot_item):
        self.knot_items.append(new_knot_item)
        self.order += 1

    def copy_basis_function(self, bf):
        for i in range(bf.order):
            new_knot_item = KnotItem(bf.knot_items[i].knot_value, bf.knot_items[i].index_of_variable,
                                     bf.knot_items[i].sign)
            self.add_knot_item(new_knot_item)

    def cal_bf_value(self, x):
        result = 1.0
        for i in range(self.order):
            result *= self.knot_items[i].cal_knot_item_value(x)
        return result
