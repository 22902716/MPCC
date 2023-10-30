import numpy as np
import trajectory_planning_helpers as tph
import csv
import casadi as ca

class my_MPCC_util:
    
    def __init__(self, map_name, w):
        # self.width = width
        self.map_name = map_name
        self.path = None
        self.el_lengths = None 
        self.psi = None
        self.nvecs = None
        self.track_length = None
        self.init_path()

        self.center_lut_x, self.center_lut_y = None, None
        self.left_lut_x, self.left_lut_y = None, None
        self.right_lut_x, self.right_lut_y = None, None

        self.center_lut_x, self.center_lut_y = self.get_interpolated_path_casadi('lut_center_x', 'lut_center_y', self.path, self.s_track)
        # self.angle_lut_t = self.get_interpolated_heading_casadi('lut_angle_t', self.psi, self.s_track)

        left_path = self.path - self.nvecs * (self.track[:, 2][:, None]  - w)
        right_path = self.path + self.nvecs * (self.track[:, 3][:, None] - w)
        self.left_lut_x, self.left_lut_y = self.get_interpolated_path_casadi('lut_left_x', 'lut_left_y', left_path, self.s_track)
        self.right_lut_x, self.right_lut_y = self.get_interpolated_path_casadi('lut_right_x', 'lut_right_y', right_path, self.s_track)

    def init_path(self):
        filename = 'maps/' + self.map_name + '_centerline.csv'
        xs, ys, w_rs, w_ls = [], [], [], []
        with open(filename, 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))
                w_rs.append(float(lines[2]))
                w_ls.append(float(lines[3]))
        xs = np.array(xs)[:, None]
        ys = np.array(ys)[:, None]
        w_ls = np.array(w_ls)[:, None]
        w_rs = np.array(w_rs)[:, None]

        # the row stacking ensures that the track is continuous past the end.
        self.track = np.hstack((xs, ys, w_rs, w_ls))
        self.track = np.row_stack((self.track, self.track[1:int(self.track.shape[0] / 2), :]))

        self.path = np.hstack((xs, ys))
        self.path = np.row_stack((self.path, self.path[1:int(self.path.shape[0] / 2), :]))

        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)

        angle_diffs = np.diff(self.psi, axis=0)
        for i in range(len(angle_diffs)):
            if angle_diffs[i] > np.pi:
                self.psi[i+1:] -= 2*np.pi
            elif angle_diffs[i] < -np.pi:
                self.psi[i+1:] += 2*np.pi

        # self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi-np.pi/2) #original
        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi)
        # self.nvecs[:, [0, 1]] = self.nvecs[:, [1, 0]]
        # print(self.nvecs)
        # exit()
        self.track_length = self.s_track[-1]

    def get_interpolated_path_casadi(self, label_x, label_y, pts, arc_lengths_arr):
        u = arc_lengths_arr
        V_X = pts[:, 0]
        V_Y = pts[:, 1]
        lut_x = ca.interpolant(label_x, 'bspline', [u], V_X)
        lut_y = ca.interpolant(label_y, 'bspline', [u], V_Y)
        # print("lut_x",lut_x)
        # print("lut_y",lut_y)
        # print("pts: ",pts,"shape",pts.shape)
        # print("arc_lengths_arr",u)
        return lut_x, lut_y