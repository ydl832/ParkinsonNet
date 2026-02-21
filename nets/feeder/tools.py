import numpy as np
import random
import torch


def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]

def rot_matrix(angles):
    radians = angles * (np.pi / 180)
    # Helper function to generate a rotation matrix
    alpha, beta, gamma = radians
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    rotation_matrix = np.matmul(Rz, np.matmul(Ry, Rx))
    return rotation_matrix

def random_rotation(pose_sequence):
    # Randomly rotate the pose sequence
    rotation_angles = np.random.uniform(-10, 10, size=3)
    rotation_matrix = rot_matrix(rotation_angles)
    pose_sequence = np.transpose(pose_sequence, (1, 2, 3, 0))
    rotated_sequence = np.matmul(pose_sequence, rotation_matrix.T)
    rotated_sequence = np.transpose(rotated_sequence, (3, 0, 1, 2))
    return rotated_sequence
                
#angle_candidate=[-10., -5., 0., 5., 10.]
#def random_move(data_numpy,
#                angle_candidate=[-3., 0., 3.],
#                scale_candidate=[0.95, 1.0, 1.05],
#                transform_candidate=[-0.1, 0.0, 0.1],
#                move_time_candidate=[1]):
#    # input: C,T,V,M
#    C, T, V, M = data_numpy.shape
#    move_time = random.choice(move_time_candidate)
#    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
#    node = np.append(node, T)
#    num_node = len(node)
#
#    A = np.random.choice(angle_candidate, num_node)
#    S = np.random.choice(scale_candidate, num_node)
#    T_x = np.random.choice(transform_candidate, num_node)
#    T_y = np.random.choice(transform_candidate, num_node)
#
#    a = np.zeros(T)
#    s = np.zeros(T)
#    t_x = np.zeros(T)
#    t_y = np.zeros(T)
#
#    # linspace
#    for i in range(num_node - 1):
#        a[node[i]:node[i + 1]] = np.linspace(
#            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
#        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
#                                             node[i + 1] - node[i])
#        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
#                                               node[i + 1] - node[i])
#        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
#                                               node[i + 1] - node[i])
#
#    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
#                      [np.sin(a) * s, np.cos(a) * s]])
#
#    # perform transformation
#    for i_frame in range(T):
#        xy = data_numpy[0:2, i_frame, :, :]
#        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
#        new_xy[0] += t_x[i_frame]
#        new_xy[1] += t_y[i_frame]
#        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
#
#    return data_numpy

def random_move(data_numpy,
                angle_candidate=[-5., 0., 5.],
                scale_candidate=[0.95, 1.0, 1.05],
                transform_candidate=[-0.01, 0.0, 0.01],
                move_time_candidate=[1]):
    # input: C, T, V, M (C=3 for x, y, z coordinates)
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A_x = np.random.choice(angle_candidate, num_node)  # Rotation angles around x-axis
    A_y = np.random.choice(angle_candidate, num_node)  # Rotation angles around y-axis
    A_z = np.random.choice(angle_candidate, num_node)  # Rotation angles around z-axis
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)
    T_z = np.random.choice(transform_candidate, num_node)

    a_x = np.zeros(T)
    a_y = np.zeros(T)
    a_z = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    t_z = np.zeros(T)

    # linspace for smooth transitions
    for i in range(num_node - 1):
        a_x[node[i]:node[i + 1]] = np.linspace(A_x[i], A_x[i + 1], node[i + 1] - node[i]) * np.pi / 180
        a_y[node[i]:node[i + 1]] = np.linspace(A_y[i], A_y[i + 1], node[i + 1] - node[i]) * np.pi / 180
        a_z[node[i]:node[i + 1]] = np.linspace(A_z[i], A_z[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
        t_z[node[i]:node[i + 1]] = np.linspace(T_z[i], T_z[i + 1], node[i + 1] - node[i])

    # Rotation matrices for each axis
    def get_rotation_matrix(a_x, a_y, a_z, s):
        # Rotation around x-axis
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(a_x), -np.sin(a_x)],
                        [0, np.sin(a_x), np.cos(a_x)]])
        # Rotation around y-axis
        R_y = np.array([[np.cos(a_y), 0, np.sin(a_y)],
                        [0, 1, 0],
                        [-np.sin(a_y), 0, np.cos(a_y)]])
        # Rotation around z-axis
        R_z = np.array([[np.cos(a_z), -np.sin(a_z), 0],
                        [np.sin(a_z), np.cos(a_z), 0],
                        [0, 0, 1]])
        # Final rotation matrix (apply in the order: z -> y -> x)
        R = np.dot(np.dot(R_z, R_y), R_x)
        # Apply scaling
        return R * s

    # Apply transformation for each frame
    for i_frame in range(T):
        xyz = data_numpy[0:3, i_frame, :, :]  # Get x, y, z coordinates
        R = get_rotation_matrix(a_x[i_frame], a_y[i_frame], a_z[i_frame], s[i_frame])
        new_xyz = np.dot(R, xyz.reshape(3, -1))  # Apply rotation and scaling
        new_xyz[0] += t_x[i_frame]
        new_xyz[1] += t_y[i_frame]
        new_xyz[2] += t_z[i_frame]
        data_numpy[0:3, i_frame, :, :] = new_xyz.reshape(3, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def mirroring(data_numpy):
    C, T, V, M = data_numpy.shape
    for i in range(0, V):
        data_numpy[0, :, i, :] = (data_numpy[0, :, 2, :] - data_numpy[0, :, i, :]) * 2 + data_numpy[0, :, i, :]
        # data_numpy[0, :, i, :] =  - data_numpy[0, :, i, :]

    return data_numpy


def mirroring_v1(data_numpy):
    C, T, V, M = data_numpy.shape
    data_numpy[0, :, :, :] = np.max(data_numpy[0, :, :, :]) + np.min(data_numpy[0, :, :, :]) - data_numpy[
                                                                                                     0,
                                                                                                     :, :, :]
    return data_numpy

#def mirror_reflection(pose_sequence):
#    mirrored_sequence = pose_sequence.copy()
#    left = [4, 5, 6, 10, 11, 12]
#    right = [7, 8, 9, 13, 14, 15]
#    mirrored_sequence[:, :, 0] *= -1
#    mirrored_sequence[:, left + right, :] = mirrored_sequence[:, right + left, :]
#    return mirrored_sequence
    
def mirror_reflection(data_numpy):
    mirrored_sequence = data_numpy.copy()
    left = [4, 5, 6, 10, 11, 12]
    right = [7, 8, 9, 13, 14, 15]
    mirrored_sequence[0, :, :, :] *= -1
    mirrored_sequence[:, :, left + right, :] = mirrored_sequence[:, :, right + left, :]

    return mirrored_sequence


def random_noise(data_numpy, mean=0, std=0.01):

    if isinstance(data_numpy, np.ndarray):
        data_numpy = torch.from_numpy(data_numpy)

    noise = torch.normal(mean, std, size=data_numpy.shape)
    noise_sequence = data_numpy + noise

    return noise_sequence


def axis_mask(data_numpy, data_dim=3):
    
    def zero_out_axis(data_numpy, data_dim):
        axis_to_mask = random.randint(0, data_dim - 1)
    
        data_numpy[axis_to_mask, :, :, :] = 0
        return data_numpy

    if isinstance(data_numpy, np.ndarray):
        data_numpy = torch.from_numpy(data_numpy)

    if data_numpy.shape[0] != data_dim:
        raise ValueError(f"{data_numpy.shape[0]} data_dim {data_dim}")

    masked_sequence = zero_out_axis(data_numpy, data_dim)
    return masked_sequence


def joint_dropout(pose_sequence, dropout_prob=0.2):
    
    C, T, V, M = pose_sequence.shape

    # ????????
    dropout_mask = np.random.choice([0, 1], size=(C, T, V, 1), p=[dropout_prob, 1 - dropout_prob])

    # ??????
    dropped_sequence = pose_sequence * dropout_mask
    
    return dropped_sequence


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
                                                             t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy

    # # match poses between 2 frames
    # if self.pose_matching:
    #     C, T, V, M = data_numpy.shape
    #     forward_map = np.zeros((T, M), dtype=int) - 1
    #     backward_map = np.zeros((T, M), dtype=int) - 1

    #     # match pose
    #     for t in range(T - 1):
    #         for m in range(M):
    #             s = (data_numpy[2, t, :, m].reshape(1, V, 1) != 0) * 1
    #             if s.sum() == 0:
    #                 continue
    #             res = data_numpy[:, t + 1, :, :] - data_numpy[:, t, :,
    #                                                           m].reshape(
    #                                                               C, V, 1)
    #             n = (res * res * s).sum(axis=1).sum(
    #                 axis=0).argsort()[0]  #next pose
    #             forward_map[t, m] = n
    #             backward_map[t + 1, n] = m

    #     # find start point
    #     start_point = []
    #     for t in range(T):
    #         for m in range(M):
    #             if backward_map[t, m] == -1:
    #                 start_point.append((t, m))

    #     # generate path
    #     path_list = []
    #     c = 0
    #     for i in range(len(start_point)):
    #         path = [start_point[i]]
    #         while (1):
    #             t, m = path[-1]
    #             n = forward_map[t, m]
    #             if n == -1:
    #                 break
    #             else:
    #                 path.append((t + 1, n))
    #             #print(c,t)
    #             c = c + 1
    #         path_list.append(path)

    #     # generate data
    #     new_M = self.num_match_trace
    #     path_length = [len(p) for p in path_list]
    #     sort_index = np.array(path_length).argsort()[::-1][:new_M]
    #     if self.mode == 'train':
    #         np.random.shuffle(sort_index)
    #         sort_index = sort_index[:M]
    #         new_data_numpy = np.zeros((C, T, V, M))
    #     else:
    #         new_data_numpy = np.zeros((C, T, V, new_M))
    #     for i, p in enumerate(sort_index):
    #         path = path_list[p]
    #         for t, m in path:
    #             new_data_numpy[:, t, :, i] = data_numpy[:, t, :, m]

    #     data_numpy = new_data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


#def uniform_sample_np(data_numpy, size):
#    C, T, V, M = data_numpy.shape
#    if T == size:
#        return data_numpy
#    interval = T / size
#    uniform_list = [int(i * interval) for i in range(size)]
#    return data_numpy[:, uniform_list]
#    
#    
#    def valid_crop_resize(data_numpy,valid_frame_num,p_interval,window):
#    # input: C,T,V,M
#    C, T, V, M = data_numpy.shape
#    begin = 0
#    end = valid_frame_num
#    valid_size = end - begin
#
#    #crop
#    if len(p_interval) == 1:
#        p = p_interval[0]
#        bias = int((1-p) * valid_size/2)
#        data = data_numpy[:, begin+bias:end-bias, :, :]# center_crop
#        cropped_length = data.shape[1]
#    else:
#        p = np.random.rand(1)*(p_interval[1]-p_interval[0])+p_interval[0]
#        cropped_length = np.minimum(np.maximum(int(np.floor(valid_size*p)),64), valid_size)# constraint cropped_length lower bound as 64
#        bias = np.random.randint(0,valid_size-cropped_length+1)
#        data = data_numpy[:, begin+bias:begin+bias+cropped_length, :, :]
#        if data.shape[1] == 0:
#            print(cropped_length, bias, valid_size)
#
#    # resize
#    data = torch.tensor(data,dtype=torch.float)
#    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
#    data = data[None, None, :, :]
#    data = F.interpolate(data, size=(C * V * M, window), mode='bilinear',align_corners=False).squeeze() # could perform both up sample and down sample
#    data = data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
#
#    return data
