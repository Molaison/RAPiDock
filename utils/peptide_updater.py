##########################################################################
# File Name: peptide_updater.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Fri 20 Oct 2023 11:09:44 AM CST
#########################################################################

import copy
import torch
import numpy as np
from scipy.spatial.transform import Rotation
# from utils.geometry import kabsch_torch, axis_angle_to_matrix

def axis_angle_to_matrix_torch(axis_angle: torch.Tensor):
    # axis_angle: (3,) or (N, 3)
    if axis_angle.dim() == 1:
        angle = torch.norm(axis_angle)
        if angle < 1e-6:
            return torch.eye(3)
        axis = axis_angle / angle
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        ux, uy, uz = axis
        R = torch.tensor([
            [cos + ux * ux * (1 - cos), ux * uy * (1 - cos) - uz * sin, ux * uz * (1 - cos) + uy * sin],
            [uy * ux * (1 - cos) + uz * sin, cos + uy * uy * (1 - cos), uy * uz * (1 - cos) - ux * sin],
            [uz * ux * (1 - cos) - uy * sin, uz * uy * (1 - cos) + ux * sin, cos + uz * uz * (1 - cos)]
        ])
        return R
    elif axis_angle.dim() == 2:
        R_list = []
        for i in range(axis_angle.size(0)):
            R_list.append(axis_angle_to_matrix_torch(axis_angle[i]))
        return torch.stack(R_list, dim=0)
    else:
        raise ValueError("axis_angle must be of shape (3,) or (N, 3)")

def kabsch_torch( P,Q):
    # P, Q: (3, N)
    assert P.shape[0] == 3 and Q.shape[0] == 3
    assert P.shape[1] == Q.shape[1]
    
    # center the points
    P_centered = P - P.mean(dim=1, keepdim=True)
    Q_centered = Q - Q.mean(dim=1, keepdim=True)
    
    # covariance matrix (注意顺序)
    C = Q_centered @ P_centered.T
    
    # SVD (使用新的API)
    U, S, Vt = torch.linalg.svd(C)
    
    # 确保proper rotation (det(R) = 1)
    d = torch.det(U @ Vt)
    if d < 0:
        Vt[-1, :] *= -1
    
    # rotation matrix
    R = U @ Vt
    
    # translation
    t = Q.mean(dim=1) - R @ P.mean(dim=1)
    
    return R, t

def peptide_updater(data, tr_update, rot_update, torsion_backbone_updates, torsion_sidechain_updates):
    pep_a_center = torch.mean(data['pep_a'].pos, dim=0, keepdim=True)
    rot_mat = axis_angle_to_matrix_torch(rot_update.squeeze())
    rigid_new_pos = (data['pep_a'].pos - pep_a_center) @ rot_mat.T + tr_update + pep_a_center
    # select edges to modify (torsion angles only)
    edge_index = data["pep_a", "pep_a"].edge_index
    edge_backbone_index = edge_index.T[data["pep_a"].mask_edges_backbone.squeeze()]
    edge_sidechain_index = edge_index.T[data["pep_a"].mask_edges_sidechain.squeeze()]
    # mask which to update
    mask_rotate_backbone = torch.tensor(data["pep_a"].mask_rotate_backbone) if torch.tensor(data["pep_a"].mask_rotate_backbone).dim() == 2 else torch.tensor(data["pep_a"].mask_rotate_backbone).squeeze(dim=0)
    mask_rotate_sidechain = torch.tensor(data["pep_a"].mask_rotate_sidechain) if torch.tensor(data["pep_a"].mask_rotate_sidechain).dim() == 2 else torch.tensor(data["pep_a"].mask_rotate_sidechain).squeeze(dim=0)
    flexible_new_pos = apply_torsion_updates(
            rigid_new_pos, edge_backbone_index, edge_sidechain_index, mask_rotate_backbone, mask_rotate_sidechain,
            torsion_backbone_updates, torsion_sidechain_updates)
    # fix orientation to disentangle torsion update
    R, t = kabsch_torch(flexible_new_pos.T, rigid_new_pos.T)
    aligned_flexible_pos = flexible_new_pos @ R.T + t.T
    data['pep_a'].pos = aligned_flexible_pos
    return data

def apply_torsion_updates(pos, edge_backbone_index, edge_sidechain_index, mask_rotate_backbone, mask_rotate_sidechain, torsion_backbone_updates, torsion_sidechain_updates):
    pos = copy.deepcopy(pos)
    if type(pos) != np.ndarray: pos = pos.cpu().numpy()
    
    ## backbone
    for idx_edge, e in enumerate(edge_backbone_index.cpu().numpy()):
        if torsion_backbone_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]
        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate_backbone[idx_edge, u]
        assert mask_rotate_backbone[idx_edge, v]
        
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_backbone_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        pos[mask_rotate_backbone[idx_edge]] = (pos[mask_rotate_backbone[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
    
    ## sidechian
    for idx_edge, e in enumerate(edge_sidechain_index.cpu().numpy()):
        if torsion_sidechain_updates[idx_edge] == 0:
            continue
        u, v = e[0], e[1]
        # check if need to reverse the edge, v should be connected to the part that gets rotated
        assert not mask_rotate_sidechain[idx_edge, u]
        assert mask_rotate_sidechain[idx_edge, v]
        
        rot_vec = pos[u] - pos[v]  # convention: positive rotation if pointing inwards
        rot_vec = rot_vec * torsion_sidechain_updates[idx_edge] / np.linalg.norm(rot_vec) # idx_edge!
        rot_mat = Rotation.from_rotvec(rot_vec).as_matrix()
        
        pos[mask_rotate_sidechain[idx_edge]] = (pos[mask_rotate_sidechain[idx_edge]] - pos[v]) @ rot_mat.T + pos[v]
    
    pos = torch.from_numpy(pos.astype(np.float32))
    return pos

def randomize_position(data_list, no_random, tr_sigma_max):
    # in place modification of the list
    # randomize torsion angles
    for complex_graph in data_list:
        torsion_updates_backbone = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['pep_a'].mask_edges_backbone.sum())
        torsion_updates_sidechain = np.random.uniform(low=-np.pi, high=np.pi, size=complex_graph['pep_a'].mask_edges_sidechain.sum())
        complex_graph['pep_a'].pos = \
            apply_torsion_updates(complex_graph['pep_a'].pos, complex_graph['pep_a', 'pep_a'].edge_index.T[complex_graph['pep_a'].mask_edges_backbone], complex_graph['pep_a', 'pep_a'].edge_index.T[complex_graph['pep_a'].mask_edges_sidechain], complex_graph['pep_a'].mask_rotate_backbone, complex_graph['pep_a'].mask_rotate_sidechain, torsion_updates_backbone, torsion_updates_sidechain)

    for complex_graph in data_list:
        # randomize position
        molecule_center = torch.mean(complex_graph['pep_a'].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(Rotation.random().as_matrix()).float()
        complex_graph['pep_a'].pos = (complex_graph['pep_a'].pos - molecule_center) @ random_rotation.T

        if not no_random:  # note for now the torsion angles are still randomised
            tr_update = torch.normal(mean=0, std=tr_sigma_max, size=(1, 3))
            complex_graph['pep_a'].pos += tr_update
            