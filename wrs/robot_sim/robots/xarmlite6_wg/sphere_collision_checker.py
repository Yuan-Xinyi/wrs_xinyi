import jax
import jax.numpy as jnp
import xmltodict
import numpy as np
from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_util import axis_angle_to_matrix, xyzrpy_to_matrix

class SphereCollisionChecker:
    def __init__(self, urdf_path):
        self.urdf_path = urdf_path
        
        links_dict, joints_dict, self.link_order, self.base_link = self._parse_urdf_structure(urdf_path)
        self.sphere_offsets, self.sphere_radii, self.sphere_link_indices = self._prepare_sphere_data(links_dict)
        joint_data = self._prepare_joint_kinematics(joints_dict)
        self.joint_statics, self.joint_axes, self.joint_types, self.q_indices, self.parent_indices = joint_data

        self._jit_update = jax.jit(self.compute_sphere_positions)

        # collision related: mask is used to ignore spheres within the same link
        self.collision_mask = self._prepare_collision_masks(ignore_adjacent=True)
        self._jit_collision_cost = jax.jit(self.self_collision_cost)

    def _parse_urdf_structure(self, urdf_path):
        with open(urdf_path, 'r') as f:
            robot_data = xmltodict.parse(f.read())['robot']
        
        links = {l['@name']: l for l in robot_data['link']}
        joints = {j['@name']: j for j in robot_data['joint']}
        
        parent_map = {j['child']['@link']: j for j in joints.values()}
        base_link = [l for l in links if l not in parent_map][0]
        
        child_map = {}
        for j in joints.values():
            child_map.setdefault(j['parent']['@link'], []).append(j['child']['@link'])
            
        link_order, queue = [], [base_link]
        while queue:
            curr = queue.pop(0)
            link_order.append(curr)
            queue.extend(child_map.get(curr, []))
            
        return links, joints, link_order, base_link

    def _prepare_sphere_data(self, links_dict):
        offsets, radii, indices = [], [], []
        name_to_idx = {name: i for i, name in enumerate(self.link_order)}
        
        for name in self.link_order:
            link = links_dict[name]
            if 'collision' not in link: continue
            
            colls = link['collision']
            if not isinstance(colls, list): colls = [colls]
            
            for col in colls:
                xyz = [float(x) for x in col['origin']['@xyz'].split()]
                rpy = [float(x) for x in col['origin']['@rpy'].split()]
                offsets.append(xyzrpy_to_matrix(xyz, rpy))
                radii.append(float(col['geometry']['sphere']['@radius']))
                indices.append(name_to_idx[name])
        
        return jnp.array(offsets), jnp.array(radii), jnp.array(indices)

    def _prepare_joint_kinematics(self, joints_dict):
        movable = [j for j in joints_dict.values() if j.get('@type') in ['revolute', 'prismatic']]
        j_to_q = {j['@name']: i for i, j in enumerate(movable)}
        
        statics, axes, types, q_idxs, p_idxs = [], [], [], [], []
        name_to_idx = {name: i for i, name in enumerate(self.link_order)}

        for name in self.link_order:
            if name == self.base_link:
                statics.append(np.eye(4))
                axes.append(np.zeros(3))
                types.append(0)
                q_idxs.append(-1)
                p_indices = -1
            else:
                j = next(jt for jt in joints_dict.values() if jt['child']['@link'] == name)
                xyz = [float(x) for x in j['origin']['@xyz'].split()]
                rpy = [float(x) for x in j['origin']['@rpy'].split()]
                statics.append(xyzrpy_to_matrix(xyz, rpy))
                
                if '@type' in j and j['@type'] in ['revolute', 'prismatic']:
                    axes.append(np.array([float(x) for x in j['axis']['@xyz'].split()]))
                    types.append(1 if j['@type'] == 'revolute' else 2)
                    q_idxs.append(j_to_q[j['@name']])
                else:
                    axes.append(np.zeros(3))
                    types.append(0)
                    q_idxs.append(-1)
                
                p_indices = name_to_idx[j['parent']['@link']]
            
            p_idxs.append(p_indices)
        
        return jnp.array(statics), jnp.array(axes), np.array(types), np.array(q_idxs), np.array(p_idxs)

    def _prepare_collision_masks(self, ignore_adjacent=True):
            num_spheres = len(self.sphere_link_indices)
            id_i = self.sphere_link_indices[:, None]
            id_j = self.sphere_link_indices[None, :]
            
            mask = (id_i != id_j)

            if ignore_adjacent:
                adj_pairs = []
                for name in self.link_order:
                    if name == self.base_link: continue
                    child_idx = self.link_order.index(name)
                    parent_idx = self.parent_indices[child_idx]
                    if parent_idx != -1:
                        adj_pairs.append((parent_idx, child_idx))
                
                for p_idx, c_idx in adj_pairs:
                    mask &= ~((id_i == p_idx) & (id_j == c_idx))
                    mask &= ~((id_i == c_idx) & (id_j == p_idx))
            
            upper_tri = jnp.triu(jnp.ones((num_spheres, num_spheres), dtype=bool), k=1)
            return mask & upper_tri

    def compute_self_collision_dist(self, q):
        spheres = self.compute_sphere_positions(q)
        radii = self.sphere_radii
        
        diff = spheres[:, None, :] - spheres[None, :, :]
        dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        
        radii_sum_matrix = radii[:, None] + radii[None, :]
        margin_matrix = dist_matrix - radii_sum_matrix
        safe_dist = jnp.where(self.collision_mask, margin_matrix, 1e6)
        
        return jnp.min(safe_dist)

    def self_collision_cost(self, q, scale = 100, min_margin=0.01):
            spheres = self.compute_sphere_positions(q)
            diff = spheres[:, None, :] - spheres[None, :, :]
            dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
            radii_sum_matrix = self.sphere_radii[:, None] + self.sphere_radii[None, :]
            
            dist_net = dist_matrix - radii_sum_matrix
            
            collision_penalty = jax.nn.relu(min_margin - dist_net)
            
            total_cost = jnp.sum(collision_penalty * self.collision_mask) * scale
            return total_cost

    def check_collisions(self, q, margin=-0.005):
            spheres = self.compute_sphere_positions(q)
            
            diff = spheres[:, None, :] - spheres[None, :, :]
            dist_matrix = jnp.linalg.norm(diff, axis=-1)
            
            radii_sum = self.sphere_radii[:, None] + self.sphere_radii[None, :]
            margin_matrix = dist_matrix - radii_sum
            
            collision_pairs = (margin_matrix < margin) & self.collision_mask
            
            is_colliding = jnp.any(collision_pairs, axis=1) | jnp.any(collision_pairs, axis=0)
            
            return spheres, is_colliding

    def compute_sphere_positions(self, q):
        num_links = len(self.link_order)
        link_transforms = jnp.tile(jnp.identity(4), (num_links, 1, 1))
        
        for i in range(num_links):
            p_idx = int(self.parent_indices[i])
            if p_idx == -1:
                continue
            
            j_type = int(self.joint_types[i])
            q_idx = int(self.q_indices[i])
            
            T_motion = jnp.identity(4)
            if j_type == 1: # Revolute
                q_val = q[q_idx]
                T_motion = axis_angle_to_matrix(self.joint_axes[i], q_val)
            elif j_type == 2: # Prismatic
                q_val = q[q_idx]
                T_motion = T_motion.at[0:3, 3].set(self.joint_axes[i] * q_val)
            
            T_world_this = link_transforms[p_idx] @ self.joint_statics[i] @ T_motion
            link_transforms = link_transforms.at[i].set(T_world_this)

        sphere_link_ts = link_transforms[self.sphere_link_indices]
        final_sphere_ts = jnp.einsum('nij,njk->nik', sphere_link_ts, self.sphere_offsets)
        
        return final_sphere_ts[:, 0:3, 3]

    def update(self, q):
        return self._jit_update(q)