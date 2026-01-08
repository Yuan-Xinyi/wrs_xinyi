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
        self.joint_statics, self.joint_axes, self.joint_types, self.q_indices, self.parent_indices = self._prepare_joint_kinematics(joints_dict)

        self.collision_mask = self._prepare_collision_masks(ignore_adjacent=True)
        
        self._jit_update = jax.jit(self.compute_sphere_positions)
        self._jit_collision_dist = jax.jit(self.compute_self_collision_dist)
        self._jit_collision_cost = jax.jit(self.self_collision_cost)
        self._jit_check_collisions = jax.jit(self.check_collisions)

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
            colls = link['collision'] if isinstance(link['collision'], list) else [link['collision']]
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
                statics.append(np.eye(4)); axes.append(np.zeros(3)); types.append(0); q_idxs.append(-1); p_indices = -1
            else:
                j = next(jt for jt in joints_dict.values() if jt['child']['@link'] == name)
                statics.append(xyzrpy_to_matrix([float(x) for x in j['origin']['@xyz'].split()], [float(x) for x in j['origin']['@rpy'].split()]))
                if '@type' in j and j['@type'] in ['revolute', 'prismatic']:
                    axes.append(np.array([float(x) for x in j['axis']['@xyz'].split()]))
                    types.append(1 if j['@type'] == 'revolute' else 2)
                    q_idxs.append(j_to_q[j['@name']])
                else:
                    axes.append(np.zeros(3)); types.append(0); q_idxs.append(-1)
                p_indices = name_to_idx[j['parent']['@link']]
            p_idxs.append(p_indices)
        return jnp.array(statics), jnp.array(axes), np.array(types), np.array(q_idxs), np.array(p_idxs)

    def _prepare_collision_masks(self, ignore_adjacent=True):
        num_spheres = len(self.sphere_link_indices)
        id_i, id_j = self.sphere_link_indices[:, None], self.sphere_link_indices[None, :]
        mask = (id_i != id_j)
        if ignore_adjacent:
            for name in self.link_order:
                if name == self.base_link: continue
                c_idx = self.link_order.index(name)
                p_idx = self.parent_indices[c_idx]
                if p_idx != -1:
                    mask &= ~((id_i == p_idx) & (id_j == c_idx))
                    mask &= ~((id_i == c_idx) & (id_j == p_idx))
        return mask & jnp.triu(jnp.ones((num_spheres, num_spheres), dtype=bool), k=1)

    def _compute_collision_data(self, q):
        spheres = self.compute_sphere_positions(q)
        diff = spheres[:, None, :] - spheres[None, :, :]
        dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        margin_matrix = dist_matrix - (self.sphere_radii[:, None] + self.sphere_radii[None, :])
        return spheres, margin_matrix

    def compute_self_collision_dist(self, q):
        _, margin_matrix = self._compute_collision_data(q)
        return jnp.min(jnp.where(self.collision_mask, margin_matrix, 1e6))

    def self_collision_cost(self, q, scale=100, min_margin=-0.005):
        _, margin_matrix = self._compute_collision_data(q)
        return jnp.sum(jax.nn.relu(min_margin - margin_matrix) * self.collision_mask) * scale

    def check_collisions(self, q, margin=-0.005):
        spheres, margin_matrix = self._compute_collision_data(q)
        pairs = (margin_matrix < margin) & self.collision_mask
        return spheres, jnp.any(pairs, axis=1) | jnp.any(pairs, axis=0)

    def compute_sphere_positions(self, q):
        num_links = len(self.link_order)
        link_transforms = jnp.tile(jnp.identity(4), (num_links, 1, 1))
        for i in range(num_links):
            p_idx = int(self.parent_indices[i])
            if p_idx == -1: continue
            T_motion = jnp.identity(4)
            j_type, q_idx = int(self.joint_types[i]), int(self.q_indices[i])
            if j_type == 1: T_motion = axis_angle_to_matrix(self.joint_axes[i], q[q_idx])
            elif j_type == 2: T_motion = T_motion.at[0:3, 3].set(self.joint_axes[i] * q[q_idx])
            link_transforms = link_transforms.at[i].set(link_transforms[p_idx] @ self.joint_statics[i] @ T_motion)
        sphere_link_ts = link_transforms[self.sphere_link_indices]
        return jnp.einsum('nij,njk->nik', sphere_link_ts, self.sphere_offsets)[:, 0:3, 3]

    def update(self, q):
        return self._jit_update(q)