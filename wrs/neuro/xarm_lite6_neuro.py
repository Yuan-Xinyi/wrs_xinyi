import math
import torch
import wrs.basis.robot_math as rm
import wrs.neuro._kinematics.jlchain as jlc
import wrs.modeling.geometric_model as mgm
import wrs.visualization.panda.world as wd
import wrs.basis.constant as bc


class XArmLite6GPU:
    def __init__(self, device=None):
        """initialize scene and robot model"""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create visualization world
        self.base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
        mgm.gen_frame().attach_to(self.base)

        # initialize robot model
        self.robot = jlc.JLChain(n_dof=6)
        self._build_robot()
        self.robot.finalize()

    def _build_robot(self):
        """define xArm Lite 6 joints and their kinematic parameters"""
        r = self.robot
        device = self.device

        # anchor
        r.jnts[0].loc_pos = torch.tensor([.0, .0, .2433], dtype=torch.float32, device=device)
        r.jnts[0].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[0].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)

        # first joint
        r.jnts[1].loc_pos = torch.tensor([.0, .0, .0], dtype=torch.float32, device=device)
        r.jnts[1].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, -1.5708, 3.1416),
                                            dtype=torch.float32, device=device)
        r.jnts[1].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[1].motion_range = torch.tensor([-2.61799, 2.61799], dtype=torch.float32, device=device)

        # second joint
        r.jnts[2].loc_pos = torch.tensor([.2, .0, .0], dtype=torch.float32, device=device)
        r.jnts[2].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-3.1416, 0., 1.5708),
                                            dtype=torch.float32, device=device)
        r.jnts[2].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[2].motion_range = torch.tensor([-0.061087, 5.235988], dtype=torch.float32, device=device)

        # third joint
        r.jnts[3].loc_pos = torch.tensor([.087, -.2276, .0], dtype=torch.float32, device=device)
        r.jnts[3].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
        r.jnts[3].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[3].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)

        # fourth joint
        r.jnts[4].loc_pos = torch.tensor([.0, .0, .0], dtype=torch.float32, device=device)
        r.jnts[4].loc_rotmat = torch.tensor(rm.rotmat_from_euler(1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
        r.jnts[4].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[4].motion_range = torch.tensor([-2.1642, 2.1642], dtype=torch.float32, device=device)

        # fifth joint
        r.jnts[5].loc_pos = torch.tensor([.0, .0615, .0], dtype=torch.float32, device=device)
        r.jnts[5].loc_rotmat = torch.tensor(rm.rotmat_from_euler(-1.5708, 0., 0.),
                                            dtype=torch.float32, device=device)
        r.jnts[5].loc_motion_ax = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        r.jnts[5].motion_range = torch.tensor([-math.pi, math.pi], dtype=torch.float32, device=device)

        # flange
        r._loc_flange_pos = torch.tensor([0.0, 0.0, 0.0], device=device)

    def ik(self, tgt_pos: torch.Tensor, tgt_rotmat: torch.Tensor):
        """solve inverse kinematics (Analytical IK)"""
        raise NotImplementedError("GPU-based IK not implemented yet.")

    def demo(self):
        """test FK + IK"""
        jnt = self.robot.rand_conf()
        pos, rotmat = self.robot.fk(jnt_values=jnt)
        print("Joint values:", jnt)
        print("FK Position:", pos)
        print("FK Rotmat:\n", rotmat)

        self.base.run()


if __name__ == "__main__":
    xarm = XArmLite6GPU()
    jnt = xarm.robot.rand_conf()
    pos, rotmat = xarm.robot.fk(jnt_values=jnt)
    print("Joint values:", jnt)
    print("FK Position:", pos)
    print("FK Rotmat:\n", rotmat)
    xarm.robot.goto_given_conf(jnt_values=jnt)
    xarm.robot.gen_stickmodel().attach_to(xarm.base)
    xarm.robot.gen_meshmodel(rgb=bc.cyan, alpha=.3).attach_to(xarm.base)
    xarm.base.run()
