import wrs.robot_sim.robots.franka_research_3.franka_research_3 as franka
from wrs import wd, rm, mcm
import wrs.visualization.panda.world as wd
from wrs import wd, rm, mcm
import wrs.modeling.geometric_model as mgm
import numpy as np
from copy import copy

from ruckig import InputParameter, Ruckig, Trajectory, Result, OutputParameter

robot = franka.FrankaResearch3(enable_cc=True)
base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
mgm.gen_frame().attach_to(base)

start_conf = [ 2.16841489, -1.74361228,  0.5306784 , -2.08889768,  2.40161585,
        4.26200876, -0.92424546]
end_conf = [ 1.03215528,  0.75227928, -1.20762504, -1.08915433,  1.43333563,
        2.44452787,  0.52188896]
# start_conf = robot.rand_conf()
# end_conf = robot.rand_conf()
print('start_conf:', repr(start_conf))
print('end_conf:', repr(end_conf))

robot.goto_given_conf(jnt_values=end_conf)
robot.gen_meshmodel(alpha=1.).attach_to(base)
robot.goto_given_conf(jnt_values=start_conf)
robot.gen_meshmodel(alpha=1.).attach_to(base)

'''ruckig'''
jnt_p = []
sampling_interval = 0.01
inp = InputParameter(robot.n_dof)
out = OutputParameter(robot.n_dof)
otg = Ruckig(robot.n_dof, sampling_interval)
inp.current_position, inp.target_position = start_conf, end_conf
inp.current_velocity = rm.np.zeros(robot.n_dof) # support non-zero vel, zero for simplicity
inp.current_acceleration = rm.np.zeros(robot.n_dof) # support non-zero acc, zero for simplicity
inp.target_velocity = rm.np.zeros(robot.n_dof)
inp.target_acceleration = rm.np.zeros(robot.n_dof)
inp.min_position = robot.jnt_ranges[:, 0]
inp.max_position = robot.jnt_ranges[:, 1]

inp.max_velocity = rm.np.asarray([rm.pi * 2 / 3] * robot.n_dof)
inp.max_acceleration = rm.np.asarray([rm.pi] * robot.n_dof)
inp.max_jerk = rm.np.asarray([rm.pi * 2] * robot.n_dof)

# Generate the trajectory within the control loop
first_output, out_list = None, []
res = Result.Working
while res == Result.Working:
    res = otg.update(inp, out)
    # print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
    
    '''append the trajectory to the dataset'''
    assert (np.array(inp.target_position) == end_conf).all(), 'goal_conf not equal to inp.target_position'
    jnt_p.append(np.array(out.new_position).reshape(1, 7))


    '''append the trajectory to the dataset'''
    out_list.append(copy(out))
    out.pass_to_input(inp)

    if not first_output:
        first_output = copy(out)

for i in range(len(jnt_p)-1):
    s_pos, _ = robot.fk(jnt_values=jnt_p[i][0])
    e_pos, _ = robot.fk(jnt_values=jnt_p[i+1][0])
    mgm.gen_stick(spos=s_pos, epos=e_pos, radius=.0025, rgb=[0,0,0]).attach_to(base)
tgt_pos, tgt_rotmat = robot.fk(jnt_values = end_conf)
mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
# tgt_pos, tgt_rotmat = robot.fk(jnt_values = robot.rand_conf())
# mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
base.run()