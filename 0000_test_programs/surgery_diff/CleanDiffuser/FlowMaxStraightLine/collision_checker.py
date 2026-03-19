_IMPORT_ERROR = None

try:
    import torch
except ModuleNotFoundError as exc:
    torch = None
    _IMPORT_ERROR = exc

try:
    from wrs.robot_sim.robots.xarmlite6_wg.sphere_collision_checker import SphereCollisionChecker
except ModuleNotFoundError as exc:
    SphereCollisionChecker = None
    _IMPORT_ERROR = exc

try:
    import jax
    import jax.numpy as jnp
    import jax2torch
except ModuleNotFoundError as exc:
    jax = None
    jnp = None
    jax2torch = None
    _IMPORT_ERROR = exc


class XarmCollisionChecker:
    def __init__(
        self,
        urdf_path="wrs/robot_sim/robots/xarmlite6_wg/xarm6_sphere_visuals.urdf",
        margin=1.0,
        penetration_threshold=-0.005,
        device="cpu",
    ):
        if _IMPORT_ERROR is not None:
            raise ImportError(
                "XarmCollisionChecker requires runtime dependencies: torch, jax, jax2torch, and wrs."
            ) from _IMPORT_ERROR

        self.urdf_path = urdf_path
        self.margin = margin
        self.penetration_threshold = penetration_threshold
        self.requested_device = device

        self.device = torch.device(device)
        self.dof = 6

        self.build_cc(margin=margin, penetration_threshold=penetration_threshold)

    def build_cc(self, margin=None, penetration_threshold=None, warmup=True):
        if margin is not None:
            self.margin = margin
        if penetration_threshold is not None:
            self.penetration_threshold = penetration_threshold

        self.cc_model = SphereCollisionChecker(self.urdf_path)
        if warmup:
            self.cc_model.update(jnp.zeros(self.dof, dtype=jnp.float32))

        vmap_jax_cost = jax.jit(
            jax.vmap(self.cc_model.self_collision_cost, in_axes=(0, None, None))
        )
        self.torch_collision_vmap = jax2torch.jax2torch(
            lambda q_batch: vmap_jax_cost(q_batch, self.margin, self.penetration_threshold)
        )

    def _ensure_torch_q(self, q):
        if isinstance(q, torch.Tensor):
            q = q.to(device=self.device, dtype=torch.float32)
            if not q.is_contiguous():
                q = q.contiguous()
            # Force owned storage to avoid repeated DLPack alignment fallback copies in jax2torch.
            q = q.clone()
        else:
            q = torch.tensor(q, dtype=torch.float32, device=self.device)
        if q.shape[-1] != self.dof:
            raise ValueError(f"Expected last dim = {self.dof}, got {q.shape[-1]}.")
        return q

    def collision_cost(self, q):
        q = self._ensure_torch_q(q)
        original_shape = q.shape[:-1]
        cost = self.torch_collision_vmap(q.reshape(-1, self.dof))
        if len(original_shape) == 0:
            return cost[0]
        return cost.reshape(original_shape)

    def check_collision(self, q, collision_threshold=0.0):
        is_collision = self.collision_cost(q) > collision_threshold
        if is_collision.ndim == 0:
            return bool(is_collision.item())
        return is_collision

    def trajectory_collision_cost(self, q_traj, reduce="none"):
        q_traj = self._ensure_torch_q(q_traj)
        if q_traj.ndim < 2:
            raise ValueError("Trajectory input must have shape [N, dof] or [B, N, dof].")

        point_cost = self.collision_cost(q_traj)
        if reduce == "none":
            return point_cost
        if reduce == "max":
            return point_cost.max(dim=-1).values
        if reduce == "mean":
            return point_cost.mean(dim=-1)
        if reduce == "sum":
            return point_cost.sum(dim=-1)
        raise ValueError(f"Unsupported reduce mode: {reduce}")

    def trajectory_in_collision(self, q_traj, collision_threshold=0.0):
        return self.trajectory_collision_cost(q_traj, reduce="none") > collision_threshold

    def trajectory_collision_free(self, q_traj, collision_threshold=0.0):
        return ~self.trajectory_in_collision(
            q_traj, collision_threshold=collision_threshold
        ).any(dim=-1)
