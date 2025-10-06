import torch
import diffvg
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. 创建目标图像（我们用一个红色圆形作为目标）
canvas_size = 256
target_svg = diffvg.Circle(p=torch.tensor([128.0, 128.0]), r=torch.tensor(60.0))
shapes = [target_svg]
shape_groups = [diffvg.ShapeGroup(shape_ids=torch.tensor([0]), fill_color=torch.tensor([1.0, 0.0, 0.0, 1.0]))]

# 渲染目标图像
scene_args = diffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, shapes, shape_groups)
render = diffvg.RenderFunction.apply
target_img = render(canvas_size, canvas_size, 2, 2, 0, None, *scene_args)
target_img = target_img[:, :, :3]

plt.imshow(target_img.cpu())
plt.title("Target Image (Red Circle)")
plt.axis("off")
plt.show()

# 2. 随机初始化一条贝塞尔曲线（我们的“笔触”）
points = torch.tensor([[50.0, 200.0], [150.0, 50.0], [200.0, 200.0]], requires_grad=True)
path = diffvg.Path(num_control_points=torch.tensor([2]), points=points, is_closed=False)
shapes = [path]
color = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)  # 红色
shape_groups = [diffvg.ShapeGroup(shape_ids=torch.tensor([0]), stroke_color=color, stroke_width=torch.tensor(5.0))]

# 优化器
optimizer = torch.optim.Adam([points, color], lr=0.05)

# 3. 迭代优化
num_iters = 200
for t in range(num_iters):
    optimizer.zero_grad()
    scene_args = diffvg.RenderFunction.serialize_scene(canvas_size, canvas_size, shapes, shape_groups)
    img = render(canvas_size, canvas_size, 2, 2, t, None, *scene_args)
    img = img[:, :, :3]

    # 损失函数：和目标图像的像素差
    loss = F.mse_loss(img, target_img)
    loss.backward()
    optimizer.step()

    if t % 20 == 0:
        print(f"iter {t}, loss = {loss.item():.4f}")

# 4. 显示优化结果
final_img = img.detach().cpu()
plt.imshow(final_img)
plt.title("Optimized Drawing")
plt.axis("off")
plt.show()
