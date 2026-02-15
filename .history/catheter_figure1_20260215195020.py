import pyvista as pv
import numpy as np
from scipy.interpolate import make_interp_spline


def create_catheter_model(points, radius=0.02):
    """
    使用样条曲线创建一个柔性导管模型
    """
    # 插值增加点的密度，使曲线更平滑
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, 100)
    spline = make_interp_spline(t, points, k=3)
    smooth_points = spline(t_new)

    # 创建 PyVista 的 PolyData 对象
    poly = pv.PolyData(smooth_points)
    # 将点连接成线
    lines = np.full((len(smooth_points) - 1, 3), 2, dtype=np.int_)
    lines[:, 1] = np.arange(len(smooth_points) - 1)
    lines[:, 2] = np.arange(1, len(smooth_points))
    poly.lines = lines

    # 将线生成管状体
    tube = poly.tube(radius=radius)
    return tube


def create_irregular_obstacle(center, size=0.5, seed=None):
    """
    创建一个不规则形状的障碍物（基于变形的球体）
    """
    if seed is not None:
        np.random.seed(seed)

    # 创建一个基础球体
    sphere = pv.Sphere(radius=size, center=center,
                       theta_resolution=20, phi_resolution=20)

    # 对球体顶点进行随机扰动以产生不规则效果
    noise = (np.random.rand(sphere.n_points, 3) - 0.5) * (size * 0.6)
    sphere.points += noise

    # 使用平滑滤镜让形状看起来更自然
    obstacle = sphere.smooth(n_iter=20)
    return obstacle


def main():
    # 1. 设置场景和绘图器
    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("white")  # 论文通常使用白色背景

    # 2. 定义并创建导管 (Catheter)
    catheter_points = np.array([
        [0, 0, 0],
        [0.2, 0.5, 0.8],
        [0.5, 1.2, 1.5],
        [1.2, 1.8, 2.0],
        [2.0, 2.2, 2.5]
    ])
    cat_radius = 0.04
    catheter = create_catheter_model(catheter_points, radius=cat_radius)

    # 添加导管到场景，设置颜色为黑灰色
    plotter.add_mesh(catheter, color="#333333", smooth_shading=True,
                     label="Robotic Catheter", specular=0.5)

    # 3. 创建多个红色小障碍物 (尺寸约为导管半径的2-3倍)
    obs_size = cat_radius * 2.5

    # 3.1 不规则形状障碍物
    obstacles_data = [
        {"center": [0.6, 0.6, 1.0], "seed": 42},
        {"center": [1.4, 2.0, 1.8], "seed": 123},
        {"center": [0.3, 1.5, 1.2], "seed": 7},
        {"center": [1.8, 1.2, 2.2], "seed": 99},
        {"center": [1.0, 1.0, 1.4], "seed": 55},
        {"center": [0.8, 1.8, 1.6], "seed": 88},
        {"center": [1.5, 0.5, 1.0], "seed": 10},
    ]

    for obs in obstacles_data:
        mesh = create_irregular_obstacle(
            obs["center"], size=obs_size, seed=obs["seed"])
        plotter.add_mesh(mesh, color="red", opacity=0.9,
                         smooth_shading=True, specular=0.3)

    # 3.2 增加其他形状：多个小圆柱体 (代替胶囊体以提高兼容性)
    capsule_positions = [
        {"center": [0.5, 1.8, 1.5], "dir": [0, 1, 1]},
        {"center": [1.2, 0.8, 0.5], "dir": [1, 0, 0]},
        {"center": [2.2, 1.5, 2.0], "dir": [0, 0, 1]},
    ]
    for cap in capsule_positions:
        # 使用 Cylinder 代替 Capsule，因为旧版本 PyVista 的 Capsule 参数可能不一致
        cylinder = pv.Cylinder(
            center=cap["center"], radius=obs_size*0.6, height=obs_size*2.5, direction=cap["dir"])
        plotter.add_mesh(cylinder, color="red",
                         opacity=0.9, smooth_shading=True)

    # 3.3 增加其他形状：更多的小球体
    small_spheres_pos = [
        [1.6, 1.8, 2.4],
        [1.2, 0.8, 1.8],
        [0.4, 0.4, 0.5],
        [2.0, 2.5, 2.6],
        [0.1, 1.0, 0.8]
    ]
    for pos in small_spheres_pos:
        s = pv.Sphere(radius=obs_size*0.7, center=pos)
        plotter.add_mesh(s, color="red", opacity=0.9, smooth_shading=True)

    # 3.4 增加一些小的圆锥体
    cone_positions = [
        [1.0, 2.2, 2.0],
        [0.7, 0.3, 1.2]
    ]
    for pos in cone_positions:
        cone = pv.Cone(center=pos, direction=[
                       0, 0, 1], radius=obs_size*0.8, height=obs_size*1.5)
        plotter.add_mesh(cone, color="red", opacity=0.8, smooth_shading=True)

    # 4. 设置视角和辅助元素
    plotter.add_axes()

    plotter.camera_position = [
        (6.0, 4.0, 5.0),  # 相机位置
        (1.0, 1.2, 1.2),  # 焦点位置
        (0.0, 0.0, 1.0)  # 向上方向
    ]

    print("正在打开可视化窗口...")
    plotter.show()


if __name__ == "__main__":
    main()
