import pyvista as pv
import numpy as np
from scipy.interpolate import make_interp_spline

# python catheter_figure1.py


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
    # 定义几个控制点，模拟弯曲的导管
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

    # 3. 创建多个红色障碍物 (尺寸约为导管半径的2-3倍)
    obs_size = cat_radius * 2.5

    # 3.1 不规则形状障碍物
    obstacles_data = [
        {"center": [0.6, 0.6, 1.0], "seed": 42},
        {"center": [1.4, 2.0, 1.8], "seed": 123},
        {"center": [0.3, 1.5, 1.2], "seed": 7},
        {"center": [1.8, 1.2, 2.2], "seed": 99},
        {"center": [1.0, 1.0, 1.4], "seed": 55},
    ]

    for obs in obstacles_data:
        mesh = create_irregular_obstacle(
            obs["center"], size=obs_size, seed=obs["seed"])
        plotter.add_mesh(mesh, color="red", opacity=0.9,
                         smooth_shading=True, specular=0.3)

    # 3.2 增加其他形状：长方体 (模拟窄道或墙壁)
    box = pv.Box(bounds=[0.8, 1.2, 1.6, 1.7, 1.4, 2.2])
    plotter.add_mesh(box, color="red", opacity=0.7, smooth_shading=True)

    # 3.3 增加其他形状：小球体
    small_sphere = pv.Sphere(radius=obs_size*0.8, center=[1.6, 1.8, 2.4])
    plotter.add_mesh(small_sphere, color="red",
                     opacity=0.9, smooth_shading=True)

    # 3.4 增加一个圆柱体
    cylinder = pv.Cylinder(center=[0.5, 0.8, 0.5], direction=[
                           1, 1, 0], radius=obs_size, height=0.4)
    plotter.add_mesh(cylinder, color="red", opacity=0.8, smooth_shading=True)

    # 4. 设置视角和辅助元素
    # 添加坐标轴
    plotter.add_axes()

    # 设置一个好的初始视角
    plotter.camera_position = [
        (6.0, 4.0, 5.0),  # 相机位置
        (1.0, 1.2, 1.2),  # 焦点位置
        (0.0, 0.0, 1.0)  # 向上方向
    ]

    print("正在打开可视化窗口... 您可以在窗口中调整视角。")
    print("提示：在窗口中按 'q' 退出，按 's' 可以保存截图。")

    # 5. 显示
    plotter.show()


if __name__ == "__main__":
    main()
