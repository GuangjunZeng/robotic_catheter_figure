import pyvista as pv
import numpy as np
from scipy.interpolate import make_interp_spline


def create_catheter_model(points, radius=0.02):
    """
    使用样条曲线创建一个柔性导管模型
    """
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, 100)
    spline = make_interp_spline(t, points, k=3)
    smooth_points = spline(t_new)

    poly = pv.PolyData(smooth_points)
    lines = np.full((len(smooth_points) - 1, 3), 2, dtype=np.int_)
    lines[:, 1] = np.arange(len(smooth_points) - 1)
    lines[:, 2] = np.arange(1, len(smooth_points))
    poly.lines = lines

    tube = poly.tube(radius=radius)
    return tube, smooth_points


def main():
    # 1. 创建双视口绘图器 (1行2列)
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background("white")

    # --- 数据准备 ---
    catheter_points = np.array([
        [0, 0, 0],
        [0.2, 0.5, 0.8],
        [0.5, 1.2, 1.5],
        [1.2, 1.8, 2.0],
        [2.0, 2.2, 2.5]
    ])
    cat_radius = 0.04
    catheter_mesh, smooth_pts = create_catheter_model(
        catheter_points, radius=cat_radius)
    tip_pos = smooth_pts[-1]  # 尖端位置
    # 计算尖端切向向量作为截面法向量
    tip_dir = smooth_pts[-1] - smooth_pts[-10]
    tip_dir = tip_dir / np.linalg.norm(tip_dir)

    obs_size = cat_radius * 2.5

    # 定义障碍物数据 (位置)
    box_pos = [[0.6, 0.6, 1.0], [1.4, 2.0, 1.8]]
    sphere_pos = [[1.6, 1.8, 2.4], [1.2, 0.8, 1.8]]
    cyl_pos = [{"center": [2.2, 1.5, 2.0], "dir": [0, 0, 1]}]

    # --- 左侧视口: 3D 全景 ---
    plotter.subplot(0, 0)
    plotter.add_text("Global Workspace", font_size=12, color="black")

    # 添加导管
    plotter.add_mesh(catheter_mesh, color="#333333",
                     smooth_shading=True, specular=0.5)

    # 添加障碍物
    for pos in box_pos:
        b = pv.Box(bounds=[pos[0]-obs_size, pos[0]+obs_size, pos[1] -
                   obs_size, pos[1]+obs_size, pos[2]-obs_size, pos[2]+obs_size])
        plotter.add_mesh(b, color="red", opacity=0.8)
    for pos in sphere_pos:
        s = pv.Sphere(radius=obs_size, center=pos)
        plotter.add_mesh(s, color="red", opacity=0.8)
    for c in cyl_pos:
        cy = pv.Cylinder(center=c["center"], direction=c["dir"],
                         radius=obs_size*0.7, height=obs_size*3)
        plotter.add_mesh(cy, color="red", opacity=0.8)

    # 在尖端添加浅灰色虚线矩形框 (ROI)，表示截面平面
    roi_plane = pv.Plane(center=tip_pos, direction=tip_dir,
                         i_size=0.6, j_size=0.6)
    plotter.add_mesh(roi_plane, color="lightgray",
                     style="wireframe", line_width=1, opacity=0.5)

    # 添加一个极浅的填充平面，增强“截面”的视觉感
    plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)

    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    # --- 右侧视口: 尖端截面细节 ---
    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    # 模拟截面视图：
    # 1. 导管截面 (中心圆)
    tip_circle = pv.Disc(center=[0, 0, 0], inner=0,
                         outer=cat_radius, normal=[0, 0, 1])
    plotter.add_mesh(tip_circle, color="#333333")

    # 2. 安全边界 (虚线圆)
    safety_margin = pv.Circle(radius=cat_radius*3)
    plotter.add_mesh(safety_margin, color="blue",
                     style="wireframe", line_width=1)

    # 3. 模拟周围障碍物的截面投影
    obs_proj1 = pv.Disc(center=[0.15, 0.15, 0],
                        inner=0, outer=obs_size*0.8, normal=[0, 0, 1])
    obs_proj2 = pv.Box(bounds=[-0.25, -0.15, 0.05, 0.25, -0.01, 0.01])
    obs_proj3 = pv.Disc(center=[-0.1, -0.2, 0],
                        inner=0, outer=obs_size*0.6, normal=[0, 0, 1])

    plotter.add_mesh(obs_proj1, color="red", opacity=0.8)
    plotter.add_mesh(obs_proj2, color="red", opacity=0.8)
    plotter.add_mesh(obs_proj3, color="red", opacity=0.8)

    # 4. 标注浅灰色边框 (对应左图的 ROI)
    roi_border = pv.Box(bounds=[-0.3, 0.3, -0.3, 0.3, -0.001, 0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 5. 添加距离标注线
    line = pv.Line([0, 0, 0], [0.15, 0.15, 0])
    plotter.add_mesh(line, color="black", line_width=2)
    plotter.add_text("d_min", position=[
                     0.6, 0.6], color="black", viewport=True, font_size=10)

    # 设置右侧视口为正交俯视图
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    # --- 打印提示并显示 ---
    print("正在打开双视口可视化窗口...")
    plotter.show()


if __name__ == "__main__":
    main()
