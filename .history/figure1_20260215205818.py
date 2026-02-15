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

    # 【修改位置 1】左侧 3D 障碍物的基础尺寸
    obs_size = cat_radius * 2.5

    # 【修改位置 2】左侧 3D 障碍物的位置定义
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
    plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)

    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    # --- 右侧视口: 尖端截面细节 ---
    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    # 1. 导管截面 (c_res=50 确保它是圆的)
    tip_circle = pv.Disc(center=[0, 0, 0], inner=0,
                         outer=cat_radius, normal=[0, 0, 1], c_res=50)
    plotter.add_mesh(tip_circle, color="#333333")

    # 2. 探测范围 (resolution=50 确保它是圆的)
    # 修复报错：旧版本可能不支持 n_points，改用 resolution
    detection_range = pv.Circle(radius=cat_radius*4, resolution=50)
    plotter.add_mesh(detection_range, color="green",
                     style="wireframe", line_width=1.5)

    # 【修改位置 3】右侧截面图障碍物的大小、位置和类型
    np.random.seed(42)
    for i in range(15):  # 增加数量到 15 个
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(cat_radius*1.5, 0.28)
        pos = [dist * np.cos(angle), dist * np.sin(angle), 0]

        rand_val = np.random.rand()
        if rand_val < 0.33:
            # 类型1: 圆形 (更小)
            obs_p = pv.Disc(center=pos, inner=0, outer=np.random.uniform(
                0.01, 0.025), normal=[0, 0, 1], c_res=30)
        elif rand_val < 0.66:
            # 类型2: 矩形 (更小)
            w, h = np.random.uniform(0.01, 0.03, 2)
            obs_p = pv.Box(bounds=[pos[0]-w, pos[0]+w,
                           pos[1]-h, pos[1]+h, -0.001, 0.001])
        else:
            # 类型3: 不等边三角形 (使用 PolyData 手动创建)
            # 随机三个顶点
            offsets = (np.random.rand(3, 2) - 0.5) * 0.06
            tri_pts = np.zeros((3, 3))
            tri_pts[:, :2] = np.array(pos[:2]) + offsets
            # 创建面
            faces = np.array([3, 0, 1, 2])
            obs_p = pv.PolyData(tri_pts, faces)

        plotter.add_mesh(obs_p, color="red", opacity=0.8)

    # 4. 标注浅灰色边框 (对应左图的 ROI)
    roi_border = pv.Box(bounds=[-0.3, 0.3, -0.3, 0.3, -0.001, 0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 设置右侧视口为正交俯视图
    plotter.view_xy()
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    # --- 打印提示并显示 ---
    print("正在打开双视口可视化窗口...")
    plotter.show()


if __name__ == "__main__":
    main()
