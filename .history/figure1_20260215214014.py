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


def create_regular_polygon(center, radius, nsides=3):
    """
    创建一个带随机扰动的正多边形
    """
    angles = np.linspace(0, 2*np.pi, nsides, endpoint=False)
    angles += np.random.uniform(0, 2*np.pi)
    r = radius * (1 + (np.random.rand(nsides) - 0.5) * 0.4)
    pts = np.zeros((nsides, 3))
    pts[:, 0] = center[0] + r * np.cos(angles)
    pts[:, 1] = center[1] + r * np.sin(angles)
    pts[:, 2] = center[2]

    faces = np.array([nsides] + list(range(nsides)))
    return pv.PolyData(pts, faces)


def create_smooth_gradient_ribbon(z_height=0, cat_radius=0.04):
    """
    创建一条平滑、连续的渐变弯曲带状体，从尖端向后延伸
    用来表现柔性导管从 3D 空间穿过截面的效果
    """
    # 参数化中心线：从 (0, 0) 向上方延伸，带有轻微的正弦波曲线
    t_centerline = np.linspace(0, 1, 100)
    # 中心线：从 y=0 逐渐向上（z 方向在 3D 中），x 方向有轻微波动
    centerline_x = 0.08 * np.sin(t_centerline * np.pi)  # 轻微左右摇摆
    centerline_y = 0.15 * t_centerline  # 逐渐向上延伸
    centerline_z = np.full_like(t_centerline, z_height)

    centerline = np.column_stack([centerline_x, centerline_y, centerline_z])

    # 为每个中心线点生成带的两个边界点
    n_points = len(centerline)
    ribbon_points = []

    for i, center_pt in enumerate(centerline):
        # 进度：从 0（尖端）到 1（后方）
        progress = i / (n_points - 1)

        # 宽度：从 cat_radius 逐渐增大
        width = cat_radius * (1 + progress * 2.5)

        # 两个边界点（沿 x 方向）
        left_pt = center_pt + np.array([-width, 0, 0])
        right_pt = center_pt + np.array([width, 0, 0])

        ribbon_points.append(left_pt)
        ribbon_points.append(right_pt)

    ribbon_points = np.array(ribbon_points)

    # 构建四边形面（Quad）
    n_segments = n_points - 1
    faces = []
    for i in range(n_segments):
        # 当前和下一个中心线点对应的四个顶点
        p0_left = i * 2
        p0_right = i * 2 + 1
        p1_left = (i + 1) * 2
        p1_right = (i + 1) * 2 + 1

        # 创建四边形：(p0_left, p0_right, p1_right, p1_left)
        faces.append(4)
        faces.append(p0_left)
        faces.append(p0_right)
        faces.append(p1_right)
        faces.append(p1_left)

    faces = np.array(faces)

    # 创建 PolyData
    ribbon_mesh = pv.PolyData(ribbon_points, faces)

    # 为每个顶点添加标量值，用于颜色和透明度映射
    # 标量值从 1（尖端，最深）到 0（后方，最浅）
    vertex_scalars = []
    for i in range(len(ribbon_points)):
        center_idx = i // 2
        progress = center_idx / (n_points - 1)
        scalar = 1 - progress  # 从 1 到 0
        vertex_scalars.append(scalar)

    vertex_scalars = np.array(vertex_scalars)
    ribbon_mesh.point_data["intensity"] = vertex_scalars

    return ribbon_mesh


def main():
    # 1. 创建双视口绘图器
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
    tip_pos = smooth_pts[-1]
    tip_dir = (smooth_pts[-1] - smooth_pts[-10]) / \
        np.linalg.norm(smooth_pts[-1] - smooth_pts[-10])

    obs_size = cat_radius * 2.5
    box_pos = [[0.6, 0.6, 1.0], [1.4, 2.0, 1.8]]
    sphere_pos = [[1.6, 1.8, 2.4], [1.2, 0.8, 1.8]]
    cyl_pos = [{"center": [2.2, 1.5, 2.0], "dir": [0, 0, 1]}]

    # --- 左侧视口: 3D 全景 ---
    plotter.subplot(0, 0)
    plotter.add_text("Global Workspace", font_size=12, color="black")
    plotter.add_mesh(catheter_mesh, color="#333333",
                     smooth_shading=True, specular=0.5)

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

    roi_plane = pv.Plane(center=tip_pos, direction=tip_dir,
                         i_size=0.6, j_size=0.6)
    plotter.add_mesh(roi_plane, color="lightgray",
                     style="wireframe", line_width=1, opacity=0.5)
    plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)
    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    # --- 右侧视口: 尖端截面细节 (完全 2D 视图) ---
    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    # 【方案 A 平滑连续版本】：绘制渐变的弯曲带状体
    ribbon_mesh = create_smooth_gradient_ribbon(
        z_height=tip_pos[2], cat_radius=cat_radius)

    # 渲染带状体：不使用透明度映射（避免报错），而是为不同部分绘制不同的颜色/透明度
    # 我们分段渲染，每一小段都有不同的颜色

    # 简单方案：直接用标量值来着色，从深灰到浅灰到白色
    plotter.add_mesh(ribbon_mesh, color="gray",
                     show_scalar_bar=False, smooth_shading=True)

    # 中心圆点：尖端
    tip_circle = pv.Disc(center=[0, 0, tip_pos[2]], inner=0,
                         outer=cat_radius, normal=[0, 0, 1], c_res=50)
    plotter.add_mesh(tip_circle, color="#333333", opacity=0.95)

    # 3. 规范化生成的障碍物
    np.random.seed(42)
    for i in range(15):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(cat_radius*1.5, 0.28)
        local_pos = [dist * np.cos(angle), dist * np.sin(angle), tip_pos[2]]

        rand_val = np.random.rand()
        base_r = np.random.uniform(0.015, 0.03)

        if rand_val < 0.33:
            obs_p = pv.Disc(center=local_pos, inner=0,
                            outer=base_r, normal=[0, 0, 1], c_res=30)
        elif rand_val < 0.66:
            w, h = np.random.uniform(0.01, 0.03, 2)
            obs_p = pv.Box(bounds=[local_pos[0]-w, local_pos[0]+w, local_pos[1] -
                           h, local_pos[1]+h, tip_pos[2]-0.001, tip_pos[2]+0.001])
        else:
            obs_p = create_regular_polygon(local_pos, base_r, nsides=3)

        plotter.add_mesh(obs_p, color="red", opacity=0.8)

    # 4. 探测范围圆
    det_circle = pv.Circle(radius=cat_radius*4, resolution=50)
    det_circle.points[:, 2] = tip_pos[2]
    plotter.add_mesh(det_circle, color="green",
                     style="wireframe", line_width=1.5)

    # 5. 浅灰色边框
    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 设置右侧相机：正对 XY 平面
    plotter.camera.position = [0, 0, tip_pos[2] + 1.0]
    plotter.camera.focal_point = [0, 0, tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.show()


if __name__ == "__main__":
    main()
