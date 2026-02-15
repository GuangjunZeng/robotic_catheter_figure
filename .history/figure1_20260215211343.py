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
    创建一个带随机扰动的正多边形，避免出现过细的形状
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

    # --- 右侧视口: 尖端截面细节 ---
    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    # 【方案 A 简化实现】：逐段渲染导管尾迹，避免使用标量值和透明度映射
    tail_pts = smooth_pts[-25:]
    n_segments = len(tail_pts) - 1

    # 分段渲染，从末端向尖端，每段的透明度和半径逐渐增加
    for i in range(n_segments):
        segment_pts = tail_pts[i:i+2]
        segment_poly = pv.PolyData(segment_pts)
        segment_poly.lines = np.array([[2, 0, 1]])

        # 半径和透明度从后向前逐渐增加
        progress = (i + 1) / n_segments  # 0 到 1
        # 从 0.2*cat_radius 到 cat_radius
        segment_radius = cat_radius * (0.2 + progress * 0.8)
        opacity = 0.2 + progress * 0.8  # 从 0.2 到 1.0

        # 创建这一段的管
        segment_tube = segment_poly.tube(radius=segment_radius)

        # 计算颜色：从白色到深灰色
        color_val = int(255 * (1 - progress * 0.8))  # 从 255 到 51
        color = f"#{color_val:02x}{color_val:02x}{color_val:02x}"

        # 添加到绘图器
        plotter.add_mesh(segment_tube, color=color,
                         opacity=opacity, smooth_shading=True)

    # 3. 规范化生成的障碍物
    np.random.seed(42)
    for i in range(15):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(cat_radius*1.5, 0.28)
        local_pos = [dist * np.cos(angle), dist * np.sin(angle), 0]

        rand_val = np.random.rand()
        base_r = np.random.uniform(0.015, 0.03)

        if rand_val < 0.33:
            obs_p = pv.Disc(center=[local_pos[0], local_pos[1], tip_pos[2]],
                            inner=0, outer=base_r, normal=[0, 0, 1], c_res=30)
        elif rand_val < 0.66:
            w, h = np.random.uniform(0.01, 0.03, 2)
            obs_p = pv.Box(bounds=[local_pos[0]-w, local_pos[0]+w, local_pos[1] -
                           h, local_pos[1]+h, tip_pos[2]-0.001, tip_pos[2]+0.001])
        else:
            obs_p = create_regular_polygon(
                [local_pos[0], local_pos[1], tip_pos[2]], base_r, nsides=3)

        plotter.add_mesh(obs_p, color="red", opacity=0.8)

    # 4. 探测范围圆
    det_circle = pv.Circle(radius=cat_radius*4, resolution=50)
    det_circle.points += [0, 0, tip_pos[2]]
    plotter.add_mesh(det_circle, color="green",
                     style="wireframe", line_width=1.5)

    # 5. 浅灰色边框
    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 设置右侧相机
    plotter.camera.position = [tip_pos[0], tip_pos[1], tip_pos[2] + 1.0]
    plotter.camera.focal_point = [tip_pos[0], tip_pos[1], tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.show()


if __name__ == "__main__":
    main()
