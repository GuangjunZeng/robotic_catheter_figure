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


def create_smooth_2d_gradient_band(z_height=0, cat_radius=0.04):
    """
    在纯 2D 平面上创建平滑的连续渐变带
    从黑色圆的边界出发，向左延伸，逐渐变窄、变浅、变透明
    """
    n_segments = 60
    segment_list = []

    for seg_idx in range(n_segments):
        t_start = seg_idx / n_segments
        t_end = (seg_idx + 1) / n_segments
        t_params = np.linspace(t_start, t_end, 15)

        # 中心线：起点在 (0, 0)，向左延伸 (负 X 方向)
        centerline_x = -0.30 * t_params
        centerline_y = 0.0 - 0.06 * t_params * np.sin(np.pi * t_params)
        centerline_z = np.full_like(t_params, z_height)

        # 宽度：从 cat_radius 缓慢变窄
        widths = cat_radius * (1 - t_params ** 1.5)

        segment_points = []
        for i in range(len(t_params)):
            segment_points.append(
                [centerline_x[i], centerline_y[i] - widths[i], z_height])
            segment_points.append(
                [centerline_x[i], centerline_y[i] + widths[i], z_height])

        segment_points = np.array(segment_points)
        segment_faces = []
        for i in range(len(t_params) - 1):
            p0_left, p0_right = i * 2, i * 2 + 1
            p1_left, p1_right = (i + 1) * 2, (i + 1) * 2 + 1
            segment_faces.extend([3, p0_left, p0_right, p1_right])
            segment_faces.extend([3, p0_left, p1_right, p1_left])

        segment_mesh = pv.PolyData(segment_points, np.array(segment_faces))

        progress = (t_start + t_end) / 2
        if progress < 0.25:
            r_val = int(51 + (102 - 51) * (progress / 0.25))
        elif progress < 0.5:
            r_val = int(102 + (153 - 102) * ((progress - 0.25) / 0.25))
        elif progress < 0.75:
            r_val = int(153 + (204 - 153) * ((progress - 0.5) / 0.25))
        else:
            r_val = int(204 + (238 - 204) * ((progress - 0.75) / 0.25))

        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"
        opacity = 0.95 * (1 - progress ** 0.8)
        segment_list.append((segment_mesh, color, opacity))

    return segment_list


def main():
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background("white")

    catheter_points = np.array([[0, 0, 0], [0.2, 0.5, 0.8], [0.5, 1.2, 1.5], [
                               1.2, 1.8, 2.0], [2.0, 2.2, 2.5]])
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

    plotter.subplot(0, 0)
    plotter.add_text("Global Workspace", font_size=12, color="black")
    plotter.add_mesh(catheter_mesh, color="#333333",
                     smooth_shading=True, specular=0.5)
    for i, pos in enumerate(box_pos):
        b = pv.Box(bounds=[pos[0]-obs_size, pos[0]+obs_size, pos[1] -
                   obs_size, pos[1]+obs_size, pos[2]-obs_size, pos[2]+obs_size])
        plotter.add_mesh(b, color="red", opacity=0.8)
        # 增加运动矢量
        direction = [0.2, -0.1, 0.1] if i == 0 else [-0.1, 0.2, -0.1]
        arrow = pv.Arrow(start=pos, direction=direction, scale=0.5)
        plotter.add_mesh(arrow, color="orange")

    for i, pos in enumerate(sphere_pos):
        s = pv.Sphere(radius=obs_size, center=pos)
        plotter.add_mesh(s, color="red", opacity=0.8)
        # 增加运动矢量
        direction = [-0.1, -0.2, 0.1] if i == 0 else [0.2, 0.1, -0.1]
        arrow = pv.Arrow(start=pos, direction=direction, scale=0.5)
        plotter.add_mesh(arrow, color="orange")

    for c in cyl_pos:
        cy = pv.Cylinder(center=c["center"], direction=c["dir"],
                         radius=obs_size*0.7, height=obs_size*3)
        plotter.add_mesh(cy, color="red", opacity=0.8)
        # 增加运动矢量
        arrow = pv.Arrow(start=c["center"], direction=[
                         0.1, 0.2, 0.1], scale=0.5)
        plotter.add_mesh(arrow, color="orange")

    roi_plane = pv.Plane(center=tip_pos, direction=tip_dir,
                         i_size=0.6, j_size=0.6)
    plotter.add_mesh(roi_plane, color="lightgray",
                     style="wireframe", line_width=1, opacity=0.5)
    plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)
    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    gradient_segments = create_smooth_2d_gradient_band(
        z_height=tip_pos[2], cat_radius=cat_radius)
    for segment_mesh, color, opacity in gradient_segments:
        plotter.add_mesh(segment_mesh, color=color,
                         opacity=opacity, smooth_shading=True)

    tip_circle = pv.Disc(center=[0, 0, tip_pos[2]], inner=0,
                         outer=cat_radius, normal=[0, 0, 1], c_res=50)
    plotter.add_mesh(tip_circle, color="#333333", opacity=1.0)

    # 【修改位置 3】右侧截面图障碍物，增加防重叠逻辑（包括渐变带）
    np.random.seed(42)
    count = 0
    attempts = 0
    while count < 35 and attempts < 200:  # 增加到 35 个，增加尝试次数
        attempts += 1
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(0.07, 0.28)
        ox = dist * np.cos(angle)
        oy = dist * np.sin(angle)

        base_r = np.random.uniform(0.008, 0.016)  # 进一步减小尺寸

        # --- 渐变带重叠检查 ---
        is_overlapping_band = False
        if ox < 0.05:  # 渐变带在左侧 (x < 0)
            # 计算渐变带在该 x 处的 y 中心和宽度
            t = ox / -0.30
            if 0 <= t <= 1.1:  # 稍微放宽范围
                band_y = 0.0 - 0.06 * t * np.sin(np.pi * t)
                band_w = cat_radius * (1 - t ** 1.5)
                # 如果障碍物中心到带中心的 y 距离小于 (带半宽 + 障碍物半径 + 缓冲)
                if abs(oy - band_y) < (band_w + base_r + 0.01):
                    is_overlapping_band = True

        if is_overlapping_band:
            continue

        # --- 生成障碍物 ---
        local_pos = [ox, oy, tip_pos[2]]
        rand_val = np.random.rand()
        if rand_val < 0.33:
            obs_p = pv.Disc(center=local_pos, inner=0,
                            outer=base_r, normal=[0, 0, 1], c_res=30)
        elif rand_val < 0.66:
            ratio = np.random.uniform(0.7, 1.3)
            w, h = base_r, base_r * ratio
            obs_p = pv.Box(bounds=[ox-w, ox+w, oy-h, oy+h,
                           tip_pos[2]-0.001, tip_pos[2]+0.001])
        else:
            obs_p = create_regular_polygon(local_pos, base_r, nsides=3)

        plotter.add_mesh(obs_p, color="red", opacity=0.8)
        count += 1

    det_circle = pv.Circle(radius=cat_radius*4, resolution=50)
    det_circle.points[:, 2] = tip_pos[2]
    plotter.add_mesh(det_circle, color="green",
                     style="wireframe", line_width=1.5)

    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    plotter.camera.position = [0, 0, tip_pos[2] + 1.0]
    plotter.camera.focal_point = [0, 0, tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.show()


if __name__ == "__main__":
    main()
