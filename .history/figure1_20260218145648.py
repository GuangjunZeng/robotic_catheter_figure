import pyvista as pv
import numpy as np
from scipy.interpolate import make_interp_spline


def create_catheter_model(points, radius=0.02):
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

        centerline_x = -0.30 * t_params
        centerline_y = 0.0 - 0.06 * t_params * np.sin(np.pi * t_params)

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
            p0_l, p0_r = i * 2, i * 2 + 1
            p1_l, p1_r = (i + 1) * 2, (i + 1) * 2 + 1
            segment_faces.extend([3, p0_l, p0_r, p1_r])
            segment_faces.extend([3, p0_l, p1_r, p1_l])

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


def create_catheter_cross_section(z_height=0, cat_radius=0.04):
    """
    创建具有三维感的导管截面：
    - 外圆环（管壁）
    - 内腔圆盘（渐变填充）
    - 高光弧（右上方白色弧，模拟圆柱体反光）
    """
    meshes = []

    # 1. 内腔填充：深灰色圆盘（模拟管道内部空腔）
    inner_disc = pv.Disc(center=[0, 0, z_height], inner=0,
                         outer=cat_radius * 0.6, normal=[0, 0, 1], c_res=80)
    meshes.append((inner_disc, "#555555", 1.0))

    # 2. 管壁圆环：深灰色环（模拟管壁厚度）
    wall_ring = pv.Disc(center=[0, 0, z_height], inner=cat_radius * 0.6,
                        outer=cat_radius, normal=[0, 0, 1], c_res=80)
    meshes.append((wall_ring, "#222222", 1.0))

    # 3. 外轮廓线：黑色细圆圈（强化边界）
    outline = pv.Circle(radius=cat_radius, resolution=80)
    outline.points[:, 2] = z_height + 0.001
    meshes.append((outline, "#111111", 1.0))

    # 4. 高光弧：在右上方约 30°-80° 范围内画一段白色弧
    # 这是让大脑识别"圆柱体截面"的关键视觉元素
    highlight_angles = np.linspace(np.pi * 0.15, np.pi * 0.45, 20)
    highlight_r = cat_radius * 0.72  # 高光位于管壁内侧
    highlight_pts = np.zeros((20, 3))
    highlight_pts[:, 0] = highlight_r * np.cos(highlight_angles)
    highlight_pts[:, 1] = highlight_r * np.sin(highlight_angles)
    highlight_pts[:, 2] = z_height + 0.002

    highlight_poly = pv.PolyData(highlight_pts)
    highlight_lines = np.full((19, 3), 2, dtype=np.int_)
    highlight_lines[:, 1] = np.arange(19)
    highlight_lines[:, 2] = np.arange(1, 20)
    highlight_poly.lines = highlight_lines
    highlight_tube = highlight_poly.tube(radius=cat_radius * 0.06)
    meshes.append((highlight_tube, "#FFFFFF", 0.9))

    return meshes


def main():
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.set_background("white")

    catheter_points = np.array([[0, 0, 0], [0.2, 0.5, 0.8], [0.5, 1.2, 1.5],
                                [1.2, 1.8, 2.0], [2.0, 2.2, 2.5]])
    cat_radius = 0.04
    catheter_mesh, smooth_pts = create_catheter_model(
        catheter_points, radius=cat_radius)
    tip_pos = smooth_pts[-1]
    tip_dir = (smooth_pts[-1] - smooth_pts[-10]) / \
        np.linalg.norm(smooth_pts[-1] - smooth_pts[-10])

    obs_size = cat_radius * 1.8
    box_pos = [[0.6, 0.6, 1.0], [1.4, 2.0, 1.8]]
    sphere_pos = [[1.6, 1.8, 2.4], [1.1, 0.6, 1.8]]
    tetra_pos = [[2.2, 1.5, 2.0], [0.8, 1.9, 2.2]]

    # --- 左侧视口: 3D 全景 ---
    plotter.subplot(0, 0)
    plotter.add_text("Global Workspace", font_size=12, color="black")
    plotter.add_mesh(catheter_mesh, color="#333333",
                     smooth_shading=True, specular=0.5)

    def add_elegant_velocity(plotter, center, direction, size=0.15):
        direction = np.array(direction) / np.linalg.norm(direction)
        start_pt = np.array(center) + direction * obs_size * 1.3
        end_pt = start_pt + direction * size
        line = pv.Line(start_pt, end_pt)
        plotter.add_mesh(line, color="hotpink", line_width=2)
        cone = pv.Cone(center=end_pt, direction=direction,
                       height=size*0.4, radius=size*0.12, resolution=20)
        plotter.add_mesh(cone, color="hotpink")

    for i, pos in enumerate(box_pos):
        b = pv.Box(bounds=[pos[0]-obs_size, pos[0]+obs_size, pos[1]-obs_size,
                           pos[1]+obs_size, pos[2]-obs_size, pos[2]+obs_size])
        plotter.add_mesh(b, color="red", style="wireframe",
                         line_width=1, opacity=0.8)
        add_elegant_velocity(
            plotter, pos, [0.2, -0.1, 0.1] if i == 0 else [-0.1, 0.2, -0.1])

    for i, pos in enumerate(sphere_pos):
        s = pv.Sphere(radius=obs_size, center=pos,
                      phi_resolution=10, theta_resolution=10)
        plotter.add_mesh(s, color="red", style="wireframe",
                         line_width=0.8, opacity=0.6)
        add_elegant_velocity(
            plotter, pos, [-0.1, -0.2, 0.1] if i == 0 else [0.2, 0.1, -0.1])

    for i, pos in enumerate(tetra_pos):
        s = obs_size * 1.2
        pts = np.array([[pos[0]+s, pos[1]+s, pos[2]+s], [pos[0]-s, pos[1]-s, pos[2]+s],
                        [pos[0]-s, pos[1]+s, pos[2]-s], [pos[0]+s, pos[1]-s, pos[2]-s]])
        faces = np.array([3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3])
        tetra = pv.PolyData(pts, faces)
        plotter.add_mesh(tetra, color="red", style="wireframe",
                         line_width=1, opacity=0.8)
        add_elegant_velocity(
            plotter, pos, [0.1, 0.2, 0.1] if i == 0 else [-0.2, -0.1, 0.2])

    roi_plane = pv.Plane(center=tip_pos, direction=tip_dir,
                         i_size=0.6, j_size=0.6)
    plotter.add_mesh(roi_plane, color="lightgray",
                     style="wireframe", line_width=1, opacity=0.5)
    plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)
    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    # --- 右侧视口: 尖端截面细节 (2D + 伪三维) ---
    plotter.subplot(0, 1)
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    # 1. 渐变带（底层，最先绘制）
    gradient_segments = create_smooth_2d_gradient_band(
        z_height=tip_pos[2], cat_radius=cat_radius)
    for segment_mesh, color, opacity in gradient_segments:
        plotter.add_mesh(segment_mesh, color=color,
                         opacity=opacity, smooth_shading=True)

    # 2. 障碍物（中层）
    np.random.seed(42)
    count = 0
    attempts = 0
    while count < 35 and attempts < 200:
        attempts += 1
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(0.07, 0.28)
        ox, oy = dist * np.cos(angle), dist * np.sin(angle)
        base_r = np.random.uniform(0.008, 0.016)

        # 渐变带重叠检查
        is_overlapping_band = False
        if ox < 0.05:
            t = ox / -0.30
            if 0 <= t <= 1.1:
                band_y = 0.0 - 0.06 * t * np.sin(np.pi * t)
                band_w = cat_radius * (1 - t ** 1.5)
                if abs(oy - band_y) < (band_w + base_r + 0.01):
                    is_overlapping_band = True
        if is_overlapping_band:
            continue

        local_pos = [ox, oy, tip_pos[2]]
        rand_val = np.random.rand()
        if rand_val < 0.33:
            obs_p = pv.Circle(radius=base_r, resolution=50)
            obs_p.points[:, 0] += ox
            obs_p.points[:, 1] += oy
            obs_p.points[:, 2] = tip_pos[2]
        elif rand_val < 0.66:
            ratio = np.random.uniform(0.7, 1.3)
            w, h = base_r, base_r * ratio
            obs_p = pv.Box(bounds=[ox-w, ox+w, oy-h, oy+h,
                           tip_pos[2]-0.001, tip_pos[2]+0.001])
        else:
            obs_p = create_regular_polygon(local_pos, base_r, nsides=3)

        plotter.add_mesh(obs_p, color="red", style="wireframe",
                         line_width=1, opacity=0.8)
        count += 1

    # 3. 探测范围圆（中层）
    det_circle = pv.Circle(radius=cat_radius*4, resolution=80)
    det_circle.points[:, 2] = tip_pos[2]
    plotter.add_mesh(det_circle, color="green",
                     style="wireframe", line_width=1.5)

    # 4. 浅灰色边框
    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 5. 导管截面（顶层，最后绘制，确保覆盖所有元素）
    cross_section_meshes = create_catheter_cross_section(
        z_height=tip_pos[2] + 0.003, cat_radius=cat_radius)
    for mesh, color, opacity in cross_section_meshes:
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         smooth_shading=True)

    # 右侧相机：正对 XY 平面
    plotter.camera.position = [0, 0, tip_pos[2] + 1.0]
    plotter.camera.focal_point = [0, 0, tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.show()


if __name__ == "__main__":
    main()
