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
    渐变带：起始颜色与圆盘深色衔接，分段更细化
    """
    n_segments = 100
    segment_list = []

    for seg_idx in range(n_segments):
        t_start = seg_idx / n_segments
        t_end = (seg_idx + 1) / n_segments
        t_params = np.linspace(t_start, t_end, 10)

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

        # 修复 1：起始颜色从 #222222 (接近圆盘深色) 开始，过渡更自然
        # 使用更细化的多段颜色映射
        if progress < 0.1:
            # 前 10%：从 #222222 → #444444（非常缓慢）
            ratio = progress / 0.1
            r_val = int(34 + (68 - 34) * ratio)
        elif progress < 0.3:
            # 10%-30%：#444444 → #777777
            ratio = (progress - 0.1) / 0.2
            r_val = int(68 + (119 - 68) * ratio)
        elif progress < 0.55:
            # 30%-55%：#777777 → #AAAAAA
            ratio = (progress - 0.3) / 0.25
            r_val = int(119 + (170 - 119) * ratio)
        elif progress < 0.78:
            # 55%-78%：#AAAAAA → #CCCCCC
            ratio = (progress - 0.55) / 0.23
            r_val = int(170 + (204 - 170) * ratio)
        else:
            # 78%-100%：#CCCCCC → #EEEEEE
            ratio = (progress - 0.78) / 0.22
            r_val = int(204 + (238 - 204) * ratio)

        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"

        # 透明度：前期下降更缓慢，后期加速消失
        opacity = 0.92 * (1 - progress ** 0.7)
        segment_list.append((segment_mesh, color, opacity))

    return segment_list


def create_perspective_ellipse(z_height=0, cat_radius=0.04):
    """
    修复 3：在圆盘后方（左侧）画椭圆形渐变区域
    模拟管道从远处延伸过来时，透视压缩产生的椭圆形截面轮廓
    颜色与渐变带最浅处一致（#EEEEEE），向外逐渐消失
    """
    segment_list = []
    n_layers = 20  # 椭圆层数，越多越平滑

    for layer_idx in range(n_layers):
        progress = layer_idx / n_layers  # 0 到 1，从内到外

        # 椭圆参数：X 轴（透视方向）比 Y 轴更窄，模拟透视压缩
        # 随着层数增加，椭圆逐渐扩大
        a = cat_radius * (0.3 + progress * 1.2)  # X 半轴（透视方向）
        b = cat_radius * (0.8 + progress * 0.5)  # Y 半轴

        # 椭圆中心向左偏移（朝着渐变带方向）
        center_x = -cat_radius * 0.5

        # 生成椭圆点
        n_pts = 60
        angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False)
        pts = np.zeros((n_pts, 3))
        pts[:, 0] = center_x + a * np.cos(angles)
        pts[:, 1] = b * np.sin(angles)
        pts[:, 2] = z_height

        # 构建闭合环形面（与内层椭圆之间的区域）
        if layer_idx == 0:
            # 最内层：直接用圆盘
            inner_a = 0
            inner_b = 0
            inner_cx = center_x
        else:
            prev_progress = (layer_idx - 1) / n_layers
            inner_a = cat_radius * (0.3 + prev_progress * 1.2)
            inner_b = cat_radius * (0.8 + prev_progress * 0.5)
            inner_cx = center_x

        inner_pts = np.zeros((n_pts, 3))
        inner_pts[:, 0] = inner_cx + inner_a * np.cos(angles)
        inner_pts[:, 1] = inner_b * np.sin(angles)
        inner_pts[:, 2] = z_height

        # 合并内外两层点
        all_pts = np.vstack([inner_pts, pts])
        faces = []
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            faces.extend([3, i, next_i, n_pts + next_i])
            faces.extend([3, i, n_pts + next_i, n_pts + i])

        ring_mesh = pv.PolyData(all_pts, np.array(faces))

        # 颜色：与渐变带最浅处一致，从内到外逐渐变浅
        r_val = int(200 + progress * 38)  # #C8C8C8 → #EEEEEE
        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"
        # 透明度：从内到外逐渐消失
        opacity = 0.35 * (1 - progress ** 0.6)
        segment_list.append((ring_mesh, color, opacity))

    return segment_list


def create_catheter_cross_section(z_height=0, cat_radius=0.04):
    """
    创建具有三维感的导管截面：
    - 外圆环（管壁）
    - 内腔圆盘（渐变填充）
    - 高光弧（优化：改为渐变的弧形，而不是突兀的白色线条）
    """
    meshes = []
    inner_disc = pv.Disc(center=[0, 0, z_height], inner=0,
                         outer=cat_radius * 0.6, normal=[0, 0, 1], c_res=80)
    meshes.append((inner_disc, "#555555", 1.0))

    wall_ring = pv.Disc(center=[0, 0, z_height], inner=cat_radius * 0.6,
                        outer=cat_radius, normal=[0, 0, 1], c_res=80)
    meshes.append((wall_ring, "#222222", 1.0))

    outline = pv.Circle(radius=cat_radius, resolution=80)
    outline.points[:, 2] = z_height + 0.001
    meshes.append((outline, "#111111", 1.0))

    n_highlight_layers = 5
    for hl_idx in range(n_highlight_layers):
        angle_center = np.pi * 0.30
        angle_half_width = np.pi * 0.18 * (1 - hl_idx * 0.15)
        highlight_angles = np.linspace(angle_center - angle_half_width,
                                       angle_center + angle_half_width, 20)
        highlight_r = cat_radius * (0.68 + hl_idx * 0.03)
        highlight_pts = np.zeros((20, 3))
        highlight_pts[:, 0] = highlight_r * np.cos(highlight_angles)
        highlight_pts[:, 1] = highlight_r * np.sin(highlight_angles)
        highlight_pts[:, 2] = z_height + 0.002

        highlight_poly = pv.PolyData(highlight_pts)
        highlight_lines = np.full((19, 3), 2, dtype=np.int_)
        highlight_lines[:, 1] = np.arange(19)
        highlight_lines[:, 2] = np.arange(1, 20)
        highlight_poly.lines = highlight_lines

        tube_radius = cat_radius * (0.07 - hl_idx * 0.01)
        hl_opacity = 0.85 - hl_idx * 0.15
        c_val = int(255 - hl_idx * 20)
        hl_color = f"#{c_val:02x}{c_val:02x}{c_val:02x}"
        highlight_tube = highlight_poly.tube(radius=max(tube_radius, 0.001))
        meshes.append((highlight_tube, hl_color, hl_opacity))

    return meshes


def main():
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # --- 左侧视口: 3D 全景 ---
    plotter.subplot(0, 0)
    plotter.set_background("white")
    plotter.add_text("Global Workspace", font_size=12, color="black")

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

    # --- 右侧视口: 尖端截面细节 ---
    plotter.subplot(0, 1)
    # 设置极浅的蓝灰色背景
    plotter.set_background("#F5F7FA")
    plotter.add_text("Tip Cross-section View (2D Control Plane)",
                     font_size=12, color="black")

    ellipse_segments = create_perspective_ellipse(
        z_height=tip_pos[2], cat_radius=cat_radius)
    for mesh, color, opacity in ellipse_segments:
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         smooth_shading=True)

    gradient_segments = create_smooth_2d_gradient_band(
        z_height=tip_pos[2] + 0.0005, cat_radius=cat_radius)
    for segment_mesh, color, opacity in gradient_segments:
        plotter.add_mesh(segment_mesh, color=color,
                         opacity=opacity, smooth_shading=True)

    np.random.seed(42)
    count, attempts = 0, 0
    while count < 35 and attempts < 200:
        attempts += 1
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(0.07, 0.28)
        ox, oy = dist * np.cos(angle), dist * np.sin(angle)
        base_r = np.random.uniform(0.008, 0.016)

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

        local_pos = [ox, oy, tip_pos[2] + 0.001]
        rand_val = np.random.rand()
        if rand_val < 0.33:
            obs_p = pv.Circle(radius=base_r, resolution=50)
            obs_p.points[:, 0] += ox
            obs_p.points[:, 1] += oy
            obs_p.points[:, 2] = tip_pos[2] + 0.001
        elif rand_val < 0.66:
            ratio = np.random.uniform(0.7, 1.3)
            w, h = base_r, base_r * ratio
            z = tip_pos[2] + 0.001
            obs_p = pv.Box(bounds=[ox-w, ox+w, oy-h, oy+h, z-0.0005, z+0.0005])
        else:
            obs_p = create_regular_polygon(local_pos, base_r, nsides=3)
        plotter.add_mesh(obs_p, color="red", style="wireframe",
                         line_width=1, opacity=0.8)
        count += 1

    # 3. 探测范围圆（改为淡绿色虚线圆）
    n_dashes = 40
    dash_angle = (2 * np.pi) / n_dashes
    for i in range(n_dashes):
        start_a = i * dash_angle
        end_a = start_a + dash_angle * 0.5
        dash_angles = np.linspace(start_a, end_a, 5)
        dash_r = cat_radius * 4
        dash_pts = np.zeros((5, 3))
        dash_pts[:, 0] = dash_r * np.cos(dash_angles)
        dash_pts[:, 1] = dash_r * np.sin(dash_angles)
        dash_pts[:, 2] = tip_pos[2] + 0.001
        dash_poly = pv.PolyData(dash_pts)
        dash_lines = np.full((4, 3), 2, dtype=np.int_)
        dash_lines[:, 1] = np.arange(4)
        dash_lines[:, 2] = np.arange(1, 5)
        dash_poly.lines = dash_lines
        plotter.add_mesh(dash_poly, color="#90EE90", line_width=1.5)

    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    cross_section_meshes = create_catheter_cross_section(
        z_height=tip_pos[2] + 0.003, cat_radius=cat_radius)
    for mesh, color, opacity in cross_section_meshes:
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         smooth_shading=True)

    plotter.camera.position = [0, 0, tip_pos[2] + 1.0]
    plotter.camera.focal_point = [0, 0, tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.show()


if __name__ == "__main__":
    main()
