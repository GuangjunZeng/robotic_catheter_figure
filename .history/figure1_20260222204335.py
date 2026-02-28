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


BAND_LENGTH = 0.30    # 渐变带总长度（X 方向）
BAND_END_K = 0.45     # 末端宽度系数：末端宽度 = cat_radius * BAND_END_K
BAND_END_COLOR = 210  # 渐变带末端颜色灰度值 (#D2D2D2)，保留足够灰度不过淡
BAND_END_OPACITY = 0.28  # 渐变带末端透明度，保留足够可见度


def create_smooth_2d_gradient_band(z_height=0, cat_radius=0.04):
    """
    渐变带：末端宽度收缩到 cat_radius * BAND_END_K，不趋近于 0，
    与椭圆区域的 Y 半轴自然衔接。
    """
    n_segments = 100
    segment_list = []

    for seg_idx in range(n_segments):
        t_start = seg_idx / n_segments
        t_end = (seg_idx + 1) / n_segments
        t_params = np.linspace(t_start, t_end, 10)

        centerline_x = -BAND_LENGTH * t_params
        centerline_y = 0.0 - 0.06 * t_params * np.sin(np.pi * t_params)

        # 末端宽度不收缩到 0，而是收缩到 cat_radius * BAND_END_K
        widths = cat_radius * (1 - (1 - BAND_END_K) * t_params ** 1.5)

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
            # 55%-78%：#AAAAAA → #C0C0C0
            ratio = (progress - 0.55) / 0.23
            r_val = int(170 + (192 - 170) * ratio)
        else:
            # 78%-100%：#C0C0C0 → #D2D2D2 (BAND_END_COLOR)
            ratio = (progress - 0.78) / 0.22
            r_val = int(192 + (BAND_END_COLOR - 192) * ratio)

        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"

        # 透明度：末端保留 BAND_END_OPACITY，不完全消失
        opacity = max(0.92 * (1 - progress ** 0.7), BAND_END_OPACITY)
        segment_list.append((segment_mesh, color, opacity))

    return segment_list


def create_perspective_ellipse(z_height=0, cat_radius=0.04):
    """
    椭圆区域：从渐变带末端状态无缝延续，向外继续淡出至透明。

    衔接原则：
    - 最内层 (progress=0) 的高度严格等于渐变带末端高度：
      Y半轴 = cat_radius*BAND_END_K
    - 最内层保留适度宽度（X半轴不再过小），避免视觉过窄
    - 各层椭圆使用同一“右端锚点”x=-BAND_LENGTH，确保与渐变带末端
      在接缝处自然连续，不会出现断层/错位
    - 颜色与透明度从渐变带末端状态连续外扩并淡出
    """
    segment_list = []
    n_layers = 30

    # 内层参数：与渐变带末端完全对齐
    b_inner = cat_radius * BAND_END_K
    # 内层宽度适度增大，避免“太细”影响观感
    a_inner = b_inner * 0.28

    # 外层参数：向外扩展幅度缩小，保持与渐变带末端视觉协调
    b_outer = b_inner * 1.5
    a_outer = a_inner * 2.0
    right_anchor_x = -BAND_LENGTH

    n_pts = 80
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    for layer_idx in range(n_layers):
        progress = layer_idx / n_layers

        # 使用缓和增长，衔接处更平滑
        ease = progress ** 0.75
        a = a_inner + (a_outer - a_inner) * ease
        b = b_inner + (b_outer - b_inner) * ease
        # 右端锚点固定在渐变带末端：x = -BAND_LENGTH
        center_x = right_anchor_x - a

        pts = np.zeros((n_pts, 3))
        pts[:, 0] = center_x + a * np.cos(angles)
        pts[:, 1] = b * np.sin(angles)
        pts[:, 2] = z_height

        if layer_idx == 0:
            inner_a, inner_b = 0.0, 0.0
            inner_center_x = center_x
        else:
            prev = (layer_idx - 1) / n_layers
            prev_ease = prev ** 0.75
            inner_a = a_inner + (a_outer - a_inner) * prev_ease
            inner_b = b_inner + (b_outer - b_inner) * prev_ease
            inner_center_x = right_anchor_x - inner_a

        inner_pts = np.zeros((n_pts, 3))
        inner_pts[:, 0] = inner_center_x + inner_a * np.cos(angles)
        inner_pts[:, 1] = inner_b * np.sin(angles)
        inner_pts[:, 2] = z_height

        all_pts = np.vstack([inner_pts, pts])
        faces = []
        for i in range(n_pts):
            next_i = (i + 1) % n_pts
            faces.extend([3, i, next_i, n_pts + next_i])
            faces.extend([3, i, n_pts + next_i, n_pts + i])

        ring_mesh = pv.PolyData(all_pts, np.array(faces))

        # 颜色：从渐变带末端颜色 (#D2D2D2) 变浅到 #F2F2F2，最外层更浅
        r_val = int(BAND_END_COLOR + (242 - BAND_END_COLOR)
                    * (progress ** 1.2))
        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"

        # 透明度：用极慢的衰减（^0.25），且末端保留 0.06 的最低可见度
        opacity = max(BAND_END_OPACITY * (1 - progress ** 0.25), 0.06)
        segment_list.append((ring_mesh, color, opacity))

    # 最外层轮廓线：极淡的中灰色，暗示椭圆边界但不突兀
    outline_pts = np.zeros((n_pts + 1, 3))
    outline_center_x = right_anchor_x - a_outer
    outline_pts[:n_pts, 0] = outline_center_x + a_outer * np.cos(angles)
    outline_pts[:n_pts, 1] = b_outer * np.sin(angles)
    outline_pts[:n_pts, 2] = z_height + 0.0001
    outline_pts[n_pts] = outline_pts[0]

    outline_poly = pv.PolyData(outline_pts)
    outline_lines = np.full((n_pts, 3), 2, dtype=np.int_)
    outline_lines[:, 1] = np.arange(n_pts)
    outline_lines[:, 2] = np.arange(1, n_pts + 1)
    outline_poly.lines = outline_lines
    segment_list.append((outline_poly, "#BBBBBB", 0.20))

    return segment_list


def create_catheter_cross_section(z_height=0, cat_radius=0.04):
    """
    创建具有可见径向渐变的导管截面：
    - 中心颜色进一步提亮，避免“整体发黑”
    - 外边缘颜色保持 #222222，与渐变带起始颜色一致
    - 渐变速度适度加快，保证肉眼可辨识
    - 多层同心圆环实现径向渐变
    """
    meshes = []
    n_rings = 28  # 适当减少层数，让每层亮度差更可见

    # 中心色灰度（继续提亮，避免视觉上“太黑”）
    center_gray = 150   # #969696
    # 外边缘色灰度：与渐变带起始颜色对齐
    edge_gray = 34      # #222222

    for ring_idx in range(n_rings):
        # progress 从 0（中心）到 1（外边缘）
        progress = ring_idx / n_rings
        next_progress = (ring_idx + 1) / n_rings

        inner_r = cat_radius * progress
        outer_r = cat_radius * next_progress

        # 颜色：加快渐变速度，提升中心到边缘的可见差异
        ease = progress ** 1.25
        r_val = int(center_gray + (edge_gray - center_gray) * ease)
        color = f"#{r_val:02x}{r_val:02x}{r_val:02x}"

        ring = pv.Disc(center=[0, 0, z_height], inner=inner_r,
                       outer=outer_r, normal=[0, 0, 1], c_res=80)
        meshes.append((ring, color, 1.0))

    # 外轮廓线：与渐变带起始颜色一致，确保衔接无缝
    outline = pv.Circle(radius=cat_radius, resolution=80)
    outline.points[:, 2] = z_height + 0.001
    meshes.append((outline, "#222222", 1.0))

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

    obs_size = cat_radius * 1.5
    # 障碍物位置
    box_pos = [[1.4, 2.11, 1.8]]     # 立方体的位置
    sphere_pos = [[1.6, 1.8, 2.6]]  # 球的位置
    tetra_pos = [[0.6, 2.15, 2.2]]   # 四面体的位置

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
            plotter, pos, [0.01, -0.001, 0.1] if i == 0 else [-0.1, 0.2, -0.1])  # 立方体的方向的箭头位置

    for i, pos in enumerate(sphere_pos):
        s = pv.Sphere(radius=obs_size, center=pos,
                      phi_resolution=10, theta_resolution=10)
        plotter.add_mesh(s, color="red", style="wireframe",
                         line_width=0.8, opacity=0.6)
        add_elegant_velocity(
            plotter, pos, [0.1, 0.2, -0.05] if i == 0 else [0.2, 0.1, -0.1])  # 球的方向的箭头位置

    for i, pos in enumerate(tetra_pos):
        s = obs_size * 1.2
        pts = np.array([[pos[0]+s, pos[1]+s, pos[2]+s], [pos[0]-s, pos[1]-s, pos[2]+s],
                        [pos[0]-s, pos[1]+s, pos[2]-s], [pos[0]+s, pos[1]-s, pos[2]-s]])
        faces = np.array([3, 0, 1, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 1, 2, 3])
        tetra = pv.PolyData(pts, faces)
        plotter.add_mesh(tetra, color="red", style="wireframe",
                         line_width=1, opacity=0.8)
        add_elegant_velocity(
            plotter, pos, [0.2, 0.03, -0.05] if i == 0 else [-0.2, -0.1, 0.2])  # 四面体的方向的箭头位置

    # roi_plane = pv.Plane(center=tip_pos, direction=tip_dir,
    #                      i_size=0.6, j_size=0.6)
    # plotter.add_mesh(roi_plane, color="lightgray",
    #                  style="wireframe", line_width=1, opacity=0.5)
    # plotter.add_mesh(roi_plane, color="lightgray", opacity=0.05)
    plotter.add_axes()
    plotter.camera_position = [
        (6.0, 4.0, 5.0), (1.0, 1.2, 1.2), (0.0, 0.0, 1.0)]

    # --- 右侧视口: 尖端截面细节 ---
    plotter.subplot(0, 1)
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

    # 手动放置三个特定形状的障碍物，确保非对称分布
    z_obs = tip_pos[2] + 0.001
    base_r = 0.015 * 1.2  # 增大 1.2 倍

    # 2D 箭头函数：纯线条实现，无任何 3D 几何体，避免矩形块
    def add_2d_velocity(plotter, center, direction, size=0.04, gap=0.025):  # 箭头后端的长度
        d = np.array([direction[0], direction[1], 0], dtype=float)
        d = d / np.linalg.norm(d)
        start_pt = np.array([center[0], center[1], z_obs]) + d * gap
        end_pt = start_pt + d * size
        # 箭杆
        line = pv.Line(start_pt, end_pt)
        plotter.add_mesh(line, color="hotpink", line_width=2)
        # V 形箭头尖端（纯线条）
        perp = np.array([-d[1], d[0], 0])
        tip_size = size * 0.2
        tip1 = pv.Line(end_pt - d * tip_size + perp * (tip_size * 0.6), end_pt)
        tip2 = pv.Line(end_pt - d * tip_size - perp *
                       (tip_size * 0.6), end_pt)  # V 形尖端角度
        plotter.add_mesh(tip1, color="hotpink", line_width=2)
        plotter.add_mesh(tip2, color="hotpink", line_width=2)

    # 1. 圆形 (Circle) - 偏左上
    ox_c, oy_c = -0.08, 0.1  # 圆形位置
    obs_circle = pv.Circle(radius=base_r, resolution=50)
    obs_circle.points[:, 0] += ox_c
    obs_circle.points[:, 1] += oy_c
    obs_circle.points[:, 2] = z_obs
    plotter.add_mesh(obs_circle, color="red",
                     style="wireframe", line_width=1, opacity=0.8)
    add_2d_velocity(plotter, [ox_c, oy_c, z_obs], [0.6, -0.5])

    # 2. 三角形 (Triangle) - 偏右上
    ox_t, oy_t = 0.11, 0.06  # 三角形位置
    obs_triangle = create_regular_polygon(
        [ox_t, oy_t, z_obs], base_r * 1.22, nsides=3)  # 仅将三角形尺寸再增大 1.15 倍
    plotter.add_mesh(obs_triangle, color="red",
                     style="wireframe", line_width=1, opacity=0.8)
    add_2d_velocity(plotter, [ox_t, oy_t, z_obs], [-0.4, -0.6])  # 三角形的箭头的位置

    # 3. 正方形 (Square) - 偏左下
    ox_s, oy_s = -0.05, -0.16
    obs_square = pv.Box(bounds=[ox_s-base_r, ox_s+base_r,
                        oy_s-base_r, oy_s+base_r, z_obs-0.0005, z_obs+0.0005])
    plotter.add_mesh(obs_square, color="red",
                     style="wireframe", line_width=1, opacity=0.8)
    add_2d_velocity(plotter, [ox_s, oy_s, z_obs], [0.2, 0.6])

    # --- 新增：Catheter 尖端运动方向箭头 (粗绿色箭头) ---
    # 设定在中心圆的右下方
    tip_action_dir = np.array([0.4, -0.65, 0.0])  # 避障方向
    tip_action_dir = tip_action_dir / np.linalg.norm(tip_action_dir)

    # 起点稍微离开中心圆边缘 (cat_radius=0.04)
    tip_arrow_start = tip_action_dir * (cat_radius + 0.01)
    tip_arrow_start[2] = z_obs
    tip_arrow_size = 0.08

    # 使用较粗的线条和实心箭头体现“控制动作”
    # 箭杆
    tip_arrow_end = tip_arrow_start + tip_action_dir * tip_arrow_size
    tip_line = pv.Line(tip_arrow_start, tip_arrow_end)
    plotter.add_mesh(tip_line, color="limegreen", line_width=5)  # 增加粗度

    # 箭头尖端 (使用小三角形模拟 2D 实心效果)
    tip_perp = np.array([-tip_action_dir[1], tip_action_dir[0], 0])
    ts = 0.025  # 箭头大小
    tip_pts = np.array([
        tip_arrow_end,
        tip_arrow_end - tip_action_dir * ts + tip_perp * (ts * 0.6),
        tip_arrow_end - tip_action_dir * ts - tip_perp * (ts * 0.6)
    ])
    tip_face = np.array([3, 0, 1, 2])
    tip_head = pv.PolyData(tip_pts, tip_face)
    plotter.add_mesh(tip_head, color="limegreen")

    # 在右侧图左下角添加二维坐标轴 (X, Y) - 优化为 2D 线条风格，避免出现矩形块
    axis_origin = [-0.45, -0.45, z_obs]
    axis_length = 0.08

    # X 轴：红色线条
    x_axis = pv.Line(
        axis_origin, [axis_origin[0] + axis_length, axis_origin[1], z_obs])
    # Y 轴：绿色线条
    y_axis = pv.Line(
        axis_origin, [axis_origin[0], axis_origin[1] + axis_length, z_obs])

    plotter.add_mesh(x_axis, color="red", line_width=3)
    plotter.add_mesh(y_axis, color="green", line_width=3)

    # 添加箭头尖端：用两条斜线拼成 V 形，纯线条无厚度
    tip_size = 0.012
    x_end = [axis_origin[0] + axis_length, axis_origin[1], z_obs]
    x_tip1 = pv.Line([x_end[0] - tip_size, x_end[1] + tip_size, z_obs], x_end)
    x_tip2 = pv.Line([x_end[0] - tip_size, x_end[1] - tip_size, z_obs], x_end)
    plotter.add_mesh(x_tip1, color="red", line_width=3)
    plotter.add_mesh(x_tip2, color="red", line_width=3)

    y_end = [axis_origin[0], axis_origin[1] + axis_length, z_obs]
    y_tip1 = pv.Line([y_end[0] - tip_size, y_end[1] - tip_size, z_obs], y_end)
    y_tip2 = pv.Line([y_end[0] + tip_size, y_end[1] - tip_size, z_obs], y_end)
    plotter.add_mesh(y_tip1, color="green", line_width=3)
    plotter.add_mesh(y_tip2, color="green", line_width=3)

    # 标签颜色与坐标轴对应
    plotter.add_point_labels([[axis_origin[0] + axis_length + 0.02, axis_origin[1], z_obs]],
                             ["X"], font_size=12, text_color="red", show_points=False)
    plotter.add_point_labels([[axis_origin[0], axis_origin[1] + axis_length + 0.02, z_obs]],
                             ["Y"], font_size=12, text_color="green", show_points=False)

    # 3. 探测范围圆（已移除绿色虚线圆）

    cross_section_meshes = create_catheter_cross_section(
        z_height=tip_pos[2] + 0.003, cat_radius=cat_radius)
    for mesh, color, opacity in cross_section_meshes:
        # 关闭光照着色，按设置的灰度原样显示，避免视觉上“全黑一块”
        plotter.add_mesh(mesh, color=color, opacity=opacity,
                         smooth_shading=False, lighting=False)

    plotter.camera.position = [0, 0, tip_pos[2] + 1.0]
    plotter.camera.focal_point = [0, 0, tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()
    plotter.show()


if __name__ == "__main__":
    main()
