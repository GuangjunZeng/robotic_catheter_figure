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
    # 计算尖端切向向量
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

    # 【方案 A 实现】：渐变尾迹体现连续体
    # 截取最后一段导管（例如最后 20 个点）
    tail_pts = smooth_pts[-25:]
    # 将这些点转换到以 tip 为原点、tip_dir 为 Z 轴的局部坐标系中
    # 这里为了演示，我们直接在 3D 空间中绘制，但将右侧相机对准它

    # 创建尾迹线
    tail_poly = pv.PolyData(tail_pts)
    tail_lines = np.full((len(tail_pts) - 1, 3), 2, dtype=np.int_)
    tail_lines[:, 1] = np.arange(len(tail_pts) - 1)
    tail_lines[:, 2] = np.arange(1, len(tail_pts))
    tail_poly.lines = tail_lines

    # 为尾迹添加标量值（从末端 1.0 到尖端 0.0，用于透明度和颜色渐变）
    scalars = np.linspace(0, 1, len(tail_pts))
    tail_poly.point_data["intensity"] = scalars

    # 生成带半径变化的管状体
    # 尖端半径为 cat_radius，向后逐渐变细
    tail_tube = tail_poly.tube(
        radius=cat_radius, radius_factor=0.2, scalars="intensity")

    # 添加渐变尾迹到右侧视口
    # 使用从黑色到白色的 colormap，并开启透明度映射
    plotter.add_mesh(tail_tube, scalars="intensity", cmap=["#333333", "white"],
                     show_scalar_bar=False, opacity="linear", smooth_shading=True)

    # 2. 探测范围
    # 在尖端平面绘制探测圆
    detection_range = pv.Circle(radius=cat_radius*4, resolution=50)
    # 将圆平移到 tip_pos 并对齐 tip_dir
    detection_range = detection_range.rotate_z(0)  # 占位
    # 这里为了简单，我们直接在右侧视口使用 2D 逻辑，但为了体现 A 方案，我们调整相机

    # 3. 规范化生成的障碍物 (在 tip 平面内生成)
    np.random.seed(42)
    for i in range(15):
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(cat_radius*1.5, 0.28)
        # 在局部 XY 平面生成位置
        local_pos = [dist * np.cos(angle), dist * np.sin(angle), 0]
        # 将局部位置转换到世界坐标系（以 tip_pos 为中心，垂直于 tip_dir）
        # 这里为了右侧视口的 view_xy，我们直接在 XY 平面绘制障碍物
        world_pos = [local_pos[0] + tip_pos[0],
                     local_pos[1] + tip_pos[1], tip_pos[2]]

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

    # 4. 探测范围圆 (在 tip_pos 平面)
    det_circle = pv.Circle(radius=cat_radius*4, resolution=50)
    det_circle.points += [0, 0, tip_pos[2]]
    plotter.add_mesh(det_circle, color="green",
                     style="wireframe", line_width=1.5)

    # 5. 浅灰色边框
    roi_border = pv.Box(
        bounds=[-0.3, 0.3, -0.3, 0.3, tip_pos[2]-0.001, tip_pos[2]+0.001])
    plotter.add_mesh(roi_border, color="lightgray",
                     style="wireframe", line_width=2)

    # 设置右侧相机：对准尖端，沿着导管方向看去
    # 相机位置在尖端前方一点，看向尖端，这样就能看到向后延伸的尾迹
    plotter.camera.position = [tip_pos[0], tip_pos[1], tip_pos[2] + 1.0]
    plotter.camera.focal_point = [tip_pos[0], tip_pos[1], tip_pos[2]]
    plotter.camera.up = [0, 1, 0]
    plotter.enable_parallel_projection()
    plotter.reset_camera()

    plotter.show()


if __name__ == "__main__":
    main()
