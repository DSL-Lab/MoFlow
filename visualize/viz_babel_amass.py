import json
from pathlib import Path

import numpy as np
import torch
import smplx
import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3d)
import imageio.v2 as imageio
from pathlib import Path

# ===================== 配置区域 =====================
BABEL_JSON_PATH = r"D:\07_data\BABEL\train.json"  # 修改成你的 BABEL json
AMASS_ROOT = Path(r"D:\04_code\HumanML3D\amass_data")                 # 修改成你的 AMASS 根目录
SMPLX_MODEL_PATH = r"D:\04_code\HumanML3D\smplx"          # 修改成你的 SMPL(H) 模型目录
BABEL_SID = "5788"                                  # 就用你给的这条样例
# ==================================================


def load_babel_entry(babel_json_path, sid):
    """根据 sid 读取一条 BABEL 样本（兼容不同顶层结构）"""
    with open(babel_json_path, "r") as f:
        data = json.load(f)

    # 有的版本是 {"seqs": {sid: {...}}}
    if "seqs" in data:
        seqs = data["seqs"]
    else:
        seqs = data

    if sid not in seqs:
        raise KeyError(f"sid={sid} not found in BABEL json")

    return seqs[sid]


def load_amass_npz(amass_root: Path, feat_p: str):
    """根据 BABEL 中的 feat_p 字段读取 AMASS 的 npz"""
    npz_path = amass_root / feat_p
    if not npz_path.exists():
        raise FileNotFoundError(f"AMASS npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return data


def smplh_to_joints_3d(npz_data, model_path, device="cpu"):
    poses = npz_data["poses"]         # (T, 156)
    trans = npz_data["trans"]         # (T, 3)
    betas = npz_data["betas"]         # (nbetas,) 这里是 16
    gender_raw = npz_data["gender"]
    fps = float(npz_data["mocap_framerate"])

    if isinstance(gender_raw, np.ndarray):
        gender_raw = gender_raw.item()
    gender = str(gender_raw)
    if gender not in ["male", "female", "neutral"]:
        gender = "neutral"

    T = poses.shape[0]
    device = torch.device(device)

    model = smplx.create(
        model_path=model_path,
        model_type="smplh",
        gender=gender,
        use_pca=False,      # 手部用 45 维轴角
        batch_size=T,
    ).to(device)

    # === 关键：根据模型的 shapedirs 维度裁剪 betas ===
    num_betas_model = model.shapedirs.shape[-1]   # 这里是 10
    # 只取前 num_betas_model 维
    betas_np = betas[:num_betas_model]            # (10,)

    betas_t = torch.tensor(
        np.repeat(betas_np[None, :], T, axis=0),  # (T, 10)
        dtype=torch.float32,
        device=device,
    )

    # 拆 pose
    global_orient = torch.tensor(poses[:, :3], dtype=torch.float32, device=device)
    body_pose = torch.tensor(poses[:, 3:66], dtype=torch.float32, device=device)
    left_hand  = torch.tensor(poses[:, 66:66+45], dtype=torch.float32, device=device)
    right_hand = torch.tensor(poses[:, 66+45:], dtype=torch.float32, device=device)
    transl     = torch.tensor(trans, dtype=torch.float32, device=device)

    output = model(
        betas=betas_t,
        global_orient=global_orient,
        body_pose=body_pose,
        left_hand_pose=left_hand,
        right_hand_pose=right_hand,
        transl=transl,
    )

    joints_3d_full = output.joints.detach().cpu().numpy()
    joints_3d_22 = joints_3d_full[:, :22, :]
    print("joints_3d shape = {}".format(joints_3d_22.shape))
    return joints_3d_22, fps, gender



def build_frame_labels(frame_ann, T, fps):
    """
    把 BABEL 的 frame_ann 转成逐帧标签列表 (len=T)，
    每一帧是一个字符串 label（如 'stand', 'pace'，或 'none'）
    """
    labels = ["none"] * T

    if frame_ann is None:
        return labels

    for seg in frame_ann.get("labels", []):
        cmd = seg["proc_label"]  # 或 act_cat[0]
        start_t = float(seg["start_t"])
        end_t = float(seg["end_t"])
        start_f = int(start_t * fps)
        end_f = int(end_t * fps)

        start_f = max(0, min(T, start_f))
        end_f = max(0, min(T, end_f))
        for f in range(start_f, end_f):
            labels[f] = cmd

    return labels


def visualize_sample(joints_3d, frame_labels, sid):
    """
    简单可视化：
    - 左：某一帧的 3D skeleton（只画散点）
    - 右：根节点轨迹（x-z 平面）并用颜色表示动作标签
    """
    T, J, _ = joints_3d.shape
    root_traj = joints_3d[:, 0, :]  # 根节点 (T, 3)

    # 选一个中间帧来画 skeleton
    mid = T // 2
    skel = joints_3d[mid]

    # 给标签一个颜色映射
    uniq_labels = sorted(set(frame_labels))
    color_map = {lab: idx for idx, lab in enumerate(uniq_labels)}
    colors = [color_map[lab] for lab in frame_labels]

    fig = plt.figure(figsize=(10, 4))

    # ---- 子图1：3D skeleton (scatter) ----
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(skel[:, 0], skel[:, 1], skel[:, 2], s=10)
    ax1.set_title(f"BABEL {sid} - 3D joints (frame {mid})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.view_init(elev=20., azim=-70)

    # ---- 子图2：根节点轨迹 (x-z 平面)，用颜色区分标签 ----
    ax2 = fig.add_subplot(1, 2, 2)
    sc = ax2.scatter(root_traj[:, 0], root_traj[:, 2], c=colors, s=5, cmap="viridis")
    ax2.set_title("Root trajectory (colored by frame label)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")

    # 构造 legend
    # 取每个 label 的一个代表点
    handles = []
    for lab, idx in color_map.items():
        handles.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                linestyle='',
                color=plt.cm.viridis(idx / max(1, len(color_map)-1)),
                label=lab
            )
        )
    ax2.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


def save_skeleton_video(
    joints_3d,
    frame_labels,
    sid,
    out_path,
    fps=30,
    downsample=1,
):
    """
    将 3D 关键点序列保存为视频，在每一帧下方标注动作标签。

    Args:
        joints_3d: np.ndarray, 形状 (T, J, 3)
        frame_labels: list[str] 或 np.ndarray，长度 T
        sid: str 或 int，BABEL 的 sample id（仅用于标题显示）
        out_path: str 或 Path，输出视频路径，如 'babel_5788.mp4' 或 'babel_5788.gif'
        fps: int，视频帧率
        downsample: int，对时间维做下采样（例如=2 时只取每隔一帧）
    """
    joints_3d = np.asarray(joints_3d)
    T, J, C = joints_3d.shape
    assert C == 3, f"joints_3d 应为 (T, J, 3)，但得到 {joints_3d.shape}"
    assert len(frame_labels) == T, "frame_labels 长度必须等于 T"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- 统一坐标范围，避免画面抖动 --------
    xs = joints_3d[:, :, 0]
    ys = joints_3d[:, :, 1]
    zs = joints_3d[:, :, 2]

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    # 用最大的跨度设一个立方包围盒
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_center = (x_max + x_min) / 2.0
    y_center = (y_max + y_min) / 2.0
    z_center = (z_max + z_min) / 2.0

    half = max_range / 2.0 * 1.1  # 略加一点 margin
    x_lim = (x_center - half, x_center + half)
    y_lim = (y_center - half, y_center + half)
    z_lim = (z_center - half, z_center + half)

    # -------- 准备写视频 --------
    frames = []

    # 如果想画骨架连线，可以在这里定义关节点连接关系（可选）
    # 这里先简单只画关键点
    for t in range(0, T, downsample):
        if t % 10 == 0:
            print("t = {}".format(t))
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        skel = joints_3d[t]  # (J, 3)
        ax.scatter(skel[:, 0], skel[:, 1], skel[:, 2], s=10)

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        ax.view_init(elev=20., azim=-70)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"BABEL {sid} - frame {t}")

        # 在图像底部标注动作标签
        label = frame_labels[t]
        fig.text(
            0.5, 0.02,
            f"action: {label}",
            ha="center",
            va="bottom",
            fontsize=10
        )

        # 调整布局，为底部文字留空间
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # 将当前 Figure 转成 RGB 数组
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()
        img = renderer.buffer_rgba()  # 替代 tostring_rgb
        # img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # img = img.reshape((height, width, 3))
        frames.append(img)

        plt.close(fig)

    # -------- 写出视频文件 --------
    # 根据后缀自动写 mp4/gif 等，mp4 需要安装 imageio-ffmpeg
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[INFO] saved video to {out_path} (frames={len(frames)}, fps={fps})")


def main():
    # 1. 读取 BABEL 标签
    babel_entry = load_babel_entry(BABEL_JSON_PATH, BABEL_SID)
    feat_p = babel_entry["feat_p"]
    frame_ann = babel_entry.get("frame_ann", None)

    print(f"[INFO] BABEL sid={BABEL_SID}")
    print(f"       feat_p = {feat_p}")
    if frame_ann is not None:
        print(f"       #frame labels = {len(frame_ann.get('labels', []))}")

    # 2. 加载 AMASS npz
    amass_data = load_amass_npz(AMASS_ROOT, feat_p)

    # 3. SMPL-H → 3D joints
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    joints_3d, fps, gender = smplh_to_joints_3d(amass_data, SMPLX_MODEL_PATH, device=device)
    T = joints_3d.shape[0]
    print(f"[INFO] joints_3d shape = {joints_3d.shape}, fps={fps}, gender={gender}")

    # 4. 把 frame_ann 映射成逐帧标签
    frame_labels = build_frame_labels(frame_ann, T, fps)
    print(f"[INFO] unique frame labels: {set(frame_labels)}")

    # 5. 可视化一条样本
    # visualize_sample(joints_3d, frame_labels, sid=BABEL_SID)
    video_out_path = r"D:\04_code\MoFlow\visualize\result\viz_babel\vid.mp4"
    save_skeleton_video(joints_3d, frame_labels, sid=BABEL_SID, out_path=video_out_path)

if __name__ == "__main__":
    main()
