import json
from pathlib import Path

import math
import numpy as np
import torch
import smplx


# ===================== 配置区域 =====================
BABEL_JSON_PATH = r"D:\07_data\BABEL\train.json"  # 修改成你的 BABEL json
AMASS_ROOT = Path(r"D:\04_code\HumanML3D\amass_data")                 # 修改成你的 AMASS 根目录
SMPL_MODEL_ROOT = r"D:\04_code\HumanML3D\smplx"          # 修改成你的 SMPL(H) 模型目录

Th = 30   # 历史长度（帧数）
Tf = 30   # 未来长度
STRIDE = 15  # 滑窗步长（帧），可以=1 做最大重叠

USE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMPL_EXT = "npz"  # 如果模型文件是 .npz 就写 "npz"，是 .pkl 就写 "pkl"
# ==================================================

def ids_to_binary_matrix(ids, num_bits):
    """
    ids: (T,) int，类别ID，-1 表示无标签
    num_bits: 使用的二进制位数，例如 13

    返回:
        bin_mat: (T, num_bits) float32，0/1 编码
    """
    ids = np.asarray(ids).astype(np.int64)
    # 无标签的帧统一编码为 0
    ids = ids.copy()
    ids[ids < 0] = 0

    # (T, 1) >> (num_bits,) → (T, num_bits)
    # 这里用 little-endian：第0位是最低位
    bit_indices = np.arange(num_bits, dtype=np.int64)
    bin_mat = ((ids[:, None] >> bit_indices[None, :]) & 1).astype(np.float32)
    return bin_mat

def load_babel_seqs(babel_json_path):
    """加载 BABEL json，并返回 seq 字典 {sid: entry}"""
    with open(babel_json_path, "r") as f:
        data = json.load(f)

    if "seqs" in data:
        seqs = data["seqs"]
    else:
        seqs = data
    return seqs


def collect_all_labels(seqs):
    """扫描所有样本，收集 frame_ann 中的所有 proc_label，构建 label_vocab"""
    label_set = set()
    for sid, entry in seqs.items():
        frame_ann = entry.get("frame_ann", None)
        if frame_ann is None:
            continue
        for seg in frame_ann.get("labels", []):
            proc_label = seg.get("proc_label", None)
            if proc_label is not None:
                label_set.add(proc_label)
    label_list = sorted(label_set)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    return label2id


def load_amass_npz(amass_root: Path, feat_p: str):
    npz_path = amass_root / feat_p
    if not npz_path.exists():
        raise FileNotFoundError(f"AMASS npz not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return npz_path, data


class SMPLHDecoder:
    """缓存 gender->SMPLH 模型，避免每条样本都重新 create。"""

    def __init__(self, model_root, device="cpu", ext="npz"):
        self.model_root = model_root
        self.device = torch.device(device)
        self.ext = ext
        self.models = {}  # gender -> model

    def get_model(self, gender: str, batch_size: int):
        gender = gender.lower()
        if gender not in ["male", "female", "neutral"]:
            gender = "neutral"

        key = (gender, batch_size)
        if key in self.models:
            return self.models[key]

        model = smplx.create(
            model_path=self.model_root,
            model_type="smplh",
            gender=gender,
            use_pca=False,  # 手部用 45 维轴角
            batch_size=batch_size,
        ).to(self.device)

        self.models[key] = model
        return model

    def decode_joints(self, npz_data):
        poses = npz_data["poses"]         # (T, 156)
        trans = npz_data["trans"]         # (T, 3)
        betas = npz_data["betas"]         # (nbetas,)
        gender_raw = npz_data["gender"]   # 可能是 bytes
        fps = float(npz_data["mocap_framerate"])

        if isinstance(gender_raw, np.ndarray):
            gender_raw = gender_raw.item()
        gender = str(gender_raw)
        if gender not in ["male", "female", "neutral"]:
            gender = "neutral"

        T = poses.shape[0]
        model = self.get_model(gender, batch_size=T)

        # === 关键：根据模型的 shapedirs 维度裁剪 betas ===
        num_betas_model = model.shapedirs.shape[-1]  # 这里是 10
        # 只取前 num_betas_model 维
        betas_np = betas[:num_betas_model]  # (10,)

        device = self.device
        global_orient = torch.tensor(poses[:, :3], dtype=torch.float32, device=device)
        body_pose     = torch.tensor(poses[:, 3:66], dtype=torch.float32, device=device)
        left_hand     = torch.tensor(poses[:, 66:66+45], dtype=torch.float32, device=device)
        right_hand    = torch.tensor(poses[:, 66+45:], dtype=torch.float32, device=device)

        betas_t = torch.tensor(
            np.repeat(betas_np[None, :], T, axis=0),
            dtype=torch.float32,
            device=device,
        )
        transl = torch.tensor(trans, dtype=torch.float32, device=device)

        with torch.no_grad():
            output = model(
                betas=betas_t,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand,
                right_hand_pose=right_hand,
                transl=transl,
            )

        joints_3d = output.joints.detach().cpu().numpy()  # (T, J, 3)
        return joints_3d, fps, gender


def build_frame_labels_binary(frame_ann, T, fps, label2id, num_bits):
    """
    把 BABEL 的 frame_ann 转成二进制编码标签矩阵 (T, num_bits)

    - 每一帧先赋一个整数 cid（-1 表示 none）
    - 再转成二进制编码，得到 (T, num_bits)
    """
    # 先存整数ID
    cid_per_frame = np.full(T, fill_value=-1, dtype=np.int64)

    if frame_ann is not None:
        for seg in frame_ann.get("labels", []):
            proc_label = seg.get("proc_label", None)
            if proc_label is None or proc_label not in label2id:
                continue
            cid = label2id[proc_label]
            start_t = float(seg["start_t"])
            end_t   = float(seg["end_t"])
            start_f = int(start_t * fps)
            end_f   = int(end_t * fps)

            start_f = max(0, min(T, start_f))
            end_f   = max(0, min(T, end_f))
            cid_per_frame[start_f:end_f] = cid

    # 再转成二进制矩阵 (T, num_bits)
    u_frame_bin = ids_to_binary_matrix(cid_per_frame, num_bits)
    return u_frame_bin        # float32, shape (T, num_bits)

def sliding_windows(joints_3d, u_frame, Th, Tf, stride):
    """
    从一条序列中切出多条 (hist, fut) 样本。
    joints_3d: (T, J, 3)
    u_frame:   (T, C)
    返回:
      x_hist_list: list[(Th, J, 3)]
      u_hist_list: list[(Th, C)]
      x_fut_list:  list[(Tf, J, 3)]
      u_fut_list:  list[(Tf, C)]
    """
    T = joints_3d.shape[0]
    x_hist_list, u_hist_list, x_fut_list, u_fut_list = [], [], [], []

    min_len = Th + Tf
    if T < min_len:
        return x_hist_list, u_hist_list, x_fut_list, u_fut_list

    for start in range(0, T - min_len + 1, stride):
        mid = start + Th
        end = mid + Tf

        x_hist = joints_3d[start:mid]      # (Th, J, 3)
        x_fut  = joints_3d[mid:end]        # (Tf, J, 3)
        u_hist = u_frame[start:mid]        # (Th, C)
        u_fut  = u_frame[mid:end]          # (Tf, C)

        x_hist_list.append(x_hist)
        u_hist_list.append(u_hist)
        x_fut_list.append(x_fut)
        u_fut_list.append(u_fut)

    return x_hist_list, u_hist_list, x_fut_list, u_fut_list


def build_babel_dataset(
    babel_json_path,
    amass_root,
    smpl_model_root,
    Th,
    Tf,
    stride=1,
    device="cpu",
    smpl_ext="npz",
    max_seqs=None,
):
    """
    核心函数：将 BABEL + AMASS 批量转换为:
      x_hist: (B, Th, J, 3)
      u_hist: (B, Th, C)
      x_fut:  (B, Tf, J, 3)
      u_fut:  (B, Tf, C)

    Args:
        babel_json_path: BABEL json 路径
        amass_root: AMASS 根目录 (Path)
        smpl_model_root: SMPL-H 模型根目录 (str)
        Th, Tf: 历史/未来长度
        stride: 时间滑窗步长
        device: "cpu" 或 "cuda"
        smpl_ext: "npz" 或 "pkl"
        max_seqs: 只处理前 max_seqs 条（调试用）

    Returns:
        x_hist, u_hist, x_fut, u_fut, label2id
    """
    seqs = load_babel_seqs(babel_json_path)
    print(f"[INFO] total BABEL seqs: {len(seqs)}")

    label2id = collect_all_labels(seqs)
    num_labels = len(label2id)
    num_bits = math.ceil(math.log2(num_labels + 1))  # +1 给无标签留一个编码

    print(f"[INFO] collected {len(label2id)} labels:", label2id)
    print(f"[INFO] num_labels={num_labels}, using num_bits={num_bits} for binary encoding")

    decoder = SMPLHDecoder(
        model_root=smpl_model_root,
        device=device,
        ext=smpl_ext,
    )

    all_x_hist, all_u_hist, all_x_fut, all_u_fut = [], [], [], []

    for idx, (sid, entry) in enumerate(seqs.items()):
        if max_seqs is not None and idx >= max_seqs:
            break

        feat_p = entry.get("feat_p", None)
        frame_ann = entry.get("frame_ann", None)
        if feat_p is None:
            continue

        try:
            _, amass_data = load_amass_npz(amass_root, feat_p)
        except FileNotFoundError as e:
            print("[WARN]", e)
            continue

        joints_3d_full, fps, gender = decoder.decode_joints(amass_data)
        joints_3d = joints_3d_full[:, :22, :]
        T, J, _ = joints_3d.shape

        u_frame = build_frame_labels_binary(
            frame_ann, T, fps,
            label2id=label2id,
            num_bits=num_bits
        )  # (T, num_bits)

        xh_list, uh_list, xf_list, uf_list = sliding_windows(
            joints_3d, u_frame, Th, Tf, stride
        )

        if len(xh_list) == 0:
            continue

        all_x_hist.extend(xh_list)
        all_u_hist.extend(uh_list)
        all_x_fut.extend(xf_list)
        all_u_fut.extend(uf_list)

        if (idx + 1) % 50 == 0:
            print(f"[INFO] processed {idx+1} seqs, current samples: {len(all_x_hist)}")

    if len(all_x_hist) == 0:
        raise RuntimeError("No valid samples extracted. Check Th/Tf/stride or data paths.")

    x_hist = np.stack(all_x_hist, axis=0)  # (B, Th, J, 3)
    u_hist = np.stack(all_u_hist, axis=0)  # (B, Th, C)
    x_fut  = np.stack(all_x_fut,  axis=0)  # (B, Tf, J, 3)
    u_fut  = np.stack(all_u_fut,  axis=0)  # (B, Tf, C)

    print(f"[DONE] x_hist: {x_hist.shape}, u_hist: {u_hist.shape}")
    print(f"       x_fut:  {x_fut.shape},  u_fut:  {u_fut.shape}")
    return x_hist, u_hist, x_fut, u_fut, label2id


import numpy as np
from pathlib import Path

def save_babel_splits(
    x_hist, u_hist, x_fut, u_fut, label2id,
    save_root,
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1
):

    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    B = x_hist.shape[0]
    print(f"[INFO] Total samples: {B}")

    # --------------------------
    # 1. flatten成你现在模型的格式
    # --------------------------
    # (Th, J, 3) + (Tf, J, 3) → (Th+Tf, J, 3)
    x_seq = np.concatenate([x_hist, x_fut], axis=1)   # (B, Th+Tf, J, 3)
    u_seq = np.concatenate([u_hist, u_fut], axis=1)   # (B, Th+Tf, C)

    T_total = x_seq.shape[1]
    J = x_seq.shape[2]
    C = u_seq.shape[2]
    print(f"[INFO] Each sequence: x_seq={x_seq.shape}, u_seq={u_seq.shape}")

    # --------------------------
    # 2. 打乱
    # --------------------------
    idx = np.random.permutation(B)
    x_seq = x_seq[idx]
    u_seq = u_seq[idx]

    # --------------------------
    # 3. 划分 7:2:1
    # --------------------------
    B_train = int(B * train_ratio)
    B_val   = int(B * val_ratio)
    B_test  = B - B_train - B_val

    x_train, u_train = x_seq[:B_train], u_seq[:B_train]
    x_val,   u_val   = x_seq[B_train:B_train+B_val], u_seq[B_train:B_train+B_val]
    x_test,  u_test  = x_seq[B_train+B_val:],        u_seq[B_train+B_val:]

    print(f"[INFO] train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")

    # --------------------------
    # 4. 计算统计信息 (均值/方差)
    # --------------------------
    x_all = x_seq.reshape(B, -1)
    mean = x_all.mean(axis=0)
    std  = x_all.std(axis=0) + 1e-6
    stats = {"mean": mean, "std": std}

    # --------------------------
    # 5. 保存
    # --------------------------
    np.save(save_root / "babel_train.npy",     x_train)
    np.save(save_root / "babel_train_cmd.npy", u_train)

    np.save(save_root / "babel_val.npy",       x_val)
    np.save(save_root / "babel_val_cmd.npy",   u_val)

    np.save(save_root / "babel_test.npy",      x_test)
    np.save(save_root / "babel_test_cmd.npy",  u_test)

    np.save(save_root / "babel_stats.npy",     stats)
    np.save(save_root / "label2id.npy",        label2id)

    print("[DONE] Saved babel dataset to:", save_root)


if __name__ == "__main__":
    x_hist, u_hist, x_fut, u_fut, label2id = build_babel_dataset(
        babel_json_path=BABEL_JSON_PATH,
        amass_root=AMASS_ROOT,
        smpl_model_root=SMPL_MODEL_ROOT,
        Th=Th,
        Tf=Tf,
        stride=STRIDE,
        device=USE_DEVICE,
        smpl_ext=SMPL_EXT,
        max_seqs=100,  # 先调试一下，OK 后可以去掉或改大
    )
    print("x_hist shape = {}".format(x_hist.shape))
    print("u_hist shape = {}".format(u_hist.shape))
    print("x_fut shape = {}".format(x_fut.shape))
    print("u_fut shape = {}".format(u_fut.shape))
    # print("label2id = {}".format(label2id))

    # 例如保存为 npz，后面直接加载训练
    save_root = Path(r"D:\04_code\MoFlow\data\babel")
    save_babel_splits(
        x_hist, u_hist, x_fut, u_fut, label2id,
        save_root,
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1
    )
    print(f"[SAVE] saved to {save_root}")
