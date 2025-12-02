import numpy as np
import matplotlib.pyplot as plt

import numpy as np


import numpy as np

def compute_bounding_box(history_pos,
                         preds_pos,
                         gt_pos,
                         padding: int = 10,
                         square: bool = False):
    """
    计算三条轨迹的最小 bounding box，并可选地扩展为正方形视野。

    参数
    ----
    history_pos : np.ndarray, shape (Th, 2)
    preds_pos   : list[np.ndarray(Tf, 2)] 或 np.ndarray, shape (K, Tf, 2)
    gt_pos      : np.ndarray, shape (Tf, 2)
    padding     : int, 在四周额外扩展的边距（坐标同单位）
    square      : bool, 若为 True，则：
                  - 令 side = max(xmax - xmin, ymax - ymin)
                  - 以 bbox 中心为中心取正方形区域
                  - 再在此基础上加 padding
    返回
    ----
    xmin, xmax, ymin, ymax : int
    """

    # -------- 处理 preds_pos 到统一形状 (N_pred, 2) --------
    if isinstance(preds_pos, list):
        if len(preds_pos) > 0:
            preds_stack = np.concatenate(preds_pos, axis=0)  # (K*Tf, 2)
        else:
            preds_stack = np.zeros((0, 2), dtype=float)
    else:
        # np.ndarray (K, Tf, 2)
        preds_stack = preds_pos.reshape(-1, 2) if preds_pos.size > 0 else \
                      np.zeros((0, 2), dtype=float)

    # -------- 合并所有点 --------
    all_points = np.concatenate([history_pos, preds_stack, gt_pos], axis=0)
    xs = all_points[:, 0]
    ys = all_points[:, 1]

    xmin_raw = xs.min()
    xmax_raw = xs.max()
    ymin_raw = ys.min()
    ymax_raw = ys.max()

    if not square:
        # 普通矩形 bbox，直接在原 bbox 上加 padding
        xmin = xmin_raw - padding
        xmax = xmax_raw + padding
        ymin = ymin_raw - padding
        ymax = ymax_raw + padding
    else:
        # 正方形模式：以原 bbox 中心为中心，边长取 max(width, height)
        width  = xmax_raw - xmin_raw
        height = ymax_raw - ymin_raw
        side = max(width, height)

        cx = 0.5 * (xmin_raw + xmax_raw)
        cy = 0.5 * (ymin_raw + ymax_raw)

        half = side / 2.0

        xmin = cx - half - padding
        xmax = cx + half + padding
        ymin = cy - half - padding
        ymax = cy + half + padding

    # 返回整数（向下/向上取整）
    xmin = int(np.floor(xmin))
    xmax = int(np.ceil(xmax))
    ymin = int(np.floor(ymin))
    ymax = int(np.ceil(ymax))

    return xmin, xmax, ymin, ymax




def plot_traj_with_heading_and_stim(
    history_pos: np.ndarray,      # (Th, 2)  历史“质心”轨迹
    preds_pos,                    # list[(Tf,2)] 或 (K, Tf, 2) 预测质心轨迹
    gt_pos: np.ndarray,           # (Tf, 2)   真实未来质心
    hist_head: np.ndarray = None, # (Th, 2)   历史 head 关键点
    hist_neck: np.ndarray = None, # (Th, 2)   历史 neck 关键点
    stim_types: np.ndarray = None,# (Th,)     历史每一帧的刺激类型
                                   #   0/None: 无
                                   #   1: 前向刺激
                                   #   2: 左转刺激
                                   #   3: 右转刺激
    title: str = "Trajectory",
    show_axis: bool = True,
):
    """
    可视化：
      - 历史路径：深蓝
      - 预测路径：浅红（多条）
      - 真实路径：深红
      - 头向（neck→head）：历史部分每隔 5 帧画一个小箭头
      - 刺激：用不同颜色的小点标在历史路径上（不画箭头）
    """
    # 颜色定义
    blue_dark  = "#1f4e79"  # 历史路径
    red_light  = "#f5b7b1"  # 预测路径
    red_dark   = "#b30000"  # 真实路径
    green_fwd  = "#2ecc71"  # 前向刺激
    orange_lft = "#e67e22"  # 左转刺激
    purple_rgt = "#9b59b6"  # 右转刺激

    # 统一 preds 形状为 list[np.ndarray]
    if isinstance(preds_pos, np.ndarray):
        if preds_pos.ndim == 3:
            preds_list = [preds_pos[k] for k in range(preds_pos.shape[0])]
        else:
            preds_list = [preds_pos]
    else:
        preds_list = preds_pos

    fig, ax = plt.subplots(figsize=(6.5, 6))

    # 1) 历史路径
    if history_pos is not None and len(history_pos) > 0:
        ax.plot(history_pos[:,0], history_pos[:,1],
                color=blue_dark, linewidth=2.5, label="History")
        ax.scatter(history_pos[0,0], history_pos[0,1],
                   color=blue_dark, s=28, zorder=5)

    # 2) 历史头向箭头（每隔 5 帧 neck→head）
    if hist_head is not None and hist_neck is not None:
        Th = min(len(hist_head), len(hist_neck), len(history_pos))
        step = 5
        for t in range(0, Th, step):
            p_neck = hist_neck[t]
            vec = hist_head[t] - hist_neck[t]  # neck→head
            # 可适当缩放箭头长度
            scale = 0.6
            v = vec * scale
            ax.arrow(p_neck[0], p_neck[1], v[0], v[1],
                     head_width=0.04, head_length=0.08,
                     fc=blue_dark, ec=blue_dark, alpha=0.9, linewidth=1.5)

    # 3) 预测路径（浅红，多条）
    first_pred = True
    for traj in preds_list:
        if traj is None or len(traj) == 0:
            continue
        ax.plot(traj[:,0], traj[:,1],
                color=red_light, linewidth=1.8, alpha=0.9,
                label="Predicted" if first_pred else None)
        first_pred = False
        ax.scatter(traj[-1,0], traj[-1,1],
                   color=red_light, s=16, alpha=0.9, zorder=5)

    # 4) 真实未来路径（深红）
    if gt_pos is not None and len(gt_pos) > 0:
        ax.plot(gt_pos[:,0], gt_pos[:,1],
                color=red_dark, linewidth=2.4, label="Ground Truth")
        ax.scatter(gt_pos[-1,0], gt_pos[-1,1],
                   color=red_dark, s=28, zorder=6)

    # 5) 刺激可视化：在历史轨迹上画彩色点
    if stim_types is not None:
        stim_types = np.asarray(stim_types)
        # 保证长度对齐
        T_stim = min(len(stim_types), len(history_pos))
        # 只给每类画一次 legend
        drawn = {"fwd": False, "left": False, "right": False}
        for t in range(T_stim):
            s = stim_types[t]
            if s in (1, "F", "fwd"):
                ax.scatter(history_pos[t,0], history_pos[t,1],
                           color=green_fwd, s=22, zorder=7,
                           label="Stim Forward" if not drawn["fwd"] else None)
                drawn["fwd"] = True
            elif s in (2, "L", "left"):
                ax.scatter(history_pos[t,0], history_pos[t,1],
                           color=orange_lft, s=22, zorder=7,
                           label="Stim Left" if not drawn["left"] else None)
                drawn["left"] = True
            elif s in (3, "R", "right"):
                ax.scatter(history_pos[t,0], history_pos[t,1],
                           color=purple_rgt, s=22, zorder=7,
                           label="Stim Right" if not drawn["right"] else None)
                drawn["right"] = True
            # 其他值视为“无刺激”，不画

    # 样式
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    if not show_axis:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    ax.legend(frameon=False, loc="best")

    xmin, xmax, ymin, ymax = compute_bounding_box(
        history_pos, preds_pos, gt_pos, padding=5, square=True
    )

    # plt.xlim(0, 640)
    ax.set_xlim(xmin, xmax)
    # plt.ylim(0, 480)
    ax.set_ylim(ymin, ymax)
    plt.tight_layout()
    return fig, ax


def data_preprocess(pred_trajs, hits_trajs, cue_trajs):
    pass

if __name__ == "__main__":
    # ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head', 'neck', 'spine']
    pred_trajs = np.load("./trajs/cond/pred_trajs.npy")
    hist_trajs = np.load("./trajs/cond/hist_trajs.npy")
    cue_trajs = np.load("./trajs/cond/hist_cue_trajs.npy")
    fut_gt_trajs = np.load("./trajs/cond/fut_gt_trajs.npy")

    print("pred_trajs shape = {}".format(pred_trajs.shape))
    print("hist_trajs shape = {}".format(hist_trajs.shape))
    print("cue_trajs shape = {}".format(cue_trajs.shape))
    print("fut_gt_trajs shape = {}".format(fut_gt_trajs.shape))
    # 1 2 3 5 8
    
    for idx in range(30):
        history_pos = hist_trajs[idx, 7, :, 0:2]
        hist_head = hist_trajs[idx, 5, :, 0:2]
        hist_neck = hist_trajs[idx, 7, :, 0:2]

        init = hist_trajs[idx, 7, -1:, 0:2]
        # init = np.expand_dims(init, 0)
        init = init.repeat(30, 0)

        gt_pos = fut_gt_trajs[idx, 7, :, :] + init

        init = np.expand_dims(init, 0)
        init = init.repeat(10, 0)
        preds_pos = pred_trajs[idx, 0:10, 7, :, :] + init
        # preds_pos = init.tolist().extend(pred_trajs[idx, 0:10, 7, :, :])

        stim_types = cue_trajs[idx, :4]    # (..., 4)
        stim_types = stim_types.argmax(axis=-1)

        fig, ax = plot_traj_with_heading_and_stim(
            history_pos, preds_pos, gt_pos,
            hist_head=hist_neck,
            hist_neck=hist_neck,
            stim_types=stim_types
        )

        fig.savefig("./result/cond/visual_{}.png".format(idx), dpi=300)
