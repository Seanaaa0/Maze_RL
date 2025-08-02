import numpy as np
import matplotlib.pyplot as plt

# 讀取 ground truth
gt_path = "C:/Users/seana/maze/outputs/mem_trap/gt_maze6_multi_103x103_SEED88.npy"
data = np.load(gt_path, allow_pickle=True).item()

maze = data["wall_map"]
start = data["start_pos"]
goals = data["goal_list"]
traps = set(map(tuple, data["trap_list"]))

# 繪圖
H, W = maze.shape
img = np.zeros((H, W, 3))
for i in range(H):
    for j in range(W):
        if maze[i, j] == 1:
            img[i, j] = (0.0, 0.0, 0.0)     # 牆
        else:
            img[i, j] = (1.0, 1.0, 1.0)     # 路

for tx, ty in traps:
    img[tx, ty] = (1.0, 1.0, 0.0)  # 陷阱：黃

for gx, gy in goals:
    img[gx, gy] = (1.0, 0.0, 0.0)  # 目標：紅

img[start[0], start[1]] = (0.0, 1.0, 0.0)   # 起點：綠

plt.figure(figsize=(8, 8))
plt.imshow(img, interpolation='nearest')
plt.title(f"Ground Truth Maze {H}x{W} with Goals & Traps")
plt.xticks([]), plt.yticks([])
plt.show()
