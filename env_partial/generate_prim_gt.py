from maze1_prim_pomdp import Maze1PrimPOMDPEnv

# 初始化環境（會自動產生迷宮並儲存）
env = Maze1PrimPOMDPEnv()

# 顯示儲存成功的提示（實際儲存由 env 內部 _save_maze 完成）
print("🎉 Prim 迷宮 Ground Truth 已成功產生並儲存。")
