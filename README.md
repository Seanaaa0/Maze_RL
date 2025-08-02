# 🧭 Maze_RL_GPT

This project explores **Reinforcement Learning (RL)** in a series of **custom maze environments**, progressively increasing in complexity. It integrates GPT data generation and inference logic, but the actual fine-tuning process is now migrated to a new project: [GPT-CoT](https://github.com/Seanaaa0/GPT-CoT)

---

## 📦 Environment Overview

All environments are custom-built in Python and partially follow the `gymnasium` interface.

| Maze ID  | Key Features |
|----------|--------------|
| `maze1`  | Fully observable, deterministic grid. Simple greedy path-finding. |
| `maze2`  | Includes partial observability (1x1 view) and non-deterministic movement. |
| `maze3`  | DFS/Prim-based maze generation with complex branching. |
| `maze4`  | POMDP environment with `Growing Tree` maze, designed for map exploration. |
| `maze5`  | Adds **traps** to the environment. Exploration must avoid trap tiles. |
| `maze6`  | Multi-goal navigation with **TSP-style shortest path planning** across several goals. |

All environments support:
- Custom seed & size settings
- Saving full exploration trajectories
- Ground truth map extraction for comparison

---

## 🎮 Key Modules

- `env/` & `env_partial/`: Maze environments (fully or partially observable)
- `train/`: RL-inspired exploration or trajectory logging
- `run/`: Path planning / TSP execution with visualization
- `outputs/`: Saved trajectories, maps, visuals
- `visual/`: Standalone tools for GT visualization, exploration rendering

---

## 🤖 GPT Integration

The Maze environments are used to generate reasoning datasets for GPT fine-tuning.

➡️ **All model fine-tuning tasks are now maintained under a new repo:**  
🔗 [GPT-CoT (Chain-of-Thought Reasoning Fine-Tuning)](https://github.com/Seanaaa0/GPT-CoT)

---

## 📊 Example Outputs

- ✅ Visualization of `run_tsp_theta_6.py`: Shows shortest goal path with TSP search
- ✅ Trap avoidance coloring in maze5
- ✅ Demo video and annotated PDF (`maze.pdf`)

---

## 📁 Demo Materials

📄 `maze.pdf`: Slide summary of motivation, environment, training logic, and output examples  
🎬 `run_demo.mp4`: Real-time TSP navigation video  
📊 `v3.py`: Visualize `gt_maze6_multi_SEED*.npy` ground truth maps

---

## 📌 Git Tips

To avoid tracking large output files:

```
# .gitignore
outputs/
experiments-linux/
```

If large files were already committed, use `git filter-repo`:

```bash
git filter-repo --force --path outputs --path experiments-linux --invert-paths
```

---

## ✨ Future Work

- Integrate Decision Transformer
- Fine-tune with policy + trajectory pairs
- Add language-based goal commands
- Combine LLM inference with real-time RL

---

## 🔗 GitHub

This project is hosted at: [Maze_RL_GPT](https://github.com/Seanaaa0/Maze_RL_GPT)  
GPT model training: [GPT-CoT](https://github.com/Seanaaa0/GPT-CoT)