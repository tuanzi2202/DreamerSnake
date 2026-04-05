# 🐍 Dreamer Snake v6.0 (PyTorch) — Multi-Model PK Arena

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c)
![Pygame](https://img.shields.io/badge/Pygame-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Dreamer Snake** 是一个基于 PyTorch 构建的高级强化学习（Reinforcement Learning）贪吃蛇项目。不仅包含自主训练的智能体环境，更在 v6.0 版本中引入了极具亮点的 **多模型 PK 竞技场 (Multi-Model PK Arena)** 功能，允许你将不同训练阶段、不同超参数下保存的模型在完全相同的随机种子下进行同台对决（擂台赛）。

---

## ✨ 核心特性 (Features)

- 🧠 **高级深度强化学习架构**：
  - 融合了基于模型的强化学习（Model-Based RL）思想，内置 **编码器 (Encoder)**、**解码器 (Decoder)** 与 **RSSM (Recurrent State Space Model)** 提取潜在状态空间 (Latent Space)。
  - 动作决策采用 **NoisyNet (噪声网络)** 用于更好的探索 (Exploration)。
  - 价值评估采用 **Dueling DQN** 架构，结合 **N-Step 奖励** 与 **PER (优先经验回放, Prioritized Experience Replay)** 加速收敛。
- ⚔️ **双通道 PK 竞技场 (PK Arena)**：扫描当前目录下的不同 Checkpoint，选择两个模型，在同种子的镜像世界中进行双屏公平对决，内置胜场积分板。
- 📊 **实时可视化数据面板**：训练模式下内置 Pygame 实时渲染的迷你折线图（Score、Reward、Loss）与动作 Q 值分布柱状图。
- 🎮 **人机无缝切换**：支持随时按下 `M` 键接管蛇的控制权（Manual Mode），也可以随时暂停并调速。
- 💾 **智能断点续训**：全自动的版本识别与权重迁移（Version Migration），向下兼容旧版本（v3.1, v4.0, v5.0）的权重和回放内存（Replay Memory）。

---

## 🛠️ 安装与运行 (Installation & Usage)

### 1. 环境依赖
请确保你的机器上安装了 Python 3.8+，并执行以下命令安装依赖：
```bash
pip install torch pygame numpy
```
*(注：如果你的设备带有 NVIDIA 显卡，请前往 [PyTorch 官网](https://pytorch.org/) 安装对应 CUDA 版本的 torch 以获取显著的训练加速，代码已内置自动设备检测)。*

### 2. 启动训练模式 (Training Mode)
直接运行主程序即可进入训练模式。若本地无权重，系统将从零开始初始化：
```bash
python main.py
```

### 3. 启动 PK 竞技场模式 (PK Arena)
1. 至少保留两次不同阶段的 Checkpoint 备份（或者将保存的 `dreamer_snake_v6_ckpt` 复制并重命名以作留存）。
2. 启动主程序后，按下键盘上的 `P` 键即可切入 PK 选择界面。
3. 使用数字键或上下箭头选择你要对战的两个模型，按下 `Enter` 键开始对决！

---

## ⌨️ 快捷键指南 (Controls & Shortcuts)

**训练模式下：**
- `[Space]` : 暂停/继续训练
- `[↑] / [↓]`: 增加/减少游戏运行速度 (Speed Multiplier)
- `[M]` : 切换手动/AI自动模式 (Manual / AI)
- `[←] / [→]`: 在手动模式下控制贪吃蛇转向
- `[H]` : 开启/关闭足迹热力图 (Heatmap)
- `[R]` : 重置当前热力图数据
- `[P]` : 保存当前进度并进入 **PK 竞技场**
- `[S]` : 强制手动保存 Checkpoint
- `[F12]` : 快速游戏截图 (保存在 `screenshots/` 目录下)
- `[Tab]` : 显示/隐藏帮助面板

**PK 竞技场模式下：**
- `[1-9, A-Z]` / `[↑] [↓]` : 选中对应的模型
- `[Enter]` : 确认选择，开始擂台对决
- `[Esc]` : 返回训练模式 / 中止当前对决
- `[N]` : 比赛结算后，开启下一轮（Next Round）

---

## 📁 目录及存档结构说明 (Directory Structure)

项目在运行期间会自动生成以下文件体系：

```text
.
├── main.py                     # 项目主源码文件
├── screenshots/                # 按下 F12 保存的截图目录
└── dreamer_snake_v6_ckpt/      # v6 版本默认存档目录
    ├── model.pth               # 当前最新模型权重与优化器状态
    ├── best_model.pth          # 历史最高 Reward 的模型权重
    ├── memory.pkl              # 序列化保存的 PER 经验回放池
    ├── stats.pkl               # 训练曲线的历史统计数据 (Loss/Score等)
    └── meta.json               # 可读的元数据 (包含版本、Episode、得分记录)
```

## 📜 许可证 (License)
本项目采用 [MIT License](LICENSE) 开源协议。欢迎提交 Issue 与 Pull Request 共同完善这一框架！