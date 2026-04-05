# Dreamer Snake v6.0 (PyTorch) — Multi-Model PK Arena
# ====================================================
# pip install torch pygame numpy

import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import os
import pickle
import copy
import time
import json
import glob

# =========================
# Version & Config
# =========================
VERSION = "6.0"
ARCH_SIGNATURE = "dreamer_v6_pk"

GAME_W, GAME_H = 400, 400
GRID = 20
COLS = GAME_W // GRID
ROWS = GAME_H // GRID

TRAIN_PANEL_W = 240
TRAIN_BOTTOM_H = 75
TRAIN_W = GAME_W + TRAIN_PANEL_W
TRAIN_H = GAME_H + TRAIN_BOTTOM_H

PK_GAME_SIZE = 300
PK_GAP = 10
PK_PANEL_W = 200
PK_BOTTOM_H = 90
PK_TOP_H = 30
PK_W = PK_GAME_SIZE * 2 + PK_GAP + PK_PANEL_W + 20
PK_H = PK_TOP_H + PK_GAME_SIZE + PK_BOTTOM_H

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 16
LATENT_DIM = 48
BATCH_SIZE = 128
MEMORY_SIZE = 30000
GAMMA = 0.99
TAU = 0.005
N_STEP = 3
BASE_FPS = 12
MAX_FPS = 60
SAVE_INTERVAL = 10
MAX_EPISODES = 999999
WARMUP_EPS = 50

CKPT_DIR = "dreamer_snake_v6_ckpt"
CKPT_MODEL = os.path.join(CKPT_DIR, "model.pth")
CKPT_BEST = os.path.join(CKPT_DIR, "best_model.pth")
CKPT_MEMORY = os.path.join(CKPT_DIR, "memory.pkl")
CKPT_STATS = os.path.join(CKPT_DIR, "stats.pkl")
CKPT_META = os.path.join(CKPT_DIR, "meta.json")

ALL_CKPT_PATTERNS = [
    "dreamer_snake_v*_ckpt",
    "dreamer_snake_ckpt",
]

# =========================
# Colors
# =========================
BG_COLOR = (10, 10, 20)
GRID_COLOR = (20, 20, 35)
PANEL_BG = (18, 18, 32)
BOTTOM_BG = (25, 25, 42)
TEXT_COLOR = (185, 185, 215)
TEXT_DIM = (105, 105, 135)
HIGHLIGHT = (255, 210, 50)
GOOD_COLOR = (60, 255, 140)
BAD_COLOR = (255, 70, 70)
WARN_COLOR = (255, 180, 40)
CHART_BG = (13, 13, 26)
CHART_SCORE = (80, 200, 255)
CHART_REWARD = (255, 175, 55)
CHART_LOSS = (255, 85, 125)
FOOD_COLOR = (255, 50, 70)
FOOD_GLOW = (255, 85, 85)

SNAKE_THEMES = [
    {"head": (0, 255, 160), "body": (0, 210, 120),
     "tail": (0, 130, 70), "name": "Green", "accent": (0, 255, 160)},
    {"head": (80, 160, 255), "body": (60, 120, 220),
     "tail": (30, 70, 160), "name": "Blue", "accent": (80, 160, 255)},
    {"head": (255, 160, 40), "body": (220, 120, 30),
     "tail": (160, 80, 20), "name": "Orange", "accent": (255, 160, 40)},
    {"head": (220, 60, 255), "body": (180, 40, 210),
     "tail": (120, 20, 150), "name": "Purple", "accent": (220, 60, 255)},
]

QBAR_COLORS = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
HEATMAP_COLORS = [
    (0, 0, 40), (0, 0, 80), (0, 40, 120), (0, 80, 160),
    (0, 160, 160), (80, 200, 80), (200, 200, 0),
    (255, 160, 0), (255, 80, 0), (255, 0, 0),
]


# =========================
# Particles
# =========================
class Particle:
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'color', 'size']

    def __init__(self, x, y, color, speed_range=(1, 4)):
        self.x, self.y = x, y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(*speed_range)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.3, 0.8)
        self.max_life = self.life
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 2.5 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        a = max(0, self.life / self.max_life)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a)) for v in self.color)
            pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)


class TrailParticle:
    __slots__ = ['x', 'y', 'life', 'color', 'size']

    def __init__(self, x, y, color):
        self.x = x + random.uniform(-2, 2)
        self.y = y + random.uniform(-2, 2)
        self.life = 0.35
        self.color = color
        self.size = random.uniform(1, 3)

    def update(self, dt):
        self.life -= dt
        return self.life > 0

    def draw(self, surf):
        a = max(0, self.life / 0.35)
        r = int(self.size * a)
        if r > 0:
            c = tuple(min(255, int(v * a * 0.5)) for v in self.color)
            pygame.draw.circle(surf, c, (int(self.x), int(self.y)), r)


# =========================
# PER
# =========================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, transition):
        mx = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(mx)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = mx
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        p = np.array(self.priorities, dtype=np.float64)
        probs = p ** self.alpha
        probs /= probs.sum()
        n = min(batch_size, len(self.buffer))
        idx = np.random.choice(len(self.buffer), n, p=probs, replace=False)
        samples = [self.buffer[i] for i in idx]
        w = (len(self.buffer) * probs[idx]) ** (-beta)
        w /= w.max()
        return samples, idx, torch.tensor(w, dtype=torch.float32, device=DEVICE)

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = abs(td) + 1e-6

    def __len__(self):
        return len(self.buffer)

    def get_data(self):
        return self.buffer[:len(self.buffer)]


# =========================
# N-Step
# =========================
class NStepBuffer:
    def __init__(self, n=N_STEP, gamma=GAMMA):
        self.n, self.gamma = n, gamma
        self.buffer = deque(maxlen=n)

    def push(self, t):
        self.buffer.append(t)

    def get(self):
        if len(self.buffer) < self.n:
            return None
        s0, a0 = self.buffer[0][0], self.buffer[0][1]
        r = sum(self.gamma ** i * self.buffer[i][2] for i in range(self.n))
        return (s0, a0, r, self.buffer[-1][3], self.buffer[-1][4])

    def flush(self):
        results = []
        while self.buffer:
            s0, a0 = self.buffer[0][0], self.buffer[0][1]
            r = sum(self.gamma ** i * self.buffer[i][2]
                    for i in range(len(self.buffer)))
            results.append((s0, a0, r, self.buffer[-1][3], self.buffer[-1][4]))
            self.buffer.popleft()
        return results

    def reset(self):
        self.buffer.clear()


# =========================
# MiniChart
# =========================
class MiniChart:
    def __init__(self, x, y, w, h, title, color, max_pts=200):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.title, self.color = title, color
        self.data = deque(maxlen=max_pts)

    def add(self, v):
        self.data.append(v)

    def draw(self, scr, font):
        pygame.draw.rect(scr, CHART_BG, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(scr, (40, 40, 60),
                         (self.x, self.y, self.w, self.h), 1)
        scr.blit(font.render(self.title, True, TEXT_DIM),
                 (self.x + 4, self.y + 2))
        if len(self.data) < 2:
            return
        dl = list(self.data)
        mn, mx = min(dl), max(dl)
        rng = mx - mn if mx != mn else 1.0
        cy, ch = self.y + 16, self.h - 20
        scr.blit(font.render(f"{dl[-1]:.1f}", True, self.color),
                 (self.x + self.w - 50, self.y + 2))
        win = min(50, len(dl))
        avg = []
        for i in range(len(dl)):
            s = max(0, i - win + 1)
            avg.append(sum(dl[s:i + 1]) / (i - s + 1))
        pts, apts = [], []
        n = len(dl)
        for i in range(n):
            px = self.x + 2 + (self.w - 4) * i / max(n - 1, 1)
            py1 = cy + ch - ch * (dl[i] - mn) / rng
            py1 = max(cy, min(cy + ch, py1))
            pts.append((px, py1))
            py2 = cy + ch - ch * (avg[i] - mn) / rng
            py2 = max(cy, min(cy + ch, py2))
            apts.append((px, py2))
        if len(pts) >= 2:
            pygame.draw.lines(scr, tuple(v // 3 for v in self.color),
                              False, pts, 1)
        if len(apts) >= 2:
            pygame.draw.lines(scr, self.color, False, apts, 2)


# =========================
# QValueBar
# =========================
class QValueBar:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.values = [0.33, 0.33, 0.33]
        self.labels = ["Fwd", "L", "R"]

    def set_values(self, p):
        self.values = [float(v) for v in p]

    def draw(self, scr, font):
        pygame.draw.rect(scr, CHART_BG, (self.x, self.y, self.w, self.h))
        pygame.draw.rect(scr, (40, 40, 60),
                         (self.x, self.y, self.w, self.h), 1)
        scr.blit(font.render("Actions", True, TEXT_DIM),
                 (self.x + 4, self.y + 2))
        by = self.y + 16
        bh = self.h - 28
        bw = (self.w - 20) // 3
        mx = max(self.values) if max(self.values) > 0 else 1
        for i in range(3):
            bx = self.x + 4 + i * (bw + 4)
            ratio = self.values[i] / mx
            fh = int(bh * ratio * 0.85)
            pygame.draw.rect(scr, (30, 30, 50), (bx, by, bw, bh))
            if fh > 0:
                pygame.draw.rect(scr, QBAR_COLORS[i],
                                 (bx, by + bh - fh, bw, fh))
            scr.blit(font.render(f"{self.values[i]:.2f}", True, QBAR_COLORS[i]), (bx, by + bh + 1))


# =========================
# NoisyLinear
# =========================
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.weight_mu = nn.Parameter(torch.empty(out_f, in_f))
        self.weight_sigma = nn.Parameter(torch.empty(out_f, in_f))
        self.register_buffer('weight_epsilon', torch.empty(out_f, in_f))
        self.bias_mu = nn.Parameter(torch.empty(out_f))
        self.bias_sigma = nn.Parameter(torch.empty(out_f))
        self.register_buffer('bias_epsilon', torch.empty(out_f))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        r = 1 / math.sqrt(self.in_f)
        self.weight_mu.data.uniform_(-r, r)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_f))
        self.bias_mu.data.uniform_(-r, r)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_f))

    def _sn(self, sz):
        x = torch.randn(sz, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        ei = self._sn(self.in_f)
        eo = self._sn(self.out_f)
        self.weight_epsilon.copy_(eo.ger(ei))
        self.bias_epsilon.copy_(eo)

    def forward(self, x):
        if self.training:
            return F.linear(
                x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(x, self.weight_mu, self.bias_mu)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), Swish(), nn.Linear(dim, dim))
        self.act = Swish()

    def forward(self, x):
        return self.act(x + self.net(x))


# =========================
# SnakeWorld (纯逻辑，可种子复现)
# =========================
class SnakeWorld:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.snake = []
        self.dir = (GRID, 0)
        self.food = (0, 0)
        self.steps_no_food = 0
        self.total_steps = 0
        self.ate_count = 0
        self.alive = True
        self.total_reward = 0.0
        self.reset()

    def reset(self):
        cx = (COLS // 2) * GRID
        cy = (ROWS // 2) * GRID
        self.snake = [(cx, cy), (cx - GRID, cy), (cx - 2 * GRID, cy)]
        self.dir = (GRID, 0)
        self.food = self._spawn()
        self.steps_no_food = 0
        self.total_steps = 0
        self.ate_count = 0
        self.alive = True
        self.total_reward = 0.0
        return self.state()

    def _spawn(self):
        ss = set(self.snake)
        cands = [(x, y) for x in range(0, GAME_W, GRID) for y in range(0, GAME_H, GRID) if (x, y) not in ss]
        if not cands:
            return (0, 0)
        return self.rng.choice(cands)

    def _dist(self, pos):
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])

    def step(self, action):
        if not self.alive:
            return self.state(), 0.0, True
        dirs = [(GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID)]
        idx = dirs.index(self.dir)
        if action == 1:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        self.dir = dirs[idx]
        old_d = self._dist(self.snake[0])
        head = (self.snake[0][0] + self.dir[0],
                self.snake[0][1] + self.dir[1])
        self.total_steps += 1
        if (head[0] < 0 or head[0] >= GAME_W or
                head[1] < 0 or head[1] >= GAME_H or
                head in self.snake):
            self.alive = False
            self.total_reward += -10.0
            return self.state(), -10.0, True
        self.snake.insert(0, head)
        self.steps_no_food += 1
        ate = False
        if head == self.food:
            reward = 10.0 + len(self.snake) * 0.5
            self.ate_count += 1
            self.food = self._spawn()
            self.steps_no_food = 0
            ate = True
        else:
            self.snake.pop()
            nd = self._dist(head)
            reward = 0.1 if nd < old_d else -0.15
            
        timeout = 80 + len(self.snake) * 25
        if self.steps_no_food > timeout:
            self.alive = False
            self.total_reward += -5.0
            return self.state(), -5.0, True
        self.total_reward += reward
        return self.state(), reward, False

    def state(self):
        h = self.snake[0]
        dirs = [(GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID)]
        idx = dirs.index(self.dir)
        ss = set(self.snake)

        def dng(p):
            return float(p[0] < 0 or p[0] >= GAME_W or
                p[1] < 0 or p[1] >= GAME_H or p in ss)

        def dw(d):
            dx, dy = d
            c = 0
            px, py = h
            while True:
                px += dx
                py += dy
                if px < 0 or px >= GAME_W or py < 0 or py >= GAME_H:
                    break
                c += 1
            return c / max(COLS, ROWS)

        da = dirs[idx]
        dl = dirs[(idx - 1) % 4]
        dr = dirs[(idx + 1) % 4]
        ah = (h[0] + da[0], h[1] + da[1])
        lh = (h[0] + dl[0], h[1] + dl[1])
        rh = (h[0] + dr[0], h[1] + dr[1])
        a2 = (h[0] + da[0] * 2, h[1] + da[1] * 2)
        fdx = self.food[0] - h[0]
        fdy = self.food[1] - h[1]
        ang = math.atan2(fdy, fdx)
        fd = (abs(fdx) + abs(fdy)) / (GAME_W + GAME_H)
        return np.array([
            fdx / GAME_W, fdy / GAME_H,
            self.dir[0] / GRID, self.dir[1] / GRID,
            dng(ah), dng(lh), dng(rh),
            dw(da), dw(dl), dw(dr),
            len(self.snake) / (COLS * ROWS),
            self.steps_no_food / 200.0,
            math.sin(ang), math.cos(ang),
            dng(a2), fd,], dtype=np.float32)

    @property
    def score(self):
        return len(self.snake) - 3


# =========================
# Neural Network Modules
# =========================
class Encoder(nn.Module):
    def __init__(self, sd=STATE_DIM, ld=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sd, 128), Swish(), nn.LayerNorm(128),
            ResBlock(128), nn.Linear(128, 64), Swish(),
            nn.Linear(64, ld * 2))

    def forward(self, x):
        h = self.net(x)
        m, ls = torch.chunk(h, 2, dim=-1)
        s = torch.exp(ls.clamp(-5, 2))
        return m + s * torch.randn_like(s), m, s


class Decoder(nn.Module):
    def __init__(self, sd=STATE_DIM, ld=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ld, 64), Swish(), nn.Linear(64, 128), Swish(),
            ResBlock(128), nn.Linear(128, sd))

    def forward(self, z):
        return self.net(z)


class RSSM(nn.Module):
    def __init__(self, ld=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ld + 1, 128), Swish(), nn.LayerNorm(128),
            ResBlock(128), nn.Linear(128, 64), Swish(),
            nn.Linear(64, ld))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class RewardModel(nn.Module):
    def __init__(self, ld=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ld, 64), Swish(), nn.Linear(64, 32), Swish(),
            nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z)


class DuelingValueModel(nn.Module):
    def __init__(self, ld=LATENT_DIM):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(ld, 64), Swish(), ResBlock(64))
        self.vh = nn.Sequential(nn.Linear(64, 32), Swish(), nn.Linear(32, 1))
        self.ah = nn.Sequential(nn.Linear(64, 32), Swish(), nn.Linear(32, 1))

    def forward(self, z):
        h = self.shared(z)
        v = self.vh(h)
        a = self.ah(h)
        return v + (a - a.mean())


class NoisyActor(nn.Module):
    def __init__(self, ld=LATENT_DIM):
        super().__init__()
        self.fc1 = nn.Linear(ld, 64)
        self.res = ResBlock(64)
        self.n1 = NoisyLinear(64, 32)
        self.n2 = NoisyLinear(32, 3)
        self.act = Swish()

    def forward(self, z):
        h = self.act(self.fc1(z))
        h = self.res(h)
        h = self.act(self.n1(h))
        return torch.softmax(self.n2(h), dim=-1)

    def reset_noise(self):
        self.n1.reset_noise()
        self.n2.reset_noise()


# =========================
# Migration
# =========================
MODEL_KEYS = ["encoder", "decoder", "rssm",
              "reward_model", "value_model", "actor"]


def migrate_weights(model, old_sd, name=""):
    ns = model.state_dict()
    mc, sc = 0, 0
    for k in ns:
        if k in old_sd and old_sd[k].shape == ns[k].shape:
            ns[k] = old_sd[k]
            mc += 1
        else:
            sc += 1
    model.load_state_dict(ns)
    return mc, sc


def migrate_memory(old_data, old_dim, new_dim):
    if old_dim >= new_dim:
        return old_data
    pad = new_dim - old_dim
    out = []
    for s, a, r, s2, d in old_data:
        if len(s) == old_dim:
            s = np.concatenate([s, np.zeros(pad, dtype=np.float32)])
        if len(s2) == old_dim:
            s2 = np.concatenate([s2, np.zeros(pad, dtype=np.float32)])
        out.append((s, a, r, s2, d))
    return out


def detect_version(ckpt):
    if "version" in ckpt:
        return ckpt["version"]
    enc = ckpt.get("encoder", {})
    for k in enc:
        if "0.weight" in k or "net.0.weight" in k:
            sh = enc[k].shape
            if sh[-1] == 12:
                return "3.1"
            elif sh[-1] == 16:
                return "4.0"
    return "unknown"


# =========================
# Model Scanner (扫描所有可用模型)
# =========================
def scan_all_models():
    """扫描当前目录下所有checkpoint目录"""
    found = []
    seen = set()
    #搜索模式
    patterns = [
        "dreamer_snake*ckpt",
        "dreamer_snake*ckpt/",
    ]
    dirs_to_check = set()
    for pat in patterns:
        for p in glob.glob(pat):
            if os.path.isdir(p):
                dirs_to_check.add(p)

    # 也检查当前目录
    for item in os.listdir("."):
        if os.path.isdir(item) and "ckpt" in item.lower():
            dirs_to_check.add(item)

    for d in sorted(dirs_to_check):
        model_path = os.path.join(d, "model.pth")
        best_path = os.path.join(d, "best_model.pth")
        meta_path = os.path.join(d, "meta.json")

        for path, suffix in [(best_path, " (best)"), (model_path, "")]:
            if os.path.exists(path) and path not in seen:
                seen.add(path)
                info = {"path": path, "dir": d, "suffix": suffix}
                # 读取meta
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path) as f:
                            meta = json.load(f)
                        info["version"] = meta.get("version", "?")
                        info["episode"] = meta.get("episode", "?")
                        info["best_reward"] = meta.get("best_reward", "?")
                    except Exception:
                        info["version"] = "?"
                        info["episode"] = "?"
                        info["best_reward"] = "?"
                else:
                    # 从pth中读取
                    try:
                        ckpt = torch.load(path, map_location="cpu", weights_only=False)
                        info["version"] = detect_version(ckpt)
                        info["episode"] = ckpt.get("episode", "?")
                        info["best_reward"] = ckpt.get("best_reward", "?")
                    except Exception:
                        info["version"] = "?"
                        info["episode"] = "?"
                        info["best_reward"] = "?"

                name = f"{d}{suffix}"
                info["name"] = name
                found.append(info)

    return found


# =========================
# ModelSlot (加载用于PK推理)
# =========================
class ModelSlot:
    def __init__(self, info):
        self.info = info
        self.name = info["name"]
        self.path = info["path"]
        self.encoder = Encoder().to(DEVICE)
        self.actor = NoisyActor().to(DEVICE)
        self._load()
        self.encoder.eval()
        self.actor.eval()

    def _load(self):
        ckpt = torch.load(self.path, map_location=DEVICE, weights_only=False)
        enc_sd = ckpt.get("encoder", {})
        act_sd = ckpt.get("actor", {})
        mc1, sc1 = migrate_weights(self.encoder, enc_sd, "encoder")
        mc2, sc2 = migrate_weights(self.actor, act_sd, "actor")
        print(f"  📦 PK loaded: {self.name} "
              f"(enc:{mc1}ok/{sc1}new, act:{mc2}ok/{sc2}new)")
              
    def get_action(self, obs):
        t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            z, _, _ = self.encoder(t)
            probs = self.actor(z)
        return torch.argmax(probs, dim=-1).item(), probs.squeeze().cpu().tolist()


# =========================
# Game Renderer
# =========================
def render_world(surf, world, theme, particles, trails,
                 food_pulse, font, show_heatmap=False, heatmap=None):
    """将SnakeWorld渲染到指定Surface上"""
    w, h = surf.get_size()
    sx = w / GAME_W
    sy = h / GAME_H
    g = int(GRID * sx)

    surf.fill(BG_COLOR)

    # 网格
    gc = GRID_COLOR
    for x in range(0, int(w), g):
        pygame.draw.line(surf, gc, (x, 0), (x, h))
    for y in range(0, int(h), g):
        pygame.draw.line(surf, gc, (0, y), (w, y))

    # 热力图
    if show_heatmap and heatmap is not None and heatmap.max() > 0:
        hn = heatmap / heatmap.max()
        hs = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
        for gy2 in range(ROWS):
            for gx2 in range(COLS):
                v = hn[gy2, gx2]
                if v > 0.01:
                    ci = min(int(v * 9), 9)
                    c = HEATMAP_COLORS[ci]
                    al = min(150, int(v * 180))
                    pygame.draw.rect(hs, (*c, al), (int(gx2 * g), int(gy2 * g), g, g))
        surf.blit(hs, (0, 0))

    # 食物
    fx = int(world.food[0] * sx)
    fy = int(world.food[1] * sy)
    gr = int(g * 0.7 + math.sin(food_pulse) * 3 * sx)
    if gr > 2:
        gs = pygame.Surface((gr * 4, gr * 4), pygame.SRCALPHA)
        pygame.draw.circle(gs, (*FOOD_GLOW, 35), (gr * 2, gr * 2), gr * 2)
        surf.blit(gs, (fx + g // 2 - gr * 2, fy + g // 2 - gr * 2))
    pygame.draw.rect(surf, FOOD_COLOR,
                     (fx + 2, fy + 2, g - 4, g - 4), border_radius=max(1, g // 4))

    # 拖影
    dt = 1.0 / 30
    i = 0
    while i < len(trails):
        if trails[i].update(dt):
            p = trails[i]
            a = max(0, p.life / 0.35)
            r = int(p.size * a * sx)
            if r > 0:
                c = tuple(min(255, int(v * a * 0.5)) for v in p.color)
                pygame.draw.circle(surf, c, (int(p.x * sx), int(p.y * sy)), max(1, r))
            i += 1
        else:
            trails.pop(i)

    # 蛇身
    n = len(world.snake)
    hc = theme["head"]
    tc = theme["tail"]
    for i, s in enumerate(world.snake):
        t = i / max(n - 1, 1)
        cr = int(hc[0] * (1 - t) + tc[0] * t)
        cg = int(hc[1] * (1 - t) + tc[1] * t)
        cb = int(hc[2] * (1 - t) + tc[2] * t)
        pad = 1 if i == 0 else 2
        px = int(s[0] * sx) + pad
        py = int(s[1] * sy) + pad
        pw = g - pad * 2
        br = max(1, g // 4) if i == 0 else max(1, g // 6)
        pygame.draw.rect(surf, (cr, cg, cb), (px, py, pw, pw), border_radius=br)
        if i == 0:
            ex = int(s[0] * sx) + g // 2 + int(world.dir[0] * sx) // 4
            ey = int(s[1] * sy) + g // 2 + int(world.dir[1] * sy) // 4
            er = max(1, int(3 * sx))
            pygame.draw.circle(surf, (255, 255, 255), (ex - max(1, int(2 * sx)), ey), er)
            pygame.draw.circle(surf, (255, 255, 255), (ex + max(1, int(2 * sx)), ey), er)
            pygame.draw.circle(surf, (0, 0, 0), (ex - max(1, int(2 * sx)), ey), max(1, er // 2))
            pygame.draw.circle(surf, (0, 0, 0), (ex + max(1, int(2 * sx)), ey), max(1, er // 2))

    # 粒子
    i = 0
    while i < len(particles):
        if particles[i].update(dt):
            p = particles[i]
            a = max(0, p.life / p.max_life)
            r = int(p.size * a * sx)
            if r > 0:
                c = tuple(min(255, int(v * a)) for v in p.color)
                pygame.draw.circle(surf, c,
                                   (int(p.x * sx), int(p.y * sy)), max(1, r))
            i += 1
        else:
            particles.pop(i)

    # 死亡标记
    if not world.alive:
        ds = pygame.Surface((int(w), int(h)), pygame.SRCALPHA)
        ds.fill((255, 0, 0, 30))
        surf.blit(ds, (0, 0))
        if font:
            txt = font.render("DEAD", True, BAD_COLOR)
            surf.blit(txt, (w // 2 - txt.get_width() // 2, h // 2 - txt.get_height() // 2))

    # 边框
    pygame.draw.rect(surf, (50, 50, 80), (0, 0, int(w), int(h)), 2)


# =========================
# PK Arena
# =========================
class PKArena:
    def __init__(self, screen, clock, fonts):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.models = []
        self.wins = {}
        self.round_num = 0
        self.food_pulse = 0.0
        self.speed = 3

    def run_selection(self, available_models):
        """模型选择界面"""
        selected = []
        scroll = 0

        while True:
            self.screen.fill(BG_COLOR)
            # 标题
            title = self.fonts["lg"].render(
                "PK ARENA - Select 2 Models", True, HIGHLIGHT)
            self.screen.blit(title, (20, 15))

            # 模型列表
            y = 55
            per_page = 12
            visible = available_models[scroll:scroll + per_page]
            for i, info in enumerate(visible):
                idx = scroll + i
                is_sel = idx in selected
                num = str(idx + 1) if idx < 9 else chr(ord('a') + idx - 9)

                bg = (40, 60, 40) if is_sel else (25, 25, 40)
                pygame.draw.rect(self.screen, bg,
                                 (15, y - 2, self.screen.get_width() - 30, 24),
                                 border_radius=4)

                mark = "✓ " if is_sel else "  "
                ver_str = f"v{info.get('version', '?')}"
                ep_str = f"ep{info.get('episode', '?')}"
                br = info.get('best_reward', '?')
                br_str = f"R={br:.1f}" if isinstance(br, (int, float)) else f"R={br}"

                txt = f"[{num}] {mark}{info['name']}"
                col = GOOD_COLOR if is_sel else TEXT_COLOR
                self.screen.blit(
                    self.fonts["md"].render(txt, True, col), (20, y))
                detail = f"  {ver_str} | {ep_str} | {br_str}"
                self.screen.blit(
                    self.fonts["sm"].render(detail, True, TEXT_DIM),
                    (350, y + 2))
                y += 26

            # 提示
            y += 15
            self.screen.blit(
                self.fonts["md"].render(
                    f"Selected: {len(selected)}/2|  "
                    f"[Enter] Start|  [Esc] Back",
                    True, TEXT_DIM), (20, y))
            if len(selected) == 2:
                self.screen.blit(
                    self.fonts["md"].render(
                        ">>> Press ENTER to begin PK <<<",
                        True, HIGHLIGHT), (20, y + 22))

            # 胜场记录
            if self.wins:
                y += 50
                self.screen.blit(
                    self.fonts["md"].render(
                        "--- Leaderboard ---", True, HIGHLIGHT), (20, y))
                y += 20
                for name, w in sorted(self.wins.items(),
                                       key=lambda x: -x[1]):
                    self.screen.blit(
                        self.fonts["md"].render(
                            f"{name}: {w} wins", True, GOOD_COLOR),
                        (20, y))
                    y += 18

            pygame.display.flip()
            self.clock.tick(30)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return None
                    if ev.key == pygame.K_RETURN and len(selected) == 2:
                        return [available_models[i] for i in selected]
                    # 数字/字母选择
                    idx_pressed = -1
                    if pygame.K_1 <= ev.key <= pygame.K_9:
                        idx_pressed = ev.key - pygame.K_1
                    elif pygame.K_a <= ev.key <= pygame.K_z:
                        idx_pressed = ev.key - pygame.K_a + 9
                    if 0 <= idx_pressed < len(available_models):
                        if idx_pressed in selected:
                            selected.remove(idx_pressed)
                        elif len(selected) < 2:
                            selected.append(idx_pressed)
                    # 翻页
                    if ev.key == pygame.K_UP:
                        scroll = max(0, scroll - 1)
                    if ev.key == pygame.K_DOWN:
                        scroll = min(max(0, len(available_models) - per_page),
                                     scroll + 1)

    def run_match(self, slot_a, slot_b):
        """执行一场PK"""
        seed = random.randint(0, 2**31)
        self.round_num += 1
        world_a = SnakeWorld(seed=seed)
        world_b = SnakeWorld(seed=seed)
        particles_a, particles_b = [], []
        trails_a, trails_b = [], []

        obs_a = world_a.reset()
        obs_b = world_b.reset()

        surf_a = pygame.Surface((PK_GAME_SIZE, PK_GAME_SIZE))
        surf_b = pygame.Surface((PK_GAME_SIZE, PK_GAME_SIZE))

        # 倒计时
        for count in [3, 2, 1]:
            self.screen.fill(BG_COLOR)
            render_world(surf_a, world_a, SNAKE_THEMES[0],
                         [], [], 0, None)
            render_world(surf_b, world_b, SNAKE_THEMES[1],
                         [], [], 0, None)
            self.screen.blit(surf_a, (10, PK_TOP_H))
            self.screen.blit(surf_b,
                             (10 + PK_GAME_SIZE + PK_GAP, PK_TOP_H))
            # 名字
            self.screen.blit(
                self.fonts["md"].render(slot_a.name[:25], True, SNAKE_THEMES[0]["accent"]),
                (10, 8))
            self.screen.blit(
                self.fonts["md"].render(slot_b.name[:25], True,
                                        SNAKE_THEMES[1]["accent"]),
                (10 + PK_GAME_SIZE + PK_GAP, 8))
            # 倒计时数字
            ct = self.fonts["xl"].render(str(count), True, HIGHLIGHT)
            cx = self.screen.get_width() // 2 - ct.get_width() // 2
            cy = PK_TOP_H + PK_GAME_SIZE // 2 - ct.get_height() // 2
            self.screen.blit(ct, (cx, cy))
            pygame.display.flip()
            pygame.time.wait(600)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None

        step_count = 0
        max_steps = 700

        while step_count < max_steps:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        return "abort"
                    if ev.key == pygame.K_UP:
                        self.speed = min(self.speed + 1, 30)
                    if ev.key == pygame.K_DOWN:
                        self.speed = max(self.speed - 1, 1)

            # AI动作
            if world_a.alive:
                act_a, _ = slot_a.get_action(obs_a)
                obs_a, _, _ = world_a.step(act_a)
                if world_a.food != world_a.snake[0]:
                    pass
            if world_b.alive:
                act_b, _ = slot_b.get_action(obs_b)
                obs_b, _, _ = world_b.step(act_b)

            step_count += 1
            self.food_pulse = (self.food_pulse + 0.12) % (2 * math.pi)

            # 吃食物粒子
            if world_a.alive and len(world_a.snake) > 0:
                hd = world_a.snake[0]
                trails_a.append(TrailParticle(
                    hd[0] + GRID // 2, hd[1] + GRID // 2,
                    SNAKE_THEMES[0]["body"]))
            if world_b.alive and len(world_b.snake) > 0:
                hd = world_b.snake[0]
                trails_b.append(TrailParticle(
                    hd[0] + GRID // 2, hd[1] + GRID // 2,
                    SNAKE_THEMES[1]["body"]))

            # 渲染
            self.screen.fill(BG_COLOR)

            render_world(surf_a, world_a, SNAKE_THEMES[0],
                         particles_a, trails_a,
                         self.food_pulse, self.fonts["md"])
            render_world(surf_b, world_b, SNAKE_THEMES[1],
                         particles_b, trails_b,
                         self.food_pulse, self.fonts["md"])

            ax, ay = 10, PK_TOP_H
            bx = 10 + PK_GAME_SIZE + PK_GAP
            self.screen.blit(surf_a, (ax, ay))
            self.screen.blit(surf_b, (bx, ay))

            # 名字标题
            self.screen.blit(
                self.fonts["md"].render(
                    f"🟢 {slot_a.name[:22]}", True,
                    SNAKE_THEMES[0]["accent"]), (ax, 6))
            self.screen.blit(
                self.fonts["md"].render(
                    f"🔵 {slot_b.name[:22]}", True,
                    SNAKE_THEMES[1]["accent"]), (bx, 6))

            # 右侧面板
            px = bx + PK_GAME_SIZE + 10
            py = PK_TOP_H
            pygame.draw.rect(self.screen, PANEL_BG,
                             (px, py, PK_PANEL_W - 10,
                              PK_GAME_SIZE))
            pygame.draw.rect(self.screen, (40, 40, 60),
                             (px, py, PK_PANEL_W - 10,
                              PK_GAME_SIZE), 1)

            self.screen.blit(
                self.fonts["lg"].render(
                    f"Round {self.round_num}", True, HIGHLIGHT),
                (px + 8, py + 8))
            iy = py + 35
            
            comps = [
                ("Score", world_a.score, world_b.score),
                ("Ate", world_a.ate_count, world_b.ate_count),
                ("Reward", world_a.total_reward, world_b.total_reward),
                ("Steps", world_a.total_steps, world_b.total_steps),
                ("Length", len(world_a.snake), len(world_b.snake)),
                ("Alive", "Yes" if world_a.alive else "No",
                 "Yes" if world_b.alive else "No"),
            ]
            for label, va, vb in comps:
                self.screen.blit(
                    self.fonts["sm"].render(label, True, TEXT_DIM),
                    (px + 8, iy))
                iy += 14
                # 双柱
                bar_w = (PK_PANEL_W - 40) // 2
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                    mx = max(abs(va), abs(vb), 1)
                    fa = int(bar_w * min(abs(va) / mx, 1))
                    fb = int(bar_w * min(abs(vb) / mx, 1))
                    pygame.draw.rect(
                        self.screen, SNAKE_THEMES[0]["accent"],
                        (px + 8, iy, fa, 10))
                    pygame.draw.rect(
                        self.screen, SNAKE_THEMES[1]["accent"],
                        (px + 12 + bar_w, iy, fb, 10))
                    va_s = f"{va:.0f}" if isinstance(va, float) else str(va)
                    vb_s = f"{vb:.0f}" if isinstance(vb, float) else str(vb)
                else:
                    va_s = str(va)
                    vb_s = str(vb)
                    ac = GOOD_COLOR if va_s == "Yes" else BAD_COLOR
                    bc = GOOD_COLOR if vb_s == "Yes" else BAD_COLOR
                    self.screen.blit(
                        self.fonts["sm"].render(va_s, True, ac),
                        (px + 8, iy))
                    self.screen.blit(
                        self.fonts["sm"].render(vb_s, True, bc),
                        (px + 12 + bar_w, iy))
                        
                self.screen.blit(
                    self.fonts["sm"].render(str(va_s), True,
                                            SNAKE_THEMES[0]["accent"]),
                    (px + 8, iy + 11))
                self.screen.blit(
                    self.fonts["sm"].render(str(vb_s), True,
                                            SNAKE_THEMES[1]["accent"]),
                    (px + 12 + bar_w, iy + 11))
                iy += 30

            # 胜场
            wa = self.wins.get(slot_a.name, 0)
            wb = self.wins.get(slot_b.name, 0)
            self.screen.blit(
                self.fonts["md"].render(
                    f"Wins: {wa} vs {wb}", True, HIGHLIGHT),
                (px + 8, iy + 5))

            # 速度
            self.screen.blit(
                self.fonts["sm"].render(
                    f"Speed: x{self.speed} [↑↓]", True, TEXT_DIM),
                (px + 8, iy + 28))

            # 底部
            by = PK_TOP_H + PK_GAME_SIZE + 5
            pygame.draw.rect(self.screen, BOTTOM_BG,
                             (0, by, self.screen.get_width(),
                              PK_BOTTOM_H))

            # 领先指示
            if world_a.score > world_b.score:
                lead = f"🟢 {slot_a.name[:20]} LEADING"
                lc = SNAKE_THEMES[0]["accent"]
            elif world_b.score > world_a.score:
                lead = f"🔵 {slot_b.name[:20]} LEADING"
                lc = SNAKE_THEMES[1]["accent"]
            else:
                lead = "⚖️ TIE"
                lc = HIGHLIGHT
            self.screen.blit(
                self.fonts["lg"].render(lead, True, lc), (15, by + 8))

            self.screen.blit(
                self.fonts["sm"].render(
                    f"Step: {step_count}/{max_steps}|  "
                    f"Seed: {seed}  |  "
                    f"[Esc] Abort [↑↓] Speed",
                    True, TEXT_DIM), (15, by + 35))

            status = []
            if not world_a.alive:
                status.append(f"🟢 died at step {world_a.total_steps}")
            if not world_b.alive:
                status.append(f"🔵 died at step {world_b.total_steps}")
            if status:
                self.screen.blit(
                    self.fonts["sm"].render(
                        "|  ".join(status), True, BAD_COLOR),
                    (15, by + 52))

            pygame.display.flip()

            fps = min(BASE_FPS * self.speed, MAX_FPS * self.speed)
            self.clock.tick(fps)
            
            # 双方都死
            if not world_a.alive and not world_b.alive:
                break

        # 结算
        return self._show_result(slot_a, slot_b, world_a, world_b, seed)

    def _show_result(self, slot_a, slot_b, wa, wb, seed):
        """结算画面"""
        sa, sb = wa.score, wb.score
        ra, rb = wa.total_reward, wb.total_reward

        if sa > sb or (sa == sb and ra > rb):
            winner = slot_a.name
            winner_theme = 0
        elif sb > sa or (sb == sa and rb > ra):
            winner = slot_b.name
            winner_theme = 1
        else:
            winner = "TIE"
            winner_theme = -1

        if winner != "TIE":
            self.wins[winner] = self.wins.get(winner, 0) + 1

        # 等待用户操作
        anim = 0
        while True:
            self.screen.fill((5, 5, 15))
            anim += 0.03

            # 标题
            if winner == "TIE":
                ttxt = "DRAW!"
                tc = HIGHLIGHT
            else:
                ttxt = f"WINNER: {winner[:28]}"
                tc = SNAKE_THEMES[winner_theme]["accent"]

            # 脉冲效果
            scale = 1.0 + math.sin(anim * 3) * 0.05
            title_surf = self.fonts["xl"].render(ttxt, True, tc)
            sw = int(title_surf.get_width() * scale)
            sh = int(title_surf.get_height() * scale)
            title_scaled = pygame.transform.smoothscale(title_surf, (sw, sh))
            self.screen.blit(
                title_scaled,
                (self.screen.get_width() // 2 - sw // 2, 30))

            # 对比表
            y = 90
            self.screen.blit(
                self.fonts["md"].render(
                    f"{'':20s} {'🟢 Model A':>15s}{'🔵 Model B':>15s}",
                    True, TEXT_DIM), (30, y))
            y += 25
            rows = [
                ("Model", slot_a.name[:18], slot_b.name[:18]),
                ("Score", str(sa), str(sb)),
                ("Ate", str(wa.ate_count), str(wb.ate_count)),
                ("Total Reward", f"{ra:.1f}", f"{rb:.1f}"),
                ("Steps Survived", str(wa.total_steps), str(wb.total_steps)),
                ("Max Length", str(len(wa.snake)), str(len(wb.snake))),
            ]
            for label, va, vb in rows:
                # 高亮赢家
                try:
                    fva, fvb = float(va), float(vb)
                    ca = GOOD_COLOR if fva > fvb else (BAD_COLOR if fva < fvb else TEXT_COLOR)
                    cb = GOOD_COLOR if fvb > fva else (BAD_COLOR if fvb < fva else TEXT_COLOR)
                except ValueError:
                    ca = cb = TEXT_COLOR

                self.screen.blit(
                    self.fonts["md"].render(
                        f"{label:20s}", True, TEXT_DIM), (30, y))
                self.screen.blit(
                    self.fonts["md"].render(
                        f"{va:>15s}", True, ca), (240, y))
                self.screen.blit(
                    self.fonts["md"].render(
                        f"{vb:>15s}", True, cb), (400, y))
                y += 22

            y += 15
            self.screen.blit(
                self.fonts["md"].render(
                    f"Seed: {seed}", True, TEXT_DIM), (30, y))
            y += 25

            # 胜场
            w_a = self.wins.get(slot_a.name, 0)
            w_b = self.wins.get(slot_b.name, 0)
            self.screen.blit(
                self.fonts["lg"].render(
                    f"Total Wins:  {w_a}  vs  {w_b}",
                    True, HIGHLIGHT), (30, y))
            y += 35

            self.screen.blit(
                self.fonts["md"].render(
                    "[N] Next Round|  [Esc] Back to Menu",
                    True, TEXT_DIM), (30, y))

            # 金色粒子
            if winner != "TIE" and random.random() < 0.3:
                px = random.randint(50, self.screen.get_width() - 50)
                c = SNAKE_THEMES[winner_theme]["accent"]
                p = Particle(px, 25, c, (1, 3))
                p.draw(self.screen)

            pygame.display.flip()
            self.clock.tick(30)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return None
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_n:
                        return "next"
                    if ev.key == pygame.K_ESCAPE:
                        return "menu"


# =========================
# Training Renderer
# =========================
class TrainRenderer:
    def __init__(self, screen, clock, fonts):
        self.screen = screen
        self.clock = clock
        self.fonts = fonts
        self.particles = []
        self.trails = []
        self.food_pulse = 0.0
        self.best_score = 0
        self.heatmap = np.zeros((ROWS, COLS), dtype=np.float32)
        self.show_heatmap = False
        self.show_help = False
        self.death_flash = 0
        self.migrate_msg = ""
        self.migrate_timer = 0
        px = GAME_W + 10
        cw = TRAIN_PANEL_W - 20
        self.chart_score = MiniChart(px, 10, cw, 80, "Score", CHART_SCORE)
        self.chart_reward = MiniChart(px, 100, cw, 80, "Reward", CHART_REWARD)
        self.chart_loss = MiniChart(px, 190, cw, 80, "Loss", CHART_LOSS)
        self.qbar = QValueBar(px, 280, cw, 65)
        
    def add_eat_particles(self, x, y):
        for _ in range(18):
            self.particles.append(
                Particle(x + GRID // 2, y + GRID // 2, FOOD_GLOW))

    def add_death_particles(self, x, y):
        self.death_flash = 8
        for _ in range(25):
            self.particles.append(
                Particle(x + GRID // 2, y + GRID // 2, BAD_COLOR, (2, 6)))

    def add_trail(self, x, y, theme_idx=0):
        self.trails.append(
            TrailParticle(x + GRID // 2, y + GRID // 2,
                          SNAKE_THEMES[theme_idx]["body"]))

    def update_heatmap(self, hx, hy):
        gx = min(max(hx // GRID, 0), COLS - 1)
        gy = min(max(hy // GRID, 0), ROWS - 1)
        self.heatmap[gy, gx] += 1

    def render(self, world, ep=0, total_r=0, best_r=0, eps=0,
               loss_val=0, speed_mult=1, paused=False, manual=False,
               grad_norm=0, lr_now=0, mem_size=0, entropy_coef=0.02,
               train_phase="", probs=None):
        dt = 1.0 / max(BASE_FPS, 30)
        self.food_pulse = (self.food_pulse + 0.12) % (2 * math.pi)

        if self.death_flash > 0:
            fi = min(255, self.death_flash * 30)
            self.screen.fill((fi, 0, 0))
            self.death_flash -= 1
        else:
            self.screen.fill(BG_COLOR)

        # 游戏区
        game_surf = pygame.Surface((GAME_W, GAME_H))
        hm = self.heatmap if self.show_heatmap else None
        render_world(game_surf, world, SNAKE_THEMES[0],
                     self.particles, self.trails,
                     self.food_pulse, self.fonts["md"],
                     show_heatmap=self.show_heatmap, heatmap=hm)
        self.screen.blit(game_surf, (0, 0))

        # 右侧面板
        pygame.draw.rect(self.screen, PANEL_BG,
                         (GAME_W, 0, TRAIN_PANEL_W, GAME_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (GAME_W, 0), (GAME_W, GAME_H), 2)

        self.chart_score.draw(self.screen, self.fonts["sm"])
        self.chart_reward.draw(self.screen, self.fonts["sm"])
        self.chart_loss.draw(self.screen, self.fonts["sm"])
        if probs:
            self.qbar.set_values(probs)
        self.qbar.draw(self.screen, self.fonts["sm"])

        iy = 355
        ipx = GAME_W + 10
        inds = [
            ("Mode", "MANUAL" if manual else "AI", HIGHLIGHT if manual else GOOD_COLOR),
            ("Phase", train_phase, WARN_COLOR),
            ("ε", f"{eps:.4f}", TEXT_COLOR),
            ("LR", f"{lr_now:.1e}", TEXT_COLOR),
            ("Grad", f"{grad_norm:.2f}", TEXT_COLOR),
            ("EntC", f"{entropy_coef:.3f}", TEXT_COLOR),
            ("Mem", f"{mem_size}", TEXT_COLOR),
        ]
        for lb, vl, cl in inds:
            self.screen.blit(
                self.fonts["sm"].render(f"{lb}:", True, TEXT_DIM),
                (ipx, iy))
            self.screen.blit(
                self.fonts["sm"].render(str(vl), True, cl),
                (ipx + 45, iy))
            iy += 14

        if self.migrate_timer > 0:
            self.migrate_timer -= 1
            self.screen.blit(
                self.fonts["sm"].render(self.migrate_msg, True, WARN_COLOR),
                (GAME_W + 5, GAME_H - 16))

        # 底部
        pygame.draw.rect(self.screen, BOTTOM_BG,
                         (0, GAME_H, TRAIN_W, TRAIN_BOTTOM_H))
        pygame.draw.line(self.screen, (50, 50, 80),
                         (0, GAME_H), (TRAIN_W, GAME_H), 2)

        score = world.score
        y1, y2, y3, y4 = GAME_H + 5, GAME_H + 21, GAME_H + 38, GAME_H + 54

        def tx(t, x, y, c, f=None):
            self.screen.blit((f or self.fonts["md"]).render(
                t, True, c), (x, y))

        tx(f"EP:{ep}", 8, y1, TEXT_COLOR)
        tx(f"S:{score}", 90, y1, GOOD_COLOR if score > 0 else TEXT_COLOR)
        tx(f"Best:{self.best_score}", 170, y1, HIGHLIGHT)
        tx(f"R:{total_r:.0f}", 285, y1,
           GOOD_COLOR if total_r > 0 else BAD_COLOR)
        tx(f"BestR:{best_r:.0f}", 390, y1, HIGHLIGHT)
        tx(f"Ate:{world.ate_count}", 520, y1, HIGHLIGHT)

        spd = "PAUSE" if paused else f"x{speed_mult}"
        tx(f"Spd:{spd}", 8, y2,
           BAD_COLOR if paused else TEXT_COLOR)
        tx(f"Loss:{loss_val:.4f}", 120, y2, TEXT_COLOR)
        tx(f"FPS:{self.clock.get_fps():.0f}", 290, y2, TEXT_DIM)
        tx(f"v{VERSION}", 380, y2, TEXT_DIM)

        tx("[Space]Pause [↑↓]Spd [M]Manual [H]Heat " "[P]PK [S]Save",
           8, y3, (85, 85, 110), self.fonts["sm"])
        tx("[R]ResetHeat [Tab]Help [F12]Snap",
           8, y4, (85, 85, 110), self.fonts["sm"])

        if self.show_help:
            self._help()

        pygame.display.flip()

    def _help(self):
        ov = pygame.Surface(
            (self.screen.get_width(), self.screen.get_height()),
            pygame.SRCALPHA)
        ov.fill((0, 0, 0, 190))
        self.screen.blit(ov, (0, 0))
        lines = [
            f"Dreamer Snake AI v{VERSION}",
            "",
            "[Space] Pause    [Up/Down] Speed",
            "[M] Manual mode  [H] Heatmap",
            "[P] PK Arena     [S] Save",
            "[R] Reset heat   [F12] Screenshot",
            "[Tab] This help",
            "",
            "In Manual: Arrow keys control snake",
            "",
            f"State={STATE_DIM}D Latent={LATENT_DIM}D",
            f"NoisyNet+Dueling+PER+{N_STEP}step",
            f"Device: {DEVICE}",
            "",
            "PK Mode: compare 2 models on same seed",
            "",
            "Press Tab to close",
        ]
        y = 35
        for i, l in enumerate(lines):
            c = HIGHLIGHT if i == 0 else TEXT_COLOR
            f = self.fonts["lg"] if i == 0 else self.fonts["md"]
            self.screen.blit(f.render(l, True, c), (35, y))
            y += 22 if i > 0 else 30

    def screenshot(self):
        os.makedirs("screenshots", exist_ok=True)
        fn = f"screenshots/snake_{int(time.time())}.png"
        pygame.image.save(self.screen, fn)
        print(f"  📸 {fn}")


# =========================
# Checkpoint
# =========================
def soft_update(tgt, src, tau=TAU):
    for tp, sp in zip(tgt.parameters(), src.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


def compute_grad_norm(params):
    return sum(p.grad.data.norm(2).item() ** 2
               for p in params if p.grad is not None) ** 0.5


def get_phase(ep):
    if ep < 50:
        return "Warmup"
    if ep < 200:
        return "Explore"
    if ep < 1000:
        return "Learn"
    return "Refine"


def save_all(models, optimizer, memory, ep, best_r, stats, ent_c):
    os.makedirs(CKPT_DIR, exist_ok=True)
    enc, dec, rssm, rm, vm, act = models
    torch.save({
        "version": VERSION, "arch": ARCH_SIGNATURE,
        "episode": ep, "best_reward": best_r,
        "state_dim": STATE_DIM, "latent_dim": LATENT_DIM,
        "entropy_coef": ent_c,
        "encoder": enc.state_dict(), "decoder": dec.state_dict(),
        "rssm": rssm.state_dict(), "reward_model": rm.state_dict(),
        "value_model": vm.state_dict(), "actor": act.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CKPT_MODEL)
    with open(CKPT_MEMORY, "wb") as f:
        pickle.dump(memory.get_data(), f)
    with open(CKPT_STATS, "wb") as f:
        pickle.dump(stats, f)
    with open(CKPT_META, "w") as f:
        json.dump({"version": VERSION, "episode": ep, "best_reward": best_r, "state_dim": STATE_DIM,
                   "latent_dim": LATENT_DIM,
                   "memory_size": len(memory),
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    print(f"  💾 Saved v{VERSION} ep={ep} mem={len(memory)} best={best_r:.1f}")


def save_best(models, best_r, ep):
    enc, dec, rssm, rm, vm, act = models
    torch.save({
        "version": VERSION, "episode": ep, "best_reward": best_r,
        "encoder": enc.state_dict(), "decoder": dec.state_dict(),
        "rssm": rssm.state_dict(), "reward_model": rm.state_dict(),
        "value_model": vm.state_dict(), "actor": act.state_dict(),
    }, CKPT_BEST)
    print(f"  🏆 Best model R={best_r:.1f} @ ep={ep}")


def load_all(models, optimizer):
    stats = {"rewards": [], "scores": [], "losses": []}
    enc, dec, rssm, rm, vm, act = models
    ml = [enc, dec, rssm, rm, vm, act]

    #搜索优先级: v6 > v5 > v4 > v3.1
    search = [CKPT_DIR, "dreamer_snake_v5_ckpt",
              "dreamer_snake_v4_ckpt", "dreamer_snake_ckpt"]
    found_path = None
    found_dir = None
    for d in search:
        p = os.path.join(d, "model.pth")
        if os.path.exists(p):
            found_path = p
            found_dir = d
            break

    if found_path is None:
        print("  🆕 No checkpoint, starting fresh")
        return 0, -1e9, PrioritizedReplayBuffer(MEMORY_SIZE), stats, 0.02, ""
        
    ckpt = torch.load(found_path, map_location=DEVICE, weights_only=False)
    ver = detect_version(ckpt)
    print(f"  📂 Found v{ver} in {found_dir}")

    total_m, total_s = 0, 0
    for name, model in zip(MODEL_KEYS, ml):
        if name in ckpt:
            m, s = migrate_weights(model, ckpt[name], name)
            total_m += m
            total_s += s

    if ver == VERSION and found_dir == CKPT_DIR:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception:
            print("  ⚠ Optimizer re-init")

    mem = PrioritizedReplayBuffer(MEMORY_SIZE)
    mem_path = os.path.join(found_dir, "memory.pkl")
    if os.path.exists(mem_path):
        with open(mem_path, "rb") as f:
            old_data = pickle.load(f)
        if old_data and len(old_data[0][0]) != STATE_DIM:
            old_data = migrate_memory(old_data, len(old_data[0][0]), STATE_DIM)
        for item in old_data:
            mem.push(item)

    stats_path = os.path.join(found_dir, "stats.pkl")
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)

    ep = ckpt.get("episode", 0)
    best = ckpt.get("best_reward", -1e9)
    ent = ckpt.get("entropy_coef", 0.02)
    msg = (f"v{ver}→v{VERSION}: {total_m}ok {total_s}new"
           if ver != VERSION or found_dir != CKPT_DIR else "")
    if msg:
        print(f"  🔄 {msg}")
    print(f"  ✅ ep={ep} mem={len(mem)} best={best:.1f}")
    return ep, best, mem, stats, ent, msg


# =========================
# Main
# =========================
def train():
    pygame.init()
    screen = pygame.display.set_mode((TRAIN_W, TRAIN_H))
    pygame.display.set_caption(f"Dreamer Snake AI v{VERSION}")
    clock = pygame.time.Clock()
    fonts = {
        "sm": pygame.font.SysFont("consolas", 11),
        "md": pygame.font.SysFont("consolas", 13, bold=True),
        "lg": pygame.font.SysFont("consolas", 17, bold=True),
        "xl": pygame.font.SysFont("consolas", 28, bold=True),
    }

    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    rssm_model = RSSM().to(DEVICE)
    reward_model = RewardModel().to(DEVICE)
    value_model = DuelingValueModel().to(DEVICE)
    actor = NoisyActor().to(DEVICE)

    target_value = copy.deepcopy(value_model).to(DEVICE)
    target_value.requires_grad_(False)

    models = (encoder, decoder, rssm_model, reward_model, value_model, actor)
    all_params = []
    for m in models:
        all_params += list(m.parameters())
    optimizer = optim.AdamW(all_params, lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=300, T_mult=2, eta_min=5e-6)

    (start_ep, best_reward, memory,
     stats, entropy_coef, migrate_msg) = load_all(models, optimizer)
    target_value.load_state_dict(value_model.state_dict())

    renderer = TrainRenderer(screen, clock, fonts)
    renderer.best_score = max(stats.get("scores", [0]), default=0)

    for s in stats.get("scores", [])[-200:]:
        renderer.chart_score.add(s)
    for r in stats.get("rewards", [])[-200:]:
        renderer.chart_reward.add(r)
    for l in stats.get("losses", [])[-200:]:
        renderer.chart_loss.add(l)
    if migrate_msg:
        renderer.migrate_msg = migrate_msg
        renderer.migrate_timer = 300

    speed_mult = 1
    paused = False
    manual_mode = False
    last_loss = 0.0
    last_grad = 0.0
    manual_act = 0
    nstep = NStepBuffer(N_STEP, GAMMA)
    cur_probs = [0.33, 0.33, 0.33]

    pk_arena = None

    print(f"\n  🚀 Dreamer Snake v{VERSION} | {DEVICE}")
    print(f"     [P] to enter PK Arena\n")

    for ep in range(start_ep, MAX_EPISODES):
        world = SnakeWorld()
        obs = world.reset()
        total_reward = 0.0
        epsilon = max(0.01, 0.3 * (0.998 ** ep))
        ep_loss = 0.0
        train_cnt = 0
        nstep.reset()
        per_beta = min(1.0, 0.4 + ep * 0.0003)
        phase = get_phase(ep)

        if ep < WARMUP_EPS:
            for pg in optimizer.param_groups:
                pg['lr'] = 3e-4 * (ep + 1) / WARMUP_EPS

        if len(stats["scores"]) >= 100:
            avg_s = sum(stats["scores"][-100:]) / 100
            if avg_s < 2:
                entropy_coef = min(0.05, entropy_coef + 0.001)
            elif avg_s > 8:
                entropy_coef = max(0.005, entropy_coef - 0.001)

        actor.reset_noise()

        for step_i in range(700):
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    save_all(models, optimizer, memory,
                             ep, best_reward, stats, entropy_coef)
                    pygame.quit()
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_SPACE:
                        paused = not paused
                    elif ev.key == pygame.K_UP:
                        if manual_mode:
                            manual_act = 0
                        else:
                            speed_mult = min(speed_mult + 1, 30)
                    elif ev.key == pygame.K_DOWN:
                        if manual_mode:
                            manual_act = 0
                        else:
                            speed_mult = max(speed_mult - 1, 1)
                    elif ev.key == pygame.K_LEFT:
                        if manual_mode:
                            manual_act = 1
                    elif ev.key == pygame.K_RIGHT:
                        if manual_mode:
                            manual_act = 2
                    elif ev.key == pygame.K_s:
                        save_all(models, optimizer, memory,
                                 ep, best_reward, stats, entropy_coef)
                    elif ev.key == pygame.K_m:
                        manual_mode = not manual_mode
                        print(f"  🎮 {'MANUAL' if manual_mode else 'AI'}")
                    elif ev.key == pygame.K_h:
                        renderer.show_heatmap = not renderer.show_heatmap
                    elif ev.key == pygame.K_r:
                        renderer.heatmap = np.zeros((ROWS, COLS), dtype=np.float32)
                    elif ev.key == pygame.K_TAB:
                        renderer.show_help = not renderer.show_help
                    elif ev.key == pygame.K_F12:
                        renderer.screenshot()
                    elif ev.key == pygame.K_p:
                        # ===进入PK模式 ===
                        save_all(models, optimizer, memory,
                                 ep, best_reward, stats, entropy_coef)
                        _run_pk(screen, clock, fonts, pk_arena)
                        #恢复训练窗口
                        screen = pygame.display.set_mode((TRAIN_W, TRAIN_H))
                        pygame.display.set_caption(
                            f"Dreamer Snake AI v{VERSION}")
                        renderer.screen = screen

            if paused:
                lr = optimizer.param_groups[0]['lr']
                renderer.render(
                    world, ep, total_reward, best_reward, epsilon,
                    last_loss, speed_mult, paused, manual_mode,
                    last_grad, lr, len(memory), entropy_coef,
                    phase, cur_probs)
                clock.tick(15)
                continue

            if manual_mode:
                action = manual_act
                manual_act = 0
                cur_probs = [0, 0, 0]
                cur_probs[action] = 1.0
            else:
                ot = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                with torch.no_grad():
                    z, _, _ = encoder(ot)
                    pr = actor(z)
                    cur_probs = pr.squeeze().cpu().tolist()
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    action = torch.multinomial(pr, 1).item()

            prev_head = world.snake[0]
            prev_ate = world.ate_count
            obs2, reward, done = world.step(action)
            total_reward += reward

            # 视觉效果
            if world.alive:
                renderer.add_trail(prev_head[0], prev_head[1])
                renderer.update_heatmap(world.snake[0][0], world.snake[0][1])
            if world.ate_count > prev_ate:
                renderer.add_eat_particles(
                    world.snake[0][0], world.snake[0][1])
            if not world.alive:
                hd = world.snake[0] if world.snake else prev_head
                renderer.add_death_particles(hd[0], hd[1])

            nstep.push((obs, action, reward, obs2, done))
            nt = nstep.get()
            if nt:
                memory.push(nt)
            obs = obs2

            fps = min(BASE_FPS * speed_mult + min(ep, 30),
                      MAX_FPS * speed_mult)
            lr = optimizer.param_groups[0]['lr']
            renderer.render(
                world, ep, total_reward, best_reward, epsilon,
                last_loss, speed_mult, paused, manual_mode,
                last_grad, lr, len(memory), entropy_coef,
                phase, cur_probs)
            clock.tick(fps)

            # 训练
            if len(memory) >= BATCH_SIZE and not manual_mode:
                batch, idx, isw = memory.sample(BATCH_SIZE, per_beta)
                bs = torch.tensor(np.array([t[0] for t in batch]),
                                  dtype=torch.float32, device=DEVICE)
                ba = torch.tensor([t[1] for t in batch],
                                  dtype=torch.float32,
                                  device=DEVICE).unsqueeze(-1)
                br = torch.tensor([t[2] for t in batch],
                                  dtype=torch.float32,
                                  device=DEVICE).unsqueeze(-1)
                bs2 = torch.tensor(np.array([t[3] for t in batch]),
                                   dtype=torch.float32, device=DEVICE)
                bd = torch.tensor([t[4] for t in batch],
                                  dtype=torch.float32,
                                  device=DEVICE).unsqueeze(-1)

                ze, mean, std = encoder(bs)
                z2p = rssm_model(ze, ba)
                z2r, _, _ = encoder(bs2)
                rl = ((decoder(ze) - bs) ** 2).mean()
                tl = ((z2p - z2r.detach()) ** 2).mean()
                rwl = ((reward_model(ze) - br) ** 2).mean()
                kl = (-torch.log(std + 1e-8) +
                      (std ** 2 + mean ** 2 - 1) / 2).mean()
                wl = rl + tl + rwl + 0.1 * kl

                with torch.no_grad():
                    z2t, _, _ = encoder(bs2)
                    vn = target_value(z2t)
                    tgt = br + GAMMA ** N_STEP * vn * (1 - bd)
                vc = value_model(ze.detach())
                td_e = (tgt - vc).detach().squeeze().cpu().numpy()
                vl = (isw.unsqueeze(-1) * (vc - tgt) ** 2).mean()
                memory.update_priorities(idx, td_e)

                adv = (tgt - vc).detach()
                pi = actor(ze.detach())
                lp = torch.log(pi.gather(1, ba.long()) + 1e-8)
                ent = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
                al = -(lp * adv).mean() - entropy_coef * ent

                loss = wl + 0.5 * vl + al
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 10.0)
                last_grad = compute_grad_norm(all_params)
                optimizer.step()
                soft_update(target_value, value_model)
                last_loss = loss.item()
                ep_loss += last_loss
                train_cnt += 1
                if done:
                    for t in nstep.flush():
                        memory.push(t)
                    break

        if ep >= WARMUP_EPS:
            scheduler.step()

        score = world.score
        if total_reward > best_reward:
            best_reward = total_reward
            save_best(models, best_reward, ep)
            renderer.screenshot()
        renderer.best_score = max(renderer.best_score, score)

        stats["rewards"].append(total_reward)
        stats["scores"].append(score)
        al = ep_loss / max(train_cnt, 1)
        stats["losses"].append(al)
        renderer.chart_score.add(score)
        renderer.chart_reward.add(total_reward)
        renderer.chart_loss.add(al)

        rr = stats["rewards"][-50:]
        rs = stats["scores"][-50:]
        lr = optimizer.param_groups[0]['lr']
        mk = "🏆" if total_reward >= best_reward else ""
        print(f"EP {ep:5d}│S {score:3d}│"
              f"R {total_reward:7.1f}│Best {best_reward:7.1f}│"
              f"Avg R:{sum(rr)/len(rr):6.1f} S:{sum(rs)/len(rs):4.1f}│"
              f"ε{epsilon:.3f}│lr{lr:.1e}│"
              f"Mem{len(memory):5d}│L{last_loss:.4f}{mk}")

        if (ep + 1) % SAVE_INTERVAL == 0:
            save_all(models, optimizer, memory,
                     ep + 1, best_reward, stats, entropy_coef)

    save_all(models, optimizer, memory,
             MAX_EPISODES, best_reward, stats, entropy_coef)
    pygame.quit()


def _run_pk(screen, clock, fonts, arena_cache):
    """PK模式主循环"""
    available = scan_all_models()
    if len(available) < 2:
        print("  ⚠ Need at least 2 model checkpoints for PK!")
        print("    Found:", [m["name"] for m in available])
        return

    print(f"\n  🏟️ PK Arena — Found {len(available)} models:")
    for i, m in enumerate(available):
        print(f"    [{i + 1}] {m['name']} "
              f"(v{m.get('version', '?')}, "
              f"ep{m.get('episode', '?')}, "
              f"R={m.get('best_reward', '?')})")

    # 调整窗口
    pk_w = max(PK_W, 830)
    pk_h = max(PK_H, 480)
    screen = pygame.display.set_mode((pk_w, pk_h))
    pygame.display.set_caption(f"Dreamer Snake v{VERSION} — PK Arena")

    arena = PKArena(screen, clock, fonts)

    while True:
        sel = arena.run_selection(available)
        if sel is None:
            return

        info_a, info_b = sel
        print(f"\n  ⚔️ Loading: {info_a['name']} vs {info_b['name']}")
        slot_a = ModelSlot(info_a)
        slot_b = ModelSlot(info_b)

        while True:
            result = arena.run_match(slot_a, slot_b)
            if result is None:
                return
            if result == "menu" or result == "abort":
                break
            # result == "next" → 继续下一轮


if __name__ == "__main__":
    train()