# Dreamer Snake v3.1(PyTorch) — Fixed & Optimized
# ==================================================
# pip install torch pygame numpy

import pygame
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import pickle
import copy
import time

# =========================
# Config
# =========================
WIDTH, HEIGHT_GAME = 400, 400
HEIGHT = 460
GRID = 20
COLS = WIDTH // GRID
ROWS = HEIGHT_GAME // GRID
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STATE_DIM = 12
LATENT_DIM = 32
BATCH_SIZE = 128
MEMORY_SIZE = 20000
GAMMA = 0.99
TAU = 0.005           # target网络软更新系数
BASE_FPS = 12
MAX_FPS = 60
SAVE_INTERVAL = 10
MAX_EPISODES = 999999

CKPT_DIR = "dreamer_snake_ckpt"
CKPT_MODEL = os.path.join(CKPT_DIR, "model.pth")
CKPT_MEMORY = os.path.join(CKPT_DIR, "memory.pkl")
CKPT_STATS = os.path.join(CKPT_DIR, "stats.pkl")

# =========================
# 颜色主题
# =========================
BG_COLOR = (15, 15, 25)
GRID_COLOR = (25, 25, 40)
SNAKE_HEAD = (0, 255, 140)
SNAKE_BODY = (0, 200, 100)
SNAKE_TAIL = (0, 140, 70)
FOOD_COLOR = (255, 60, 80)
FOOD_GLOW = (255, 100, 100)
PANEL_BG = (30, 30, 50)
TEXT_COLOR = (200, 200, 230)
HIGHLIGHT = (255, 220, 80)
GOOD_COLOR = (80, 255, 140)
BAD_COLOR = (255, 80, 80)


# =========================
# 粒子效果
# =========================
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.uniform(0.3, 0.8)
        self.max_life = self.life
        self.color = color
        self.size = random.uniform(2, 5)

    def update(self, dt):
        self.x += self.vx
        self.y += self.vy
        self.vy += 2 * dt
        self.life -= dt
        return self.life > 0

    def draw(self, screen):
        alpha = max(0, self.life / self.max_life)
        r = int(self.size * alpha)
        if r > 0:
            c = tuple(int(v * alpha) for v in self.color)
            pygame.draw.circle(screen, c, (int(self.x), int(self.y)), r)


# =========================
# Game
# =========================
class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("🐍 Dreamer Snake AI v3.1")
        self.clock = pygame.time.Clock()
        self.font_sm = pygame.font.SysFont("consolas", 13)
        self.font_md = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 20, bold=True)
        self.particles = []
        self.food_pulse = 0
        self.best_score = 0
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
        return self.state()

    def _spawn(self):
        candidates = []
        for x in range(0, WIDTH, GRID):
            for y in range(0, HEIGHT_GAME, GRID):
                if (x, y) not in self.snake:
                    candidates.append((x, y))
        if not candidates:
            return (0, 0)
        return random.choice(candidates)

    def _dist_to_food(self, pos):
        return abs(pos[0] - self.food[0]) + abs(pos[1] - self.food[1])

    def step(self, action):
        dirs = [(GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID)]
        idx = dirs.index(self.dir)
        if action == 1:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        self.dir = dirs[idx]

        old_dist = self._dist_to_food(self.snake[0])
        head = (self.snake[0][0] + self.dir[0],
                self.snake[0][1] + self.dir[1])

        self.total_steps += 1

        # 撞墙或撞身
        if (head[0] < 0 or head[0] >= WIDTH or
                head[1] < 0 or head[1] >= HEIGHT_GAME or
                head in self.snake):
            return self.state(), -10.0, True

        self.snake.insert(0, head)
        self.steps_no_food += 1

        if head == self.food:
            reward = 10.0 + len(self.snake) * 0.5
            self.ate_count += 1
            # 粒子特效
            for _ in range(15):
                self.particles.append(
                    Particle(head[0] + GRID // 2, head[1] + GRID // 2, FOOD_GLOW))
            self.food = self._spawn()
            self.steps_no_food = 0                    # ← 修复1
        else:                                         # ← 修复1
            self.snake.pop()
            new_dist = self._dist_to_food(head)
            if new_dist < old_dist:
                reward = 0.1
            else:
                reward = -0.15

        # 超时
        timeout = 100 + len(self.snake) * 20          # ← 修复2
        if self.steps_no_food > timeout:
            return self.state(), -5.0, True

        self.best_score = max(self.best_score, len(self.snake) - 3)
        return self.state(), reward, False

    def state(self):
        h = self.snake[0]
        dirs = [(GRID, 0), (0, GRID), (-GRID, 0), (0, -GRID)]
        idx = dirs.index(self.dir)

        def danger(pos):
            return float(
                pos[0] < 0 or pos[0] >= WIDTH or
                pos[1] < 0 or pos[1] >= HEIGHT_GAME or
                pos in self.snake)

        def dist_wall(direction):
            dx, dy = direction
            count = 0
            px, py = h
            while True:
                px += dx
                py += dy
                if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT_GAME:
                    break
                count += 1
            return count / max(COLS, ROWS)

        ahead = (h[0] + dirs[idx][0], h[1] + dirs[idx][1])
        left = (h[0] + dirs[(idx - 1) % 4][0], h[1] + dirs[(idx - 1) % 4][1])
        right = (h[0] + dirs[(idx + 1) % 4][0], h[1] + dirs[(idx + 1) % 4][1])

        return np.array([
            (self.food[0] - h[0]) / WIDTH,
            (self.food[1] - h[1]) / HEIGHT_GAME,
            self.dir[0] / GRID,
            self.dir[1] / GRID,
            danger(ahead),
            danger(left),
            danger(right),
            len(self.snake) / (COLS * ROWS),
            dist_wall(dirs[idx]),
            dist_wall(dirs[(idx - 1) % 4]),
            dist_wall(dirs[(idx + 1) % 4]),
            self.steps_no_food / 200.0,
        ], dtype=np.float32)

    def render(self, ep=0, total_r=0, best_r=0, eps=0, loss_val=0,
               speed_mult=1, paused=False, train_stats=None):
        self.screen.fill(BG_COLOR)

        # 网格
        for x in range(0, WIDTH, GRID):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, HEIGHT_GAME))
        for y in range(0, HEIGHT_GAME, GRID):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (WIDTH, y))

        # 食物发光
        self.food_pulse = (self.food_pulse + 0.1) % (2 * math.pi)
        glow_r = int(GRID * 0.8 + math.sin(self.food_pulse) * 3)
        glow_surf = pygame.Surface((glow_r * 4, glow_r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*FOOD_GLOW, 40),
                           (glow_r * 2, glow_r * 2), glow_r * 2)
        self.screen.blit(glow_surf,
                         (self.food[0] + GRID // 2 - glow_r * 2,
                          self.food[1] + GRID // 2 - glow_r * 2))
        pygame.draw.rect(self.screen, FOOD_COLOR,
                         (self.food[0] + 2, self.food[1] + 2,
                          GRID - 4, GRID - 4), border_radius=6)

        # 蛇身渐变
        n = len(self.snake)
        for i, s in enumerate(self.snake):
            t = i / max(n - 1, 1)
            r = int(SNAKE_HEAD[0] * (1 - t) + SNAKE_TAIL[0] * t)
            g = int(SNAKE_HEAD[1] * (1 - t) + SNAKE_TAIL[1] * t)
            b = int(SNAKE_HEAD[2] * (1 - t) + SNAKE_TAIL[2] * t)
            pad = 1 if i == 0 else 2                # ← 修复3
            br = 5 if i == 0 else 3
            pygame.draw.rect(self.screen, (r, g, b),
                             (s[0] + pad, s[1] + pad,
                              GRID - pad * 2, GRID - pad * 2),
                             border_radius=br)
            if i == 0:                                 # ← 修复4
                # 蛇眼
                ex = s[0] + GRID // 2 + self.dir[0] // 4
                ey = s[1] + GRID // 2 + self.dir[1] // 4
                pygame.draw.circle(self.screen, (255, 255, 255), (ex, ey), 3)
                pygame.draw.circle(self.screen, (0, 0, 0), (ex, ey), 1)  # ← 修复5

        # 粒子
        dt = 1.0 / max(BASE_FPS, 30)
        self.particles = [p for p in self.particles if p.update(dt)]
        for p in self.particles:
            p.draw(self.screen)

        # 底部信息面板
        pygame.draw.rect(self.screen, PANEL_BG, (0, HEIGHT_GAME, WIDTH, 60))
        pygame.draw.line(self.screen, (60, 60, 90),
                         (0, HEIGHT_GAME), (WIDTH, HEIGHT_GAME), 2)

        score = len(self.snake) - 3
        # 第一行
        y1 = HEIGHT_GAME + 6
        self._text(f"EP:{ep}", 8, y1, TEXT_COLOR)
        self._text(f"Score:{score}", 90, y1, GOOD_COLOR if score > 0 else TEXT_COLOR)
        self._text(f"Best:{self.best_score}", 185, y1, HIGHLIGHT)
        self._text(f"R:{total_r:.0f}", 285, y1,
                   GOOD_COLOR if total_r > 0 else BAD_COLOR)

        # 第二行
        y2 = HEIGHT_GAME + 26
        self._text(f"\u03b5:{eps:.3f}", 8, y2, TEXT_COLOR)
        self._text(f"Loss:{loss_val:.2f}", 90, y2, TEXT_COLOR)
        spd_txt = "PAUSE" if paused else f"x{speed_mult}"
        self._text(f"Spd:{spd_txt}", 195, y2,
                   BAD_COLOR if paused else TEXT_COLOR)
        self._text(f"Ate:{self.ate_count}", 285, y2, HIGHLIGHT)

        # 第三行
        y3 = HEIGHT_GAME + 43
        self._text("[Space]=Pause [↑↓]=Speed [S]=Save", 8, y3,
                   (120, 120, 150), font=self.font_sm)

        pygame.display.flip()

    def _text(self, txt, x, y, color, font=None):
        f = font or self.font_md
        self.screen.blit(f.render(txt, True, color), (x, y))


# =========================
# Dreamer Modules (加深加宽)
# =========================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ELU(),
            nn.LayerNorm(128), # ← 优化: LayerNorm稳定训练
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, LATENT_DIM * 2))

    def forward(self, x):
        h = self.net(x)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        std = torch.exp(log_std.clamp(-5, 2))
        z = mean + std * torch.randn_like(std)
        return z, mean, std


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, STATE_DIM))

    def forward(self, z):
        return self.net(z)


class RSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + 1, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, LATENT_DIM))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], dim=-1))


class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z)


class ValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 1))

    def forward(self, z):
        return self.net(z)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 64), nn.ELU(),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, 3))

    def forward(self, z):
        return torch.softmax(self.net(z), dim=-1)


# =========================
# Target网络软更新 (优化)
# =========================
def soft_update(target, source, tau=TAU):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# =========================
# Checkpoint
# =========================
def save_all(models, optimizer, memory, episode, best_reward, stats):
    os.makedirs(CKPT_DIR, exist_ok=True)
    enc, dec, rssm_m, rm, vm, act = models
    torch.save({
        "episode": episode,
        "best_reward": best_reward,
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "rssm": rssm_m.state_dict(),
        "reward_model": rm.state_dict(),
        "value_model": vm.state_dict(),
        "actor": act.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, CKPT_MODEL)
    with open(CKPT_MEMORY, "wb") as f:
        pickle.dump(list(memory), f)
    with open(CKPT_STATS, "wb") as f:
        pickle.dump(stats, f)
    print(f"  💾 Saved | ep={episode} | mem={len(memory)} | best={best_reward:.1f}")


def load_all(models, optimizer):
    stats = {"rewards": [], "scores": [], "losses": []}
    if not os.path.exists(CKPT_MODEL):
        return 0, -1e9, deque(maxlen=MEMORY_SIZE), stats  # ← 修复6
    enc, dec, rssm_m, rm, vm, act = models
    ckpt = torch.load(CKPT_MODEL, map_location=DEVICE, weights_only=False) # ← 修复7
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    rssm_m.load_state_dict(ckpt["rssm"])
    rm.load_state_dict(ckpt["reward_model"])
    vm.load_state_dict(ckpt["value_model"])
    act.load_state_dict(ckpt["actor"])
    optimizer.load_state_dict(ckpt["optimizer"])
    mem = deque(maxlen=MEMORY_SIZE)
    if os.path.exists(CKPT_MEMORY):
        with open(CKPT_MEMORY, "rb") as f:
            mem.extend(pickle.load(f))
    if os.path.exists(CKPT_STATS):
        with open(CKPT_STATS, "rb") as f:
            stats = pickle.load(f)
    ep = ckpt["episode"]
    best = ckpt.get("best_reward", -1e9)
    print(f"  📂 Loaded | ep={ep} | mem={len(mem)} | best={best:.1f}")
    return ep, best, mem, stats


# =========================
# Training
# =========================
def train():
    game = SnakeGame()

    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    rssm_model = RSSM().to(DEVICE)
    reward_model = RewardModel().to(DEVICE)
    value_model = ValueModel().to(DEVICE)
    actor = Actor().to(DEVICE)

    # 优化: Target Value网络 —稳定TD目标
    target_value = copy.deepcopy(value_model).to(DEVICE)
    target_value.requires_grad_(False)

    models = (encoder, decoder, rssm_model, reward_model, value_model, actor)

    all_params = []
    for m in models:
        all_params += list(m.parameters())
    optimizer = optim.Adam(all_params, lr=3e-4)

    # 优化: 学习率余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=200, T_mult=2, eta_min=1e-5)

    start_ep, best_reward, memory, stats = load_all(models, optimizer)
    game.best_score = max(stats.get("scores", [0]), default=0)

    speed_mult = 1
    paused = False
    last_loss = 0.0

    for ep in range(start_ep, MAX_EPISODES):
        obs = game.reset()
        total_reward = 0.0
        # 优化: 指数衰减epsilon
        epsilon = max(0.03, 0.6 * (0.997 ** ep))
        ep_loss = 0.0
        train_count = 0

        for step_i in range(500):
            # 事件处理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    save_all(models, optimizer, memory, ep, best_reward, stats)
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        speed_mult = min(speed_mult + 1, 10)
                    elif event.key == pygame.K_DOWN:
                        speed_mult = max(speed_mult - 1, 1)
                    elif event.key == pygame.K_s:
                        save_all(models, optimizer, memory, ep, best_reward, stats)

            if paused:
                game.render(ep, total_reward, best_reward, epsilon,
                            last_loss, speed_mult, paused)
                game.clock.tick(15)
                continue

            # 选动作
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                z, _, _ = encoder(obs_t)
                probs = actor(z)

            if random.random() < epsilon:
                action = random.randint(0, 2)
            else:
                action = torch.multinomial(probs, 1).item()

            obs2, reward, done = game.step(action)
            memory.append((obs, action, reward, obs2, done))
            obs = obs2
            total_reward += reward

            # 渲染
            fps = min(BASE_FPS * speed_mult + min(ep, 30),
                      MAX_FPS * speed_mult)
            game.render(ep, total_reward, best_reward, epsilon,
                        last_loss, speed_mult, paused)
            game.clock.tick(fps)

            # ========== 批量训练 ==========
            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                b_s = torch.tensor(
                    np.array([t[0] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                b_a = torch.tensor(
                    [t[1] for t in batch],
                    dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                b_r = torch.tensor(
                    [t[2] for t in batch],
                    dtype=torch.float32, device=DEVICE).unsqueeze(-1)
                b_s2 = torch.tensor(
                    np.array([t[3] for t in batch]),
                    dtype=torch.float32, device=DEVICE)
                b_d = torch.tensor(
                    [t[4] for t in batch],
                    dtype=torch.float32, device=DEVICE).unsqueeze(-1)

                # --- World Model ---
                z_enc, mean, std = encoder(b_s)
                z2_pred = rssm_model(z_enc, b_a)
                z2_real, _, _ = encoder(b_s2)

                recon_loss = ((decoder(z_enc) - b_s) ** 2).mean()
                trans_loss = ((z2_pred - z2_real.detach()) ** 2).mean()
                reward_loss = ((reward_model(z_enc) - b_r) ** 2).mean()
                kl_loss = (-torch.log(std + 1e-8) + (std ** 2 + mean ** 2 - 1) / 2).mean()

                world_loss = (recon_loss + trans_loss +
                              reward_loss + 0.1 * kl_loss)

                # --- Value (使用target网络计算TD目标) ---
                with torch.no_grad():
                    z2_tgt, _, _ = encoder(b_s2)
                    v_next = target_value(z2_tgt)           # ← 优化
                    target = b_r + GAMMA * v_next * (1 - b_d)
                v_cur = value_model(z_enc.detach())
                value_loss = ((v_cur - target) ** 2).mean()

                # --- Actor (策略梯度 +熵正则) ---
                advantage = (target - v_cur).detach()
                pi = actor(z_enc.detach())
                log_pi = torch.log(pi.gather(1, b_a.long()) + 1e-8)
                entropy = -(pi * torch.log(pi + 1e-8)).sum(-1).mean()
                actor_loss = -(log_pi * advantage).mean() - 0.02 * entropy

                loss = world_loss + 0.5 * value_loss + actor_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 10.0)
                optimizer.step()

                # 优化: target网络软更新
                soft_update(target_value, value_model)

                last_loss = loss.item()                # ← 修复8
                ep_loss += last_loss
                train_count += 1
                if done:                               # ← 修复9
                    break

        # 学习率调度
        scheduler.step()

        # 统计
        score = len(game.snake) - 3
        if total_reward > best_reward:
            best_reward = total_reward
        game.best_score = max(game.best_score, score)

        stats["rewards"].append(total_reward)
        stats["scores"].append(score)
        if train_count > 0:
            stats["losses"].append(ep_loss / train_count)

        # 最近50轮平均
        recent_r = stats["rewards"][-50:]
        recent_s = stats["scores"][-50:]
        avg_r = sum(recent_r) / len(recent_r)
        avg_s = sum(recent_s) / len(recent_s)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"EP {ep:5d} │ Score {score:3d} │ "
              f"R {total_reward:7.1f} │ Best {best_reward:7.1f} │ "
              f"Avg50_R {avg_r:6.1f} │ Avg50_S {avg_s:4.1f} │ "
              f"ε {epsilon:.3f} │ lr {lr_now:.1e} │ "
              f"Mem {len(memory):5d} │ Loss {last_loss:.3f}")

        # 定期保存
        if (ep + 1) % SAVE_INTERVAL == 0:
            save_all(models, optimizer, memory, ep + 1, best_reward, stats)

    save_all(models, optimizer, memory, MAX_EPISODES, best_reward, stats)
    pygame.quit()


if __name__ == "__main__":
    train()