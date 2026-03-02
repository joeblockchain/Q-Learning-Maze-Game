import numpy as np
import pygame
import sys
import time
from typing import Tuple

class GridWorldEnv:
    def __init__(self, width: int = 6, height: int = 6):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start_pos = (0, 0)
        self.goal_pos = (5, 4)
        self.grid[0, 0] = 3
        self.grid[4, 5] = 2
        self.grid[0, 3] = 1
        self.grid[1, 3] = 1
        self.grid[3, 1] = 1
        self.grid[4, 1] = 1
        self.grid[1, 1] = -1
        self.grid[3, 3] = -1
        self.agent_pos = self.start_pos

    @property
    def n_states(self) -> int:
        return self.width * self.height

    @property
    def n_actions(self) -> int:
        return 4

    def reset(self) -> int:
        self.agent_pos = self.start_pos
        return self._pos_to_state(self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool]:
        x, y = self.agent_pos
        new_x, new_y = x, y
        if action == 0:
            new_y -= 1
        elif action == 1:
            new_x += 1
        elif action == 2:
            new_y += 1
        elif action == 3:
            new_x -= 1
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            reward = -5.0
            done = False
            return self._pos_to_state(self.agent_pos), reward, done
        cell_value = self.grid[new_y, new_x]
        if cell_value == 1:
            reward = -5.0
            done = False
            return self._pos_to_state(self.agent_pos), reward, done
        self.agent_pos = (new_x, new_y)
        if cell_value == 2:
            reward = 10.0
            done = True
        elif cell_value == -1:
            reward = -10.0
            done = True
        else:
            reward = -1.0
            done = False
        return self._pos_to_state(self.agent_pos), reward, done

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return y * self.width + x

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        y = state // self.width
        x = state % self.width
        return x, y


class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next_q = 0.0 if done else np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


CELL_SIZE = 80
MARGIN = 2
INFO_PANEL_WIDTH = 260
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

COLOR_BACKGROUND = (220, 220, 220)
COLOR_EMPTY = (255, 255, 255)
COLOR_WALL = (0, 0, 0)
COLOR_TRAP = (255, 100, 100)
COLOR_GOAL = (100, 255, 100)
COLOR_START = (200, 200, 255)
COLOR_AGENT = (50, 50, 255)
COLOR_PANEL_TEXT = (0, 0, 0)
COLOR_MENU_BG = (40, 40, 60)
COLOR_BUTTON = (80, 120, 200)
COLOR_BUTTON_HOVER = (120, 160, 240)
COLOR_BUTTON_TEXT = (255, 255, 255)
COLOR_TITLE = (255, 255, 255)

MODE_MENU = "MENU"
MODE_TRAIN = "TRAIN"
MODE_DEMO = "DEMO"
MODE_HUMAN = "HUMAN"


def draw_env(screen, font, env: GridWorldEnv, episode: int, step: int,
             last_action: int, last_reward: float, total_reward: float,
             epsilon: float, mode: str):
    screen.fill(COLOR_BACKGROUND)
    for y in range(env.height):
        for x in range(env.width):
            rect = pygame.Rect(
                x * CELL_SIZE + MARGIN,
                y * CELL_SIZE + MARGIN,
                CELL_SIZE - 2 * MARGIN,
                CELL_SIZE - 2 * MARGIN,
            )
            cell = env.grid[y, x]
            color = COLOR_EMPTY
            if cell == 1:
                color = COLOR_WALL
            elif cell == -1:
                color = COLOR_TRAP
            elif cell == 2:
                color = COLOR_GOAL
            elif cell == 3:
                color = COLOR_START
            pygame.draw.rect(screen, color, rect)
    ax, ay = env.agent_pos
    agent_rect = pygame.Rect(
        ax * CELL_SIZE + 10,
        ay * CELL_SIZE + 10,
        CELL_SIZE - 20,
        CELL_SIZE - 20,
    )
    pygame.draw.rect(screen, COLOR_AGENT, agent_rect)
    panel_x = env.width * CELL_SIZE + 10
    text_y = 20

    def blit_text(text):
        nonlocal text_y
        surf = font.render(text, True, COLOR_PANEL_TEXT)
        screen.blit(surf, (panel_x, text_y))
        text_y += 28

    blit_text(f"Mode: {mode}")
    blit_text(f"Episode: {episode}")
    blit_text(f"Step: {step}")
    blit_text(f"Last action: {ACTION_NAMES.get(last_action, '-')}")
    blit_text(f"Last reward: {last_reward:.2f}")
    blit_text(f"Total reward: {total_reward:.2f}")
    blit_text(f"Epsilon: {epsilon:.3f}")
    blit_text("Press M for menu")
    pygame.display.flip()


def draw_menu(screen, title_font, button_font, buttons):
    screen.fill(COLOR_MENU_BG)
    width, height = screen.get_size()
    title_surf = title_font.render("Q-Learning Maze Game", True, COLOR_TITLE)
    title_rect = title_surf.get_rect(center=(width // 2, 100))
    screen.blit(title_surf, title_rect)
    subtitle = "Reinforcement Learning with Q-Learning"
    subtitle_surf = button_font.render(subtitle, True, COLOR_TITLE)
    subtitle_rect = subtitle_surf.get_rect(center=(width // 2, 160))
    screen.blit(subtitle_surf, subtitle_rect)
    mouse_pos = pygame.mouse.get_pos()
    for label, rect in buttons.items():
        if rect.collidepoint(mouse_pos):
            color = COLOR_BUTTON_HOVER
        else:
            color = COLOR_BUTTON
        pygame.draw.rect(screen, color, rect, border_radius=8)
        text_surf = button_font.render(label, True, COLOR_BUTTON_TEXT)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)
    pygame.display.flip()


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Q-learning Grid Maze")
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 38, bold=True)
        self.button_font = pygame.font.SysFont("Arial", 24)
        self.env = GridWorldEnv()
        self.agent = QLearningAgent(
            n_states=self.env.n_states,
            n_actions=self.env.n_actions,
            learning_rate=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.995,
        )
        self.screen_width = self.env.width * CELL_SIZE + INFO_PANEL_WIDTH
        self.screen_height = self.env.height * CELL_SIZE
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.mode = MODE_MENU
        self.training_configured = False
        self.demo_configured = False
        self.human_configured = False
        self.train_episodes = 800
        self.max_steps_per_episode = 50
        self.current_episode = 0
        self.current_step = 0
        self.state = 0
        self.total_reward = 0.0
        self.last_action = -1
        self.last_reward = 0.0
        self.demo_last_step_time = 0.0
        self.demo_step_interval = 0.2
        self.menu_buttons = self._create_menu_buttons()
        self.pause_until = 0.0
        self.demo_pending_reset = False
        self.human_pending_reset = False

    def _create_menu_buttons(self):
        buttons = {}
        center_x = self.screen_width // 2
        start_y = 220
        w = 260
        h = 50
        gap = 20
        labels = ["Train Agent", "Watch Agent", "Human Play", "Quit"]
        for i, label in enumerate(labels):
            rect = pygame.Rect(0, 0, w, h)
            rect.center = (center_x, start_y + i * (h + gap))
            buttons[label] = rect
        return buttons

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_m:
                        self.mode = MODE_MENU
                    if self.mode == MODE_HUMAN:
                        if event.key == pygame.K_UP:
                            self._human_take_action(0)
                        elif event.key == pygame.K_RIGHT:
                            self._human_take_action(1)
                        elif event.key == pygame.K_DOWN:
                            self._human_take_action(2)
                        elif event.key == pygame.K_LEFT:
                            self._human_take_action(3)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mode == MODE_MENU:
                        self._handle_menu_click(event.pos)

            now = time.time()
            if now < self.pause_until:
                if self.mode == MODE_MENU:
                    draw_menu(self.screen, self.title_font, self.button_font, self.menu_buttons)
                elif self.mode in (MODE_TRAIN, MODE_DEMO, MODE_HUMAN):
                    draw_env(
                        self.screen,
                        self.font,
                        self.env,
                        self.current_episode if self.mode == MODE_TRAIN else 0,
                        self.current_step,
                        self.last_action,
                        self.last_reward,
                        self.total_reward,
                        self.agent.epsilon if self.mode != MODE_HUMAN else 0.0,
                        self.mode,
                    )
                continue

            if self.mode == MODE_MENU:
                draw_menu(self.screen, self.title_font, self.button_font, self.menu_buttons)
            elif self.mode == MODE_TRAIN:
                self._training_step()
            elif self.mode == MODE_DEMO:
                self._demo_step()
            elif self.mode == MODE_HUMAN:
                self._human_draw()
        pygame.quit()
        sys.exit()

    def _handle_menu_click(self, pos):
        for label, rect in self.menu_buttons.items():
            if rect.collidepoint(pos):
                if label == "Train Agent":
                    self._configure_training()
                    self.mode = MODE_TRAIN
                elif label == "Watch Agent":
                    if not self.training_configured:
                        self._configure_training()
                        for _ in range(300):
                            self._training_inner_step(render=False)
                    self._configure_demo()
                    self.mode = MODE_DEMO
                elif label == "Human Play":
                    self._configure_human()
                    self.mode = MODE_HUMAN
                elif label == "Quit":
                    pygame.quit()
                    sys.exit()

    def _configure_training(self):
        self.training_configured = True
        self.current_episode = 1
        self.current_step = 0
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.last_action = -1
        self.last_reward = 0.0

    def _training_inner_step(self, render: bool):
        action = self.agent.select_action(self.state)
        next_state, reward, done = self.env.step(action)
        self.agent.update(self.state, action, reward, next_state, done)
        self.state = next_state
        self.total_reward += reward
        self.last_action = action
        self.last_reward = reward
        self.current_step += 1
        if render:
            draw_env(
                self.screen,
                self.font,
                self.env,
                self.current_episode,
                self.current_step,
                self.last_action,
                self.last_reward,
                self.total_reward,
                self.agent.epsilon,
                MODE_TRAIN,
            )
            pygame.time.delay(40)
        if done or self.current_step >= self.max_steps_per_episode:
            self.agent.decay_epsilon()
            if self.current_episode % 50 == 0 and render:
                print(
                    f"Episode {self.current_episode}/{self.train_episodes}, "
                    f"total_reward={self.total_reward:.1f}, epsilon={self.agent.epsilon:.3f}"
                )
            self.current_episode += 1
            self.current_step = 0
            self.state = self.env.reset()
            self.total_reward = 0.0
            self.last_action = -1
            self.last_reward = 0.0

    def _training_step(self):
        if self.current_episode > self.train_episodes:
            np.save("q_table_trained.npy", self.agent.q_table)
            np.savetxt("q_table_trained.csv", self.agent.q_table, delimiter=",")
            print("Q-table saved as q_table_trained.npy and q_table_trained.csv")
            self.mode = MODE_MENU
            return
        self._training_inner_step(render=True)

    def _configure_demo(self):
        self.demo_configured = True
        self.agent.epsilon = 0.0
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.last_action = -1
        self.last_reward = 0.0
        self.current_step = 0
        self.demo_last_step_time = time.time()
        self.demo_pending_reset = False

    def _demo_step(self):
        if self.demo_pending_reset:
            if time.time() >= self.pause_until:
                self.demo_pending_reset = False
                self.state = self.env.reset()
                self.total_reward = 0.0
                self.last_action = -1
                self.last_reward = 0.0
                self.current_step = 0
            draw_env(
                self.screen,
                self.font,
                self.env,
                0,
                self.current_step,
                self.last_action,
                self.last_reward,
                self.total_reward,
                self.agent.epsilon,
                MODE_DEMO,
            )
            return
        now = time.time()
        if now - self.demo_last_step_time >= self.demo_step_interval:
            self.demo_last_step_time = now
            action = self.agent.select_action(self.state)
            next_state, reward, done = self.env.step(action)
            self.state = next_state
            self.total_reward += reward
            self.last_action = action
            self.last_reward = reward
            self.current_step += 1
            if done or self.current_step >= self.max_steps_per_episode:
                self.demo_pending_reset = True
                self.pause_until = time.time() + 0.8
        draw_env(
            self.screen,
            self.font,
            self.env,
            0,
            self.current_step,
            self.last_action,
            self.last_reward,
            self.total_reward,
            self.agent.epsilon,
            MODE_DEMO,
        )

    def _configure_human(self):
        self.human_configured = True
        self.state = self.env.reset()
        self.total_reward = 0.0
        self.last_action = -1
        self.last_reward = 0.0
        self.current_step = 0
        self.human_pending_reset = False

    def _human_take_action(self, action: int):
        if self.human_pending_reset:
            return
        next_state, reward, done = self.env.step(action)
        self.state = next_state
        self.total_reward += reward
        self.last_action = action
        self.last_reward = reward
        self.current_step += 1
        if done or self.current_step >= self.max_steps_per_episode:
            self.human_pending_reset = True
            self.pause_until = time.time() + 0.6

    def _human_draw(self):
        if self.human_pending_reset and time.time() >= self.pause_until:
            self.human_pending_reset = False
            self.state = self.env.reset()
            self.total_reward = 0.0
            self.last_action = -1
            self.last_reward = 0.0
            self.current_step = 0
        draw_env(
            self.screen,
            self.font,
            self.env,
            0,
            self.current_step,
            self.last_action,
            self.last_reward,
            self.total_reward,
            0.0,
            MODE_HUMAN,
        )


if __name__ == "__main__":
    game = Game()
    game.run()