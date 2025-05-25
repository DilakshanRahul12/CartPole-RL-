import pygame
import numpy as np
import math
import time

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CartPole with Pygame")

# Colors and Constants
WHITE, BLACK, RED = (255, 255, 255), (0, 0, 0), (200, 0, 0)
PINK = (255, 105, 180)  # Pink for Point A
BLUE = (0, 0, 255)      # Blue for Point B
FPS = 60
PIXELS_PER_METER = 100
CART_WIDTH, CART_HEIGHT = 60, 30
POLE_WIDTH = 10

# Physics constants
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
POLE_LENGTH = 0.5  # meters
FORCE_MAG = 15.0
TAU = 0.02
MAX_POSITION = 2.4  # meters
EPISODE_DURATION = 5.0  # seconds

# Font for scoreboard
FONT = pygame.font.Font(None, 36)

class CartPoleEnv:
    def __init__(self):
        self.cart_y = HEIGHT / 2 + 100
        self.reset()
        
    def reset(self):
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0  # Pole upright with Point B above Point A
        self.theta_dot = 0.0
        self.start_time = time.time()  # Record episode start time
        return self.get_state()

    def get_state(self):
        return (self.x, self.x_dot, self.theta, self.theta_dot)

    def step(self, action):
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        
        # Physics calculations
        costheta = math.cos(self.theta)
        sintheta = math.sin(self.theta)
        
        total_mass = CART_MASS + POLE_MASS
        pole_half_length = POLE_LENGTH / 2
        
        temp = (force + POLE_MASS * pole_half_length * self.theta_dot**2 * sintheta) / total_mass
        theta_acc = (GRAVITY * sintheta - costheta * temp) / (
            pole_half_length * (4/3 - (POLE_MASS * costheta**2) / total_mass)
        )
        x_acc = temp - (POLE_MASS * pole_half_length * theta_acc * costheta) / total_mass

        # Update state
        self.x += self.x_dot * TAU
        self.x_dot += x_acc * TAU
        self.theta += self.theta_dot * TAU
        self.theta_dot += theta_acc * TAU

        # Friction
        self.x_dot *= 0.95

        # Calculate positions
        cart_x = WIDTH / 2 + self.x * PIXELS_PER_METER
        point_a_x = cart_x
        point_a_y = self.cart_y - CART_HEIGHT  # 370.0
        point_b_x = cart_x + POLE_LENGTH * PIXELS_PER_METER * math.sin(self.theta)
        point_b_y = self.cart_y - CART_HEIGHT - POLE_LENGTH * PIXELS_PER_METER * math.cos(self.theta)
        is_b_above_a = point_b_y < point_a_y
        max_distance = POLE_LENGTH * PIXELS_PER_METER
        position_penalty = -abs(self.x) / MAX_POSITION

        if is_b_above_a and abs(self.x) < MAX_POSITION:
            reward = (point_a_y - point_b_y) / max_distance + position_penalty
        else:
            reward = -10.0

        # Check if 5 seconds have elapsed
        elapsed_time = time.time() - self.start_time
        done = elapsed_time >= EPISODE_DURATION
        
        return self.get_state(), reward, done, (point_a_x, point_a_y, point_b_x, point_b_y)

class QLearningAgent:
    def __init__(self, n_actions, state_bins):
        self.n_actions = n_actions
        self.state_bins = state_bins
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.alpha = 0.1
        self.gamma = 0.99
        self.q_table = np.random.uniform(low=-1, high=1, size=state_bins + (n_actions,))

    def discretize(self, state):
        x, x_dot, theta, theta_dot = state
        bins = [
            np.linspace(-MAX_POSITION, MAX_POSITION, self.state_bins[0]),
            np.linspace(-5, 5, self.state_bins[1]),
            np.linspace(-math.pi / 2, math.pi / 2, self.state_bins[2]),
            np.linspace(-math.pi, math.pi, self.state_bins[3])
        ]
        indices = []
        for i, value in enumerate([x, x_dot, theta, theta_dot]):
            indices.append(np.digitize(value, bins[i]) - 1)
            indices[-1] = max(0, min(len(bins[i])-1, indices[-1]))
        return tuple(indices)

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        predict = self.q_table[state][action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (target - predict)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def draw(env, total_reward):
    screen.fill(WHITE)
    pygame.draw.line(screen, BLACK, (0, env.cart_y), (WIDTH, env.cart_y), 4)
    cart_x = WIDTH / 2 + env.x * PIXELS_PER_METER
    point_a_x = cart_x
    point_a_y = env.cart_y - CART_HEIGHT
    point_b_x = cart_x + POLE_LENGTH * PIXELS_PER_METER * math.sin(env.theta)
    point_b_y = env.cart_y - CART_HEIGHT - POLE_LENGTH * PIXELS_PER_METER * math.cos(env.theta)
    pygame.draw.rect(screen, BLACK, (cart_x - CART_WIDTH // 2, env.cart_y - CART_HEIGHT, CART_WIDTH, CART_HEIGHT))
    pygame.draw.line(screen, RED, (point_a_x, point_a_y), (point_b_x, point_b_y), POLE_WIDTH)
    pygame.draw.circle(screen, PINK, (int(point_a_x), int(point_a_y)), 5)
    pygame.draw.circle(screen, BLUE, (int(point_b_x), int(point_b_y)), 5)
    
    # Render scoreboard
    reward_text = FONT.render(f"Reward: {total_reward:.2f}", True, BLACK)
    screen.blit(reward_text, (10, 10))
    
    pygame.display.flip()

def main():
    clock = pygame.time.Clock()
    env = CartPoleEnv()
    agent = QLearningAgent(n_actions=2, state_bins=(10, 10, 20, 20))
    running = True
    episode = 0
    total_rewards = []
    
    # Spinning phase
    SPIN_DURATION = 5.0
    spin_time = 0.0
    state = env.reset()
    print("Spinning pole for 5 seconds (Point B starts above Point A)...")
    
    while spin_time < SPIN_DURATION and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        action = 1 if int(spin_time * 10) % 2 == 0 else 0
        next_state, _, _, _ = env.step(action)
        state = next_state
        draw(env, 0)  # No reward during spinning
        clock.tick(FPS)
        spin_time += 1 / FPS
    
    if running:
        print("Spinning phase complete. Starting training...")
    
    # Training phase
    while running:
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    done = True
            discrete_state = agent.discretize(state)
            action = agent.choose_action(discrete_state)
            next_state, reward, done, points = env.step(action)
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            draw(env, total_reward)
            clock.tick(FPS)
        
        agent.decay_epsilon()
        total_rewards.append(total_reward)
        episode += 1
        point_a_x, point_a_y, point_b_x, point_b_y = points
        print(f"Episode {episode:3d} | "
              f"Total Reward: {total_reward:6.1f} | "
              f"Epsilon: {agent.epsilon:.3f} | "
              f"Point A: ({point_a_x:.1f}, {point_a_y:.1f}) | "
              f"Point B: ({point_b_x:.1f}, {point_b_y:.1f})")
        
        if len(total_rewards) >= 100 and np.mean(total_rewards[-100:]) >= 195:
            print("Environment solved!")
            break
    
    pygame.quit()

if __name__ == "__main__":
    main()