# main.py
import pygame
import numpy as np
import math

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("CartPole with Pygame")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Clock
clock = pygame.time.Clock()
FPS = 60

# Physics constants
GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
TOTAL_MASS = CART_MASS + POLE_MASS
POLE_LENGTH = 0.5  # actual length in meters
POLE_WIDTH = 10
FORCE_MAG = 10.0
TAU = 0.02  # time interval between state updates

# Scale for rendering (pixels per meter)
PIXELS_PER_METER = 100
TRACK_WIDTH = WIDTH // PIXELS_PER_METER

MAX_ANGLE = 12 * math.pi / 180  # ~12 degrees in radians
MAX_POSITION = (WIDTH // 2) / PIXELS_PER_METER  # Cart can't go beyond screen edges

# Initial state
x = 0.0  # cart position
x_dot = 0.0  # cart velocity
theta = 0  # pole is upright (180 degrees)
theta_dot = 0.0  # angular velocity

def step(force):
    global x, x_dot, theta, theta_dot

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    # For simplicity, treat pole as point mass at end
    temp = (force + POLE_MASS * POLE_LENGTH * theta_dot ** 2 * sintheta) / TOTAL_MASS
    theta_acc = (GRAVITY * sintheta - costheta * temp) / (
        POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * costheta ** 2 / TOTAL_MASS)
    )
    x_acc = temp - POLE_MASS * POLE_LENGTH * theta_acc * costheta / TOTAL_MASS

    # Update state
    x += x_dot * TAU
    x_dot += x_acc * TAU
    theta += theta_dot * TAU
    theta_dot += theta_acc * TAU

    # Simulate basic friction (optional)
    friction = -0.05 * x_dot
    x_dot += friction

def get_reward():
    # Calculate angle difference from upright (0 degrees)
    angle_diff = abs(theta - math.pi)

    # Reward for keeping the pole upright (smaller angle difference is better)
    reward = 1 - (angle_diff / MAX_ANGLE)  # Reward is based on angle difference

    # Penalize for cart not moving (when velocity is low)
    if abs(x_dot) < 0.05:
        reward -= 0.1  # Small penalty for not moving

    # Penalize for falling (pole too tilted or cart out of bounds)
    if abs(angle_diff) > MAX_ANGLE or abs(x) > MAX_POSITION:
        reward = -10  # Big penalty for failure

    return reward


def draw():
    screen.fill(WHITE)

    # Draw ground line
    pygame.draw.line(screen, BLACK, (0, HEIGHT // 2 + 100), (WIDTH, HEIGHT // 2 + 100), 4)

    # Convert cart's x position to pixels
    cart_x = int(WIDTH // 2 + x * PIXELS_PER_METER)
    cart_y = HEIGHT // 2 + 100

    # Draw cart
    cart_width, cart_height = 60, 30
    pygame.draw.rect(screen, BLACK, (cart_x - cart_width // 2, cart_y - cart_height, cart_width, cart_height))

    # Draw pole
    pole_len_px = POLE_LENGTH * PIXELS_PER_METER
    pole_x_end = cart_x + pole_len_px * math.sin(theta)
    pole_y_end = cart_y - cart_height - pole_len_px * math.cos(theta)
    pygame.draw.line(screen, (200, 0, 0), (cart_x, cart_y - cart_height), (pole_x_end, pole_y_end), POLE_WIDTH)

    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Episode: {episode_count}  Score: {total_reward}", True, (0, 0, 0))
    screen.blit(text, (10, 10))

    pygame.display.flip()

def wrap_cart_position():
    global x

    # Convert physical x to screen boundaries in meters
    max_pos = (WIDTH // 2) / PIXELS_PER_METER

    if x * PIXELS_PER_METER > WIDTH // 2:
        x = -max_pos
    elif x * PIXELS_PER_METER < -WIDTH // 2:
        x = max_pos 

# Reward

def get_reward():
    if abs(theta - math.pi) < MAX_ANGLE:
        return 1
    else:
        return -10

def is_done():
    angle_ok = abs(theta - math.pi) < MAX_ANGLE
    position_ok = abs(x) < MAX_POSITION
    return not (angle_ok and position_ok)

# Agent

def discretize_state(state):
    """
    Discretize the continuous state (cart position, cart velocity, pole angle, pole velocity).
    """
    cart_position, cart_velocity, pole_angle, pole_velocity = state

    # Define bins for each state dimension
    position_bins = 10
    velocity_bins = 10
    angle_bins = 10
    angular_velocity_bins = 10

    # Define boundaries for each dimension (the min/max values of the state space)
    position_boundaries = (-2.4, 2.4)  # Cart position range
    velocity_boundaries = (-2, 2)  # Cart velocity range
    angle_boundaries = (-np.pi/2, np.pi/2)  # Pole angle range
    angular_velocity_boundaries = (-2, 2)  # Pole angular velocity range

    # Discretize the state by mapping it to bins
    position_index = np.digitize(cart_position, np.linspace(position_boundaries[0], position_boundaries[1], position_bins)) - 1
    velocity_index = np.digitize(cart_velocity, np.linspace(velocity_boundaries[0], velocity_boundaries[1], velocity_bins)) - 1
    angle_index = np.digitize(pole_angle, np.linspace(angle_boundaries[0], angle_boundaries[1], angle_bins)) - 1
    angular_velocity_index = np.digitize(pole_velocity, np.linspace(angular_velocity_boundaries[0], angular_velocity_boundaries[1], angular_velocity_bins)) - 1

    # Return a tuple that represents the discretized state
    return (position_index, velocity_index, angle_index, angular_velocity_index)


def get_discrete_state():
    """
    Get the current state of the environment in a discretized format.
    """
    return discretize_state((x, x_dot, theta, theta_dot))

class QLearningAgent:
    def __init__(self, action_space_size, state_space_size):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        

        # Initialize Q-table with dimensions (state_space_size x action_space_size)
        self.q_table = np.zeros(state_space_size + (action_space_size,))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(3)  # 0: left, 1: right, 2: brake
        return np.argmax(self.q_table[state])  # Greedy action based on Q-table


    def learn(self, state, action, reward, next_state, done):
        ALPHA = 0.1  # Learning rate
        GAMMA = 0.99  # Discount factor
        
        # Discretize the state and next_state
        state_idx = discretize_state(state)
        next_state_idx = discretize_state(next_state)

        # Q-learning formula
        current_q = self.q_table[state_idx][action]
        next_max_q = np.max(self.q_table[next_state_idx])  # Max Q-value for the next state

        new_q = current_q + ALPHA * (reward + GAMMA * next_max_q - current_q)
        self.q_table[state_idx][action] = new_q



def reset():
    global x, x_dot, theta, theta_dot

    # Reset state to initial conditions
    x = 0.0  # Reset cart position
    x_dot = 0.0  # Reset cart velocity
    theta = np.pi  # Pole is upright (180 degrees)
    theta_dot = 0.0  # Reset angular velocity

    # Return the initial state (for the agent to observe)
    return get_discrete_state()

episode_count = 0
total_reward = 0

def main():

    global total_reward
    global episode_count
    
    state_space_size = (10, 10, 10, 10)  # Based on your discretization
    action_space_size = 3  # For example, left or right

    agent = QLearningAgent(action_space_size, state_space_size)

    running = True
    while running:
        clock.tick(FPS)

        # Handle input
        force = 0
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         running = False
        #     keys = pygame.key.get_pressed()
        #     if keys[pygame.K_LEFT]:
        #         force = -FORCE_MAG
        #     elif keys[pygame.K_RIGHT]:
        #         force = FORCE_MAG
        #     elif keys[pygame.K_SPACE] or keys[pygame.K_DOWN]:
        #         # Apply braking force proportional to opposite of velocity
        #         force = -np.sign(x_dot) * FORCE_MAG * 0.5  # Soft brake

        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     force = -FORCE_MAG
        # elif keys[pygame.K_RIGHT]:
        #     force = FORCE_MAG

        state = get_discrete_state()
        action = agent.choose_action(state)

        if action == 0:
            force = -FORCE_MAG
        elif action == 1:
            force = FORCE_MAG
        else:
            force = -np.sign(x_dot) * FORCE_MAG * 0.5

        # Step the physics
        step(force)
        wrap_cart_position()
        reward = get_reward()
        print(f"Angle: {theta}, Position: {x}, Reward: {reward}")
        done = is_done()
        next_state = get_discrete_state()
        agent.learn(state, action, reward, next_state, done)
        total_reward += reward

        if done:
            print(f"Episode {episode_count} ended. Total reward: {total_reward}")
            episode_count += 1
            #reset()
            total_reward = 0


        # Draw everything
        draw()

    pygame.quit()


if __name__ == "__main__":
    main()
