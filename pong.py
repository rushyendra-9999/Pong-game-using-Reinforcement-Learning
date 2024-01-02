import pygame
import random
import matplotlib.pyplot as plt
import numpy as np

class GameParameters:
    def __init__(self):
        self.WIDTH, self.HEIGHT = 600, 400
        self.PADDLE_WIDTH, self.PADDLE_HEIGHT = 10, 60
        self.BALL_SIZE = 10
        self.FPS = 60
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

class QLearning:
    def __init__(self, height):
        self.Q = np.zeros((height // 10, 3))  # Q-table
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.2  # Epsilon for exploration-exploitation tradeoff

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2])  # Exploration: random action
        else:
            return np.argmax(self.Q[state])  # Exploitation: choose the best action from Q-table

    def update_Q_values(self, state, action, reward, new_state):
        self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (reward + self.gamma * np.max(self.Q[new_state]))

class PongGame:
    def __init__(self, game_params, q_learning_params):
        pygame.init()
        self.game_params = game_params
        self.q_learning_params = q_learning_params

        # Initialize scores for agents
        self.agent1_score = 0
        self.agent2_score = 0

        # Initialize game elements and variables
        self.ball = pygame.Rect(self.game_params.WIDTH // 2 - self.game_params.BALL_SIZE // 2,
                                self.game_params.HEIGHT // 2 - self.game_params.BALL_SIZE // 2,
                                self.game_params.BALL_SIZE, self.game_params.BALL_SIZE)
        self.paddle1 = pygame.Rect(50, self.game_params.HEIGHT // 2 - self.game_params.PADDLE_HEIGHT // 2,
                                   self.game_params.PADDLE_WIDTH, self.game_params.PADDLE_HEIGHT)
        self.paddle2 = pygame.Rect(self.game_params.WIDTH - 50 - self.game_params.PADDLE_WIDTH,
                                   self.game_params.HEIGHT // 2 - self.game_params.PADDLE_HEIGHT // 2,
                                   self.game_params.PADDLE_WIDTH, self.game_params.PADDLE_HEIGHT)
        self.ball_speed_x = 3 * random.choice((1, -1))
        self.ball_speed_y = 3 * random.choice((1, -1))
        self.paddle1_speed = 3
        self.paddle2_speed = 3

        # Other parameters, episode scores lists, etc.
        self.episode_scores_agent1 = []
        self.episode_scores_agent2 = []

    def get_state(self, paddle_y):
        return min(max(int(paddle_y // 10), 0), self.game_params.HEIGHT // 10 - 1)

    def update_game_elements(self):
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        self.paddle1.y += self.paddle1_speed
        self.paddle2.y += self.paddle2_speed

        self.paddle1.y = max(min(self.paddle1.y, self.game_params.HEIGHT - self.game_params.PADDLE_HEIGHT), 0)
        self.paddle2.y = max(min(self.paddle2.y, self.game_params.HEIGHT - self.game_params.PADDLE_HEIGHT), 0)

        if self.ball.top <= 0 or self.ball.bottom >= self.game_params.HEIGHT:
            self.ball_speed_y *= -1
        if self.ball.left <= 0:
            self.agent2_score += 1
            self.ball_speed_x *= -1
        if self.ball.right >= self.game_params.WIDTH:
            self.agent1_score += 1
            self.ball_speed_x *= -1

        if self.ball.colliderect(self.paddle1) or self.ball.colliderect(self.paddle2):
            self.ball_speed_x *= -1

    def draw_game_elements(self, screen):
        pygame.draw.rect(screen, self.game_params.WHITE, self.paddle1)
        pygame.draw.rect(screen, self.game_params.WHITE, self.paddle2)
        pygame.draw.ellipse(screen, self.game_params.WHITE, self.ball)

        font = pygame.font.Font(None, 36)
        text1 = font.render(f"Agent 1: {self.agent1_score}", True, self.game_params.WHITE)
        text2 = font.render(f"Agent 2: {self.agent2_score}", True, self.game_params.WHITE)
        screen.blit(text1, (50, 50))
        screen.blit(text2, (self.game_params.WIDTH - 200, 50))

    def run_game(self):
        screen = pygame.display.set_mode((self.game_params.WIDTH, self.game_params.HEIGHT))
        pygame.display.set_caption("Pong")
        
        running = True
        episode_count = 0
        sum_score_agent1 = 0
        sum_score_agent2 = 0
        overall_average_scores = []

        while running and episode_count<20000:
            screen.fill(self.game_params.BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            state1 = self.get_state(self.paddle1.y)
            action1 = self.q_learning_params.choose_action(state1)
            reward1 = 0  # Define reward based on game dynamics (e.g., hitting ball, missing ball)
            new_state1 = self.get_state(self.paddle1.y + self.paddle1_speed)
            self.q_learning_params.update_Q_values(state1, action1, reward1, new_state1)

            state2 = self.get_state(self.paddle2.y)
            action2 = self.q_learning_params.choose_action(state2)
            reward2 = 0  # Define reward based on game dynamics (e.g., hitting ball, missing ball)
            new_state2 = self.get_state(self.paddle2.y + self.paddle2_speed)
            self.q_learning_params.update_Q_values(state2, action2, reward2, new_state2)

            self.paddle1_speed = random.choice([-5, 5, 0])
            self.paddle2_speed = random.choice([-5, 5, 0])

            self.update_game_elements()
            self.draw_game_elements(screen)

            pygame.display.flip()
            pygame.time.Clock().tick(self.game_params.FPS)

            # Update episode scores
            sum_score_agent1 += self.agent1_score
            sum_score_agent2 += self.agent2_score
            episode_count += 1

            if episode_count % 10 == 0:  # Calculate average scores every 10 episodes
                overall_average = (sum_score_agent1 + sum_score_agent2) / 20  # Calculate overall average
                overall_average_scores.append(overall_average)  # Store the overall average
                sum_score_agent1 = 0  # Reset sum for Agent 1
                sum_score_agent2 = 0  # Reset sum for Agent 2

        # Plotting the overall average scores
        plt.plot(overall_average_scores, label='Overall Average Scores')
        plt.xlabel('Episodes (Every 10 Episodes)')
        plt.ylabel('Overall Average Score')
        plt.title('Overall Average Scores of Agents vs Episodes')
        plt.legend()
        plt.show()

        print(f"Final Scores - Agent 1: {self.agent1_score}, Agent 2: {self.agent2_score}")

if __name__ == "__main__":
    game_params = GameParameters()
    q_learning_params = QLearning(game_params.HEIGHT)

    game = PongGame(game_params, q_learning_params)
    game.run_game()
    pygame.quit()