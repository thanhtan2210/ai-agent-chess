import pygame
import chess
from ui.game_ui import GameUI

def select_difficulty():
    pygame.init()
    screen = pygame.display.set_mode((400, 60 + 60 * 9))  # Increased height for new option
    pygame.display.set_caption("Select AI Level")
    font = pygame.font.SysFont("Arial", 32)
    running = True
    level = None
    buttons = []
    for i in range(7):  # 0: Random, 1-6: Minimax depth
        y = 30 + i * 60
        text = "Random" if i == 0 else f"Minimax depth={i}"
        buttons.append({"rect": pygame.Rect(100, y, 200, 50), "text": text, "level": i})
    # Add Agent vs Agent button
    buttons.append({"rect": pygame.Rect(100, 30 + 7 * 60, 200, 50), "text": "Agent vs Agent", "level": "agent_vs_agent"})
    # Add Deep Learning button
    buttons.append({"rect": pygame.Rect(100, 30 + 8 * 60, 200, 50), "text": "Deep Learning", "level": "deep_learning"})
    while running:
        screen.fill((220, 220, 220))
        for btn in buttons:
            pygame.draw.rect(screen, (100, 100, 200), btn["rect"])
            text = font.render(btn["text"], True, (255, 255, 255))
            text_rect = text.get_rect(center=btn["rect"].center)
            screen.blit(text, text_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn["rect"].collidepoint(event.pos):
                        level = btn["level"]
                        running = False
        pygame.display.flip()
    pygame.quit()
    return level

def select_player_color():
    pygame.init()
    screen = pygame.display.set_mode((400, 200))
    pygame.display.set_caption("Select Your Color")
    font = pygame.font.SysFont("Arial", 32)
    running = True
    color = None
    buttons = [
        {"rect": pygame.Rect(60, 60, 120, 60), "text": "White", "color": chess.WHITE},
        {"rect": pygame.Rect(220, 60, 120, 60), "text": "Black", "color": chess.BLACK},
    ]
    while running:
        screen.fill((220, 220, 220))
        for btn in buttons:
            pygame.draw.rect(screen, (100, 100, 200), btn["rect"])
            text = font.render(btn["text"], True, (255, 255, 255))
            text_rect = text.get_rect(center=btn["rect"].center)
            screen.blit(text, text_rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                for btn in buttons:
                    if btn["rect"].collidepoint(event.pos):
                        color = btn["color"]
                        running = False
        pygame.display.flip()
    pygame.quit()
    return color

def main():
    level = select_difficulty()
    if level == "agent_vs_agent":
        agent_type = "agent_vs_agent"
        player_color = None
    elif level == "deep_learning":
        agent_type = "deep_learning"
        player_color = select_player_color()
    elif level == 0:
        agent_type = "random"
        player_color = select_player_color()
    else:
        agent_type = f"minimax{level}"
        player_color = select_player_color()
    game = GameUI(agent_type=agent_type, player_color=player_color)
    game.run()

if __name__ == "__main__":
    main() 