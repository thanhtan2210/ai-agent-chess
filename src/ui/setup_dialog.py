import pygame
from .config import COLORS, FONTS, FONT_SIZES

def select_difficulty_and_color():
    """Hiển thị màn hình chọn độ khó, màu quân và chế độ AI vs AI với giao diện hiện đại, dễ nhìn."""
    pygame.init()
    screen = pygame.display.set_mode((600, 800))
    pygame.display.set_caption("Chess Game Setup")
    # Fonts
    title_font = pygame.font.SysFont(FONTS['default'], 52, bold=True)
    label_font = pygame.font.SysFont(FONTS['default'], 30)
    button_font = pygame.font.SysFont(FONTS['default'], 36, bold=True)
    tip_font = pygame.font.SysFont(FONTS['default'], 18)
    running = True
    selected_depth = 2
    selected_color = 0  # 0: White, 1: Black
    ai_vs_ai = False
    # Button dimensions
    button_width = 80
    button_height = 68
    button_margin = 28
    color_button_width = 200
    color_button_height = 62
    max_depth = 4  # Giới hạn tối đa depth là 4 để tránh overload
    agent_options = [
        ("Minimax", "MinimaxAgent"),
        ("AlphaBeta", "AlphaBetaAgent"),
        ("MCTS", "MCTSAgent"),
        ("DeepLearning", "DeepLearningAgent"),
        ("Random", "RandomAgent")
    ]
    agent_white_idx = 0
    agent_black_idx = 0
    dropdown_open = None  # None, 'white', 'black'
    while running:
        screen.fill(COLORS['background'])
        # Draw title
        title = title_font.render("Chess Game Setup", True, COLORS['text_primary'])
        title_rect = title.get_rect(centerx=300, y=36)
        screen.blit(title, title_rect)
        # Draw difficulty selection
        diff_label = label_font.render("Select Difficulty", True, COLORS['text_primary'])
        diff_label_rect = diff_label.get_rect(centerx=300, y=120)
        screen.blit(diff_label, diff_label_rect)
        # Draw difficulty buttons (centered)
        total_width = max_depth * button_width + (max_depth-1) * button_margin
        start_x = (600 - total_width) // 2
        for i in range(1, max_depth+1):
            button_x = start_x + (i-1)*(button_width + button_margin)
            button_rect = pygame.Rect(button_x, 170, button_width, button_height)
            color = COLORS['button_active'] if i == selected_depth else COLORS['button_bg']
            # Hiệu ứng hover
            if button_rect.collidepoint(pygame.mouse.get_pos()):
                color = COLORS['button_hover']
            pygame.draw.rect(screen, color, button_rect, border_radius=24)
            # Border cho nút được chọn
            if i == selected_depth:
                pygame.draw.rect(screen, (80, 220, 120), button_rect, 6, border_radius=24)
            text = button_font.render(str(i), True, COLORS['text_primary'])
            text_rect = text.get_rect(center=button_rect.center)
            screen.blit(text, text_rect)
        # Draw color selection
        color_label = label_font.render("Choose Your Color", True, COLORS['text_primary'])
        color_label_rect = color_label.get_rect(centerx=300, y=270)
        screen.blit(color_label, color_label_rect)
        # Draw color buttons (centered)
        white_rect = pygame.Rect(70, 320, color_button_width, color_button_height)
        black_rect = pygame.Rect(330, 320, color_button_width, color_button_height)
        white_color = COLORS['light_square'] if selected_color == 0 and not ai_vs_ai else COLORS['button_bg']
        black_color = COLORS['dark_square'] if selected_color == 1 and not ai_vs_ai else COLORS['button_bg']
        # Hiệu ứng hover
        if white_rect.collidepoint(pygame.mouse.get_pos()) and not ai_vs_ai:
            white_color = COLORS['button_hover']
        if black_rect.collidepoint(pygame.mouse.get_pos()) and not ai_vs_ai:
            black_color = COLORS['button_hover']
        pygame.draw.rect(screen, white_color, white_rect, border_radius=24)
        pygame.draw.rect(screen, black_color, black_rect, border_radius=24)
        # Border/shadow cho nút được chọn
        if selected_color == 0 and not ai_vs_ai:
            pygame.draw.rect(screen, (220,220,120), white_rect, 6, border_radius=24)
        if selected_color == 1 and not ai_vs_ai:
            pygame.draw.rect(screen, (120,220,220), black_rect, 6, border_radius=24)
        white_text = button_font.render("White", True, COLORS['text_primary'])
        black_text = button_font.render("Black", True, COLORS['text_primary'])
        screen.blit(white_text, white_text.get_rect(center=white_rect.center))
        screen.blit(black_text, black_text.get_rect(center=black_rect.center))
        # Draw agent selection group (centered, không có chữ vs)
        agent_group_y = 370
        agent_group_width = 500
        agent_group_x = (600 - agent_group_width) // 2
        dropdown_width = 180
        dropdown_height = 44
        agent_label_font = pygame.font.SysFont(FONTS['default'], 26, bold=True)
        # White agent dropdown
        white_enabled = ai_vs_ai or (selected_color == 1)
        black_enabled = ai_vs_ai or (selected_color == 0)
        white_dropdown_rect = pygame.Rect(agent_group_x, agent_group_y + 32, dropdown_width, dropdown_height)
        black_dropdown_rect = pygame.Rect(agent_group_x + agent_group_width - dropdown_width, agent_group_y + 32, dropdown_width, dropdown_height)
        # Draw white label and dropdown
        white_label = agent_label_font.render("White Agent", True, COLORS['text_primary'])
        screen.blit(white_label, (white_dropdown_rect.x + 20, white_dropdown_rect.y - 32))
        pygame.draw.rect(screen, (255,255,255) if white_enabled else (200,200,200), white_dropdown_rect, border_radius=16)
        pygame.draw.rect(screen, (180,200,255), white_dropdown_rect, 2, border_radius=16)
        agent_white_text = agent_label_font.render(agent_options[agent_white_idx][0], True, (40,40,40))
        screen.blit(agent_white_text, agent_white_text.get_rect(center=white_dropdown_rect.center))
        # Draw black label and dropdown
        black_label = agent_label_font.render("Black Agent", True, COLORS['text_primary'])
        screen.blit(black_label, (black_dropdown_rect.x + 20, black_dropdown_rect.y - 32))
        pygame.draw.rect(screen, (255,255,255) if black_enabled else (200,200,200), black_dropdown_rect, border_radius=16)
        pygame.draw.rect(screen, (180,200,255), black_dropdown_rect, 2, border_radius=16)
        agent_black_text = agent_label_font.render(agent_options[agent_black_idx][0], True, (40,40,40))
        screen.blit(agent_black_text, agent_black_text.get_rect(center=black_dropdown_rect.center))
        # Draw dropdown options if open
        if dropdown_open == 'white' and white_enabled:
            for i, (name, _) in enumerate(agent_options):
                opt_rect = pygame.Rect(white_dropdown_rect.x, white_dropdown_rect.y + (i+1)*dropdown_height, dropdown_width, dropdown_height)
                color = (180,220,255) if i == agent_white_idx else (240,240,255)
                pygame.draw.rect(screen, color, opt_rect, border_radius=12)
                pygame.draw.rect(screen, (120,180,255), opt_rect, 2, border_radius=12)
                opt_text = agent_label_font.render(name, True, (40,40,40))
                screen.blit(opt_text, opt_text.get_rect(center=opt_rect.center))
        if dropdown_open == 'black' and black_enabled:
            for i, (name, _) in enumerate(agent_options):
                opt_rect = pygame.Rect(black_dropdown_rect.x, black_dropdown_rect.y + (i+1)*dropdown_height, dropdown_width, dropdown_height)
                color = (180,220,255) if i == agent_black_idx else (240,240,255)
                pygame.draw.rect(screen, color, opt_rect, border_radius=12)
                pygame.draw.rect(screen, (120,180,255), opt_rect, 2, border_radius=12)
                opt_text = agent_label_font.render(name, True, (40,40,40))
                screen.blit(opt_text, opt_text.get_rect(center=opt_rect.center))
        # Draw AI vs AI toggle (centered, y=440)
        ai_vs_ai_rect = pygame.Rect(180, 440, 240, 62)
        ai_vs_ai_color = (100, 180, 255) if ai_vs_ai else (120, 160, 120)
        if ai_vs_ai_rect.collidepoint(pygame.mouse.get_pos()):
            ai_vs_ai_color = (140, 200, 255) if ai_vs_ai else COLORS['button_hover']
        pygame.draw.rect(screen, ai_vs_ai_color, ai_vs_ai_rect, border_radius=24)
        if ai_vs_ai:
            pygame.draw.rect(screen, (80,180,255), ai_vs_ai_rect, 6, border_radius=24)
        ai_vs_ai_text = button_font.render("AI vs AI", True, COLORS['text_primary'])
        screen.blit(ai_vs_ai_text, ai_vs_ai_text.get_rect(center=ai_vs_ai_rect.center))
        # Draw Start Game button (centered, y=510)
        start_rect = pygame.Rect(180, 510, 240, 62)
        start_color = (80, 160, 255) if start_rect.collidepoint(pygame.mouse.get_pos()) else (60, 130, 220)
        pygame.draw.rect(screen, start_color, start_rect, border_radius=24)
        pygame.draw.rect(screen, (180, 220, 255), start_rect, 6, border_radius=24)
        start_text = button_font.render("Start Game", True, (255,255,255))
        screen.blit(start_text, start_text.get_rect(center=start_rect.center))
        # Tip nhỏ dưới cùng
        tip = tip_font.render("Tip: Choose AI vs AI to see two AI play against each other!", True, (180,180,180))
        tip_rect = tip.get_rect(centerx=300, y=570)
        screen.blit(tip, tip_rect)
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys; sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                # Check difficulty buttons
                for i in range(1, max_depth+1):
                    button_x = start_x + (i-1)*(button_width + button_margin)
                    button_rect = pygame.Rect(button_x, 170, button_width, button_height)
                    if button_rect.collidepoint(mouse_pos):
                        selected_depth = i
                # Check color buttons
                if white_rect.collidepoint(mouse_pos) and not ai_vs_ai:
                    selected_color = 0
                elif black_rect.collidepoint(mouse_pos) and not ai_vs_ai:
                    selected_color = 1
                # Check AI vs AI toggle
                if ai_vs_ai_rect.collidepoint(mouse_pos):
                    ai_vs_ai = not ai_vs_ai
                # Check Start Game button
                if start_rect.collidepoint(mouse_pos):
                    running = False
                # Dropdown logic
                if white_dropdown_rect.collidepoint(mouse_pos) and white_enabled:
                    dropdown_open = 'white' if dropdown_open != 'white' else None
                elif black_dropdown_rect.collidepoint(mouse_pos) and black_enabled:
                    dropdown_open = 'black' if dropdown_open != 'black' else None
                elif dropdown_open == 'white' and white_enabled:
                    for i in range(len(agent_options)):
                        opt_rect = pygame.Rect(white_dropdown_rect.x, white_dropdown_rect.y + (i+1)*dropdown_height, dropdown_width, dropdown_height)
                        if opt_rect.collidepoint(mouse_pos):
                            agent_white_idx = i
                            dropdown_open = None
                elif dropdown_open == 'black' and black_enabled:
                    for i in range(len(agent_options)):
                        opt_rect = pygame.Rect(black_dropdown_rect.x, black_dropdown_rect.y + (i+1)*dropdown_height, dropdown_width, dropdown_height)
                        if opt_rect.collidepoint(mouse_pos):
                            agent_black_idx = i
                            dropdown_open = None
                else:
                    dropdown_open = None
                # Nếu vừa tắt AI vs AI, reset agent phía Player về mặc định
                if not ai_vs_ai:
                    if selected_color == 0:
                        agent_white_idx = 0  # Player là trắng, reset white agent
                    else:
                        agent_black_idx = 0  # Player là đen, reset black agent
        pygame.display.flip()
    agent_white_name = agent_options[agent_white_idx][1] if (ai_vs_ai or selected_color == 1) else None
    agent_black_name = agent_options[agent_black_idx][1] if (ai_vs_ai or selected_color == 0) else None
    return selected_depth, selected_color, ai_vs_ai, agent_white_name, agent_black_name 