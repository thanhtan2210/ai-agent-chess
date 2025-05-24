import pygame

# Window Configuration
pygame.init()
info = pygame.display.Info()
WINDOW_WIDTH = 1280  # Fixed width cho layout ổn định
WINDOW_HEIGHT = 800  # Fixed height cho layout ổn định
WINDOW_TITLE = "AI Chess Game"
WINDOW_SIZE = (WINDOW_WIDTH, WINDOW_HEIGHT)

# Layout Configuration
SIDE_PANEL_WIDTH = 340  # Chiều rộng sidebar như chess.com
MARGIN = 28  # Khoảng lề phù hợp

# Board Configuration
BOARD_SIZE = min(
    WINDOW_HEIGHT - 2 * MARGIN,
    WINDOW_WIDTH - SIDE_PANEL_WIDTH - 2 * MARGIN
)
BOARD_OFFSET_X = MARGIN  # Sát lề trái
BOARD_OFFSET_Y = (WINDOW_HEIGHT - BOARD_SIZE) // 2  # Căn giữa dọc
SQUARE_SIZE = BOARD_SIZE // 8

# Colors - Chess.com theme
COLORS = {
    'background': (22, 22, 22),         # Nền tối như chess.com
    'light_square': (238, 238, 210),    # Ô trắng hơi ngả kem
    'dark_square': (125, 146, 98),      # Ô xanh lá nhạt như chess.com
    'highlight': (255, 255, 102, 200),  # Màu highlight vàng rõ hơn
    'selected': (255, 252, 153, 220),   # Màu chọn vàng nhạt
    'valid_move': (119, 151, 86, 180),  # Điểm xanh cho nước đi hợp lệ
    'last_move': (205, 210, 106, 160),  # Highlight nước đi cuối
    'check': (232, 88, 86, 200),        # Đỏ cho tình huống chiếu
    'hover': (255, 255, 255, 25),       # Hover tinh tế hơn
    'shadow': (0, 0, 0, 40),            # Bóng đổ rõ hơn
    'sidebar_bg': (32, 32, 32),         # Nền sidebar tối
    'sidebar_border': (45, 45, 45),     # Viền tinh tế
    'text_primary': (255, 255, 255),    # Text chính màu trắng
    'text_secondary': (170, 170, 170),  # Text phụ màu xám
    'button_bg': (73, 89, 52),          # Nút xanh tối hơn
    'button_hover': (92, 111, 66),      # Hover xanh sáng
    'button_active': (61, 74, 44),      # Active xanh tối
    'piece_white': (255, 255, 255),     # Quân trắng
    'piece_black': (0, 0, 0),           # Quân đen
    'promotion_bg': (0, 0, 0, 180),     # Nền hộp thoại phong cấp
    'time_warning': (255, 81, 81)       # Cảnh báo hết thời gian
}

# Animation Configuration - Match chess.com's smooth animations
ANIMATION = {
    'move_duration': 180,        # Thời gian di chuyển quân dài hơn
    'fade_duration': 150,        # Fade nhanh hơn
    'hover_scale': 1.05,        # Scale nhỏ hơn để tinh tế
    'promotion_duration': 200,   # Animation phong cấp
    'check_pulse_duration': 800, # Hiệu ứng nhấp nháy khi chiếu
    'capture_duration': 250,     # Hiệu ứng bắt quân
    'easing': {
        'move': 'easeOutQuint',      # Di chuyển mượt mà
        'hover': 'easeOutCubic',     # Hover mềm mại
        'bounce': 'easeOutElastic',  # Cho hiệu ứng bật
        'pulse': 'easeInOutSine'     # Cho hiệu ứng nhấp nháy
    },
    'bounce_height': 3
}

# Font Configuration
FONTS = {
    'default': 'Arial',
    'title': 'Arial Black',
    'chess': 'DejaVu Sans',
    'piece': 'DejaVuSans'
}

# Font sizes for different text elements
FONT_SIZES = {
    'title': 48,
    'large': 36,
    'medium': 28,
    'small': 20,
    'tiny': 16
}

# Board coordinate styling
COORDINATES = {
    'font': 'Arial',
    'size': 14,
    'color': (180, 180, 180),
    'margin': 4
}

# UI element styling
UI_STYLE = {
    'button_padding': 12,
    'button_radius': 6,
    'panel_radius': 8,
    'shadow_offset': 2,
    'border_width': 1
}

# Sound settings
SOUNDS = {
    'move': 'src/ui/sounds/move.wav',
    'capture': 'src/ui/sounds/capture.wav',
    'check': 'src/ui/sounds/check.wav',
    'game_start': 'src/ui/sounds/game-start.wav',
    'game_end': 'src/ui/sounds/game-end.wav',
    'promote': 'src/ui/sounds/promote.wav'
}

# Button styles
BUTTON_STYLE = {
    'width': SIDE_PANEL_WIDTH - 40,
    'height': 40,
    'margin': 10,
    'radius': 5,
    'font_size': 18
}

# Dialog styles
DIALOG_STYLE = {
    'width': 300,
    'height': 200,
    'background': (255, 255, 255),
    'border': (200, 200, 200),
    'radius': 10,
    'padding': 20
}

# Move list styles
MOVE_LIST_STYLE = {
    'max_moves': 10,
    'font_size': 14,
    'line_height': 20,
    'padding': 5
}

# Player info styles
PLAYER_INFO_STYLE = {
    'height': 60,
    'padding': 10,
    'font_size': 16,
    'rating_font_size': 14
}

# Settings styles
SETTINGS_STYLE = {
    'width': 300,
    'height': 400,
    'background': (255, 255, 255),
    'border': (200, 200, 200),
    'radius': 10,
    'padding': 20,
    'checkbox_size': 20,
    'slider_width': 200,
    'slider_height': 10
}