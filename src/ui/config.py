# Window Configuration
WINDOW_TITLE = "AI Chess Game"
WINDOW_SIZE = (1024, 768)  # Width, Height
WINDOW_BG = (40, 44, 52)

# Board Configuration 
BOARD_SIZE = 600
BOARD_OFFSET_X = 50  # Left padding
BOARD_OFFSET_Y = 50  # Top padding
SQUARE_SIZE = BOARD_SIZE // 8

# Colors
COLORS = {
    'background': (40, 44, 52),
    'light_square': (238, 238, 210),  # Light cream
    'dark_square': (118, 150, 86),    # Forest green
    'highlight': (186, 202, 68),      # Move highlight
    'selected': (246, 246, 105),      # Selected piece
    'valid_move': (119, 199, 149),    # Valid move indicator
    'last_move': (205, 210, 106),     # Last move highlight
    'check': (219, 83, 83)           # Check highlight
}

# UI Elements
FONTS = {
    'default': 'Arial',
    'chess': 'Chess7',  # For chess symbols
}

FONT_SIZES = {
    'large': 32,
    'medium': 24,
    'small': 16,
    'tiny': 12
}

# Button Configuration
BUTTON_COLOR = (70, 136, 241)
BUTTON_HOVER_COLOR = (60, 116, 221)
BUTTON_TEXT_COLOR = (255, 255, 255)
BUTTON_BORDER_COLOR = (50, 106, 201)  # Slightly darker than button color
BUTTON_STYLES = {
    'default': {
        'bg_color': BUTTON_COLOR,
        'hover_color': BUTTON_HOVER_COLOR,
        'text_color': BUTTON_TEXT_COLOR,
        'border_color': BUTTON_BORDER_COLOR,
        'padding': 10,
        'border_radius': 5,
        'width': 160,
        'height': 40
    },
    'small': {
        'bg_color': BUTTON_COLOR,
        'hover_color': BUTTON_HOVER_COLOR,
        'text_color': BUTTON_TEXT_COLOR,
        'border_color': BUTTON_BORDER_COLOR,
        'padding': 5,
        'border_radius': 3,
        'width': 100,
        'height': 30
    }
}

# Side Panel Configuration
SIDE_PANEL_WIDTH = 300
PANEL = {
    'width': SIDE_PANEL_WIDTH,
    'padding': 20,
    'bg_color': (45, 49, 58),
    'text_color': (200, 200, 200),
    'border_color': (60, 65, 75),
    'hover_color': (55, 59, 68)
}

# Animation Configuration
ANIMATIONS = {
    'move_duration': 0.2,
    'fade_duration': 0.3,
    'popup_duration': 0.5
}

# Game Info Display
INFO_DISPLAY = {
    'height': 100,
    'padding': 10,
    'bg_color': (45, 49, 58),
    'text_color': (200, 200, 200),
    'border_color': (60, 65, 75)
}

# Piece Configuration
PIECE_STYLES = {
    'default': {
        'scale': 0.9,  # Piece size relative to square
        'shadow': True,
        'shadow_offset': (2, 2)
    }
}

# Clock Configuration
CLOCK = {
    'bg_color': (45, 49, 58),
    'text_color': (200, 200, 200),
    'warning_color': (219, 83, 83),
    'height': 60,
    'width': 120
}

# Menu Configuration
MENU = {
    'width': 400,
    'height': 60,
    'button_height': 50,
    'button_width': 200,
    'button_margin': 10,
    'bg_color': (45, 49, 58),
    'border_color': (60, 65, 75),
    'text_color': (200, 200, 200),
    'hover_color': (55, 59, 68)
}