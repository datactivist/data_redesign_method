"""
Color palette definitions and CSS custom properties.

Implements the blue-toned palette following Gael Penessot's DataGyver philosophy.
All colors are based on Tailwind CSS slate palette with blue accent.
"""

# Python dictionary for programmatic access
COLORS = {
    # Backgrounds - cool blue-gray tones (slate palette)
    "bg_primary": "#f8fafc",       # Main background (slate-50)
    "bg_secondary": "#f1f5f9",     # Sidebar, cards (slate-100)
    "bg_elevated": "#ffffff",      # Cards, modals (white)

    # Text - deep slate for readability
    "text_primary": "#0f172a",     # Main text (slate-900)
    "text_secondary": "#475569",   # Captions, hints (slate-600)
    "text_muted": "#94a3b8",       # Disabled, placeholders (slate-400)

    # Accent - Klein Blue (International Klein Blue, IKB)
    # Strong visual identity inspired by Yves Klein's iconic pigment
    "accent": "#002fa7",           # Klein Blue - primary actions, links
    "accent_hover": "#001d6e",     # Darker Klein for hover states
    "accent_subtle": "#e6eaf7",    # Very light Klein for backgrounds
    "accent_light": "#f0f2fa",     # Lightest Klein for subtle tints

    # Borders - cool gray
    "border": "#e2e8f0",           # Default borders (slate-200)
    "border_subtle": "#f1f5f9",    # Subtle dividers (slate-100)

    # States
    "success": "#22c55e",          # Success (green-500)
    "success_bg": "#f0fdf4",       # Success background (green-50)
    "warning": "#f59e0b",          # Warning (amber-500)
    "warning_bg": "#fffbeb",       # Warning background (amber-50)
    "error": "#ef4444",            # Error (red-500)
    "error_bg": "#fef2f2",         # Error background (red-50)
}

# CSS custom properties for use in stylesheets
PALETTE_CSS = """
:root {
    --color-bg-primary: #f8fafc;
    --color-bg-secondary: #f1f5f9;
    --color-bg-elevated: #ffffff;
    --color-text-primary: #0f172a;
    --color-text-secondary: #475569;
    --color-text-muted: #94a3b8;
    --color-accent: #002fa7;
    --color-accent-hover: #001d6e;
    --color-accent-light: #f0f2fa;
    --color-border: #e2e8f0;
    --color-success: #22c55e;
    --color-warning: #f59e0b;
    --color-error: #ef4444;
}
"""
