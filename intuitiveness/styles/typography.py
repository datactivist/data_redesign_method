"""
Typography tokens and font loading CSS.

Implements IBM Plex Sans from Google Fonts with system font fallbacks.
Based on css-contracts.md TYPOGRAPHY_CSS specification.
"""

# Python dictionary for programmatic access
TYPOGRAPHY = {
    "font_family": "'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_family_mono": "'IBM Plex Mono', 'Menlo', 'Monaco', monospace",

    # Font sizes (rem)
    "text_xs": "0.75rem",    # 12px
    "text_sm": "0.875rem",   # 14px
    "text_base": "1rem",     # 16px
    "text_lg": "1.125rem",   # 18px
    "text_xl": "1.25rem",    # 20px
    "text_2xl": "1.5rem",    # 24px
    "text_3xl": "1.875rem",  # 30px

    # Font weights
    "font_normal": "400",
    "font_medium": "500",
    "font_semibold": "600",

    # Line heights
    "leading_tight": "1.25",
    "leading_normal": "1.5",
    "leading_relaxed": "1.625",

    # Letter spacing
    "tracking_tight": "-0.02em",
    "tracking_normal": "0",
    "tracking_wide": "0.05em",
}

# CSS for font loading and typography styles
TYPOGRAPHY_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

h1 {
    font-weight: 600;
    letter-spacing: -0.02em;
}

h2, h3 {
    font-weight: 500;
}

.stCaption {
    color: #57534e;
    font-size: 0.875rem;
}
"""
