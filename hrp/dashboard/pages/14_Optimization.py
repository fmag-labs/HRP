"""
Optimization Dashboard Page (Streamlit numbered entry).

Thin wrapper around hrp.dashboard.pages.optimization, which holds the actual
implementation so it can be imported by the dashboard router and tests.
"""

from hrp.dashboard.pages.optimization import main, render_optimization_page

__all__ = ["render_optimization_page", "main"]


if __name__ == "__main__":
    main()
