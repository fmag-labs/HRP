"""
Optimization Dashboard Page.

Provides UI for configuring and running hyperparameter optimization
using Optuna, with cross-validation and study management.

This is the importable module used by the dashboard router
(hrp.dashboard.app) and tests; the numbered Streamlit entry
(14_Optimization.py) is a thin wrapper around it.
"""

import streamlit as st
from loguru import logger

from hrp.api.optimization_api import OptimizationAPI
from hrp.api.platform import PlatformAPI
from hrp.dashboard.components.optimization_controls import (
    render_date_range,
    render_feature_selector,
    render_fold_analysis_tab,
    render_folds_slider,
    render_model_selector,
    render_optimization_preview,
    render_results_tab,
    render_sampler_selector,
    render_scoring_selector,
    render_strategy_selector,
    render_study_history_tab,
    render_trials_slider,
)
from hrp.ml.optimization import OptimizationConfig

# Default prediction target when the UI does not expose a target selector.
DEFAULT_TARGET = "returns_5d"


def render_optimization_page(api: PlatformAPI) -> None:
    """Render the Optimization dashboard page.

    Args:
        api: PlatformAPI instance
    """
    st.markdown(
        """
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; font-weight: 700; letter-spacing: -0.03em; margin: 0;">
            Strategy Optimization
        </h1>
        <p style="color: #9ca3af; margin: 0.5rem 0 0 0;">
            Hyperparameter optimization with Optuna and cross-validation
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    opt_api = OptimizationAPI(api)

    # Initialize session state
    st.session_state.setdefault("optimization_result", None)
    st.session_state.setdefault("optimization_config", None)
    st.session_state.setdefault("last_study_id", None)

    # ── Sidebar configuration ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Configuration")

        strategy_name = render_strategy_selector(api)
        model_type = render_model_selector()

        st.divider()
        st.markdown("**Optimization Settings**")
        sampler = render_sampler_selector()
        n_trials = render_trials_slider(default=50, min_val=10, max_val=200)
        n_folds = render_folds_slider(default=5, min_val=3, max_val=10)
        scoring = render_scoring_selector()

        st.divider()
        start_date, end_date = render_date_range()

        st.divider()
        feature_names = render_feature_selector(api, default_features=None)

        st.divider()
        st.markdown("**Actions**")
        if st.button("🔄 Reset Configuration", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith("opt_"):
                    del st.session_state[key]
            st.rerun()

    # ── Validate configuration ───────────────────────────────────────────
    if not strategy_name:
        st.warning("⚠️ No strategies available. Please configure strategies first.")
        return

    if not feature_names:
        st.warning("⚠️ No features selected. Please select at least one feature.")
        return

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        return

    try:
        param_space = opt_api.get_default_param_space(model_type)
    except ValueError as e:
        st.error(f"❌ {e}")
        return

    config = OptimizationConfig(
        hypothesis_id=f"opt_{strategy_name}_{model_type}",
        model_type=model_type,
        target=DEFAULT_TARGET,
        param_space=param_space,
        sampler=sampler,
        n_trials=n_trials,
        n_folds=n_folds,
        scoring_metric=scoring,
        features=feature_names,
        start_date=start_date,
        end_date=end_date,
        enable_pruning=True,
    )
    st.session_state["optimization_config"] = config

    # ── Configuration preview ────────────────────────────────────────────
    st.markdown("### Configuration Preview")
    try:
        preview = opt_api.preview_configuration(config)
        render_optimization_preview(preview)
    except Exception as e:
        logger.error(f"Failed to preview configuration: {e}")
        st.error(f"❌ Failed to generate preview: {e}")
        return

    st.divider()

    # ── Run optimization ─────────────────────────────────────────────────
    if st.button("▶️ Run Optimization", type="primary", use_container_width=True):
        try:
            symbols_result = api.fetchall_readonly(
                """
                SELECT DISTINCT symbol
                FROM prices
                WHERE date >= ? AND date <= ?
                ORDER BY symbol
                LIMIT 100
                """,
                (str(start_date), str(end_date)),
            )
            symbols = [r[0] for r in symbols_result]
            if not symbols:
                st.error("❌ No symbols found for selected date range.")
                return
            st.info(f"🎯 Running optimization on {len(symbols)} symbols...")
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            st.error(f"❌ Failed to fetch symbols: {e}")
            return

        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(current: int, total: int) -> None:
            progress = current / total
            progress_bar.progress(progress)
            progress_text.text(f"Trial {current}/{total} ({int(progress * 100)}%)")

        try:
            with st.spinner("Running optimization..."):
                result = opt_api.run_optimization(
                    config=config,
                    symbols=symbols,
                    progress_callback=progress_callback,
                )
                st.session_state["optimization_result"] = result
                st.session_state["last_study_id"] = getattr(
                    result, "hypothesis_id", None
                )
                progress_text.empty()
                progress_bar.empty()
                st.success(
                    f"✅ Optimization complete! Best score: {result.best_score:.4f}"
                )
                st.session_state["opt_tab"] = "Results"
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            st.error(f"❌ Optimization failed: {e}")
            progress_text.empty()
            progress_bar.empty()
            return

    st.divider()

    # ── Results tabs ─────────────────────────────────────────────────────
    tabs = st.tabs(["📊 Results", "📈 Fold Analysis", "📚 Study History"])

    with tabs[0]:
        if st.session_state["optimization_result"] is not None:
            render_results_tab(st.session_state["optimization_result"])
        else:
            st.info("👆 Configure and run an optimization to see results here.")

    with tabs[1]:
        if st.session_state["optimization_result"] is not None:
            render_fold_analysis_tab(st.session_state["optimization_result"])
        else:
            st.info("👆 Configure and run an optimization to see fold analysis here.")

    with tabs[2]:
        try:
            studies = opt_api.list_studies()
            render_study_history_tab(studies)
        except Exception as e:
            logger.error(f"Failed to list studies: {e}")
            st.error(f"❌ Failed to load study history: {e}")


def main() -> None:
    """Standalone entry point for the Streamlit numbered page."""
    st.set_page_config(page_title="Optimization", page_icon="⚙️", layout="wide")
    api = PlatformAPI(read_only=True)
    render_optimization_page(api)


if __name__ == "__main__":
    main()
