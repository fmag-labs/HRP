"""Agents Monitor - Real-time agent status and historical timeline."""

from __future__ import annotations

import streamlit as st

from hrp.dashboard.agents_monitor import get_all_agent_status, AgentStatus
from hrp.api.platform import PlatformAPI


st.title("🤖 Agents Monitor")

# Page controls
col1, col2 = st.columns([3, 1])
with col1:
    auto_refresh = st.checkbox("Auto-refresh", value=True)
with col2:
    if st.button("Refresh Now"):
        st.rerun()

# Real-time Monitor Section
st.subheader("Real-Time Monitor")
st.info("Loading agent status...")

# Historical Timeline Section (placeholder)
st.markdown("---")
st.subheader("Historical Timeline")
st.info("Timeline view coming soon...")
