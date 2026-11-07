"""
UI Components for SAFE-Cascade Demo

Reusable Streamlit components for the demo interface.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional
from PIL import Image


def render_input_panel(example_data: Dict, image_path: Optional[str] = None, show_gold: bool = False):
    """Render input panel with chart and question."""
    st.header("📊 Input")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Chart Image")
        if image_path:
            try:
                img = Image.open(image_path)
                st.image(img, use_container_width=True)
            except:
                st.warning("⚠️ Image file not found")
                st.info(f"Expected: {image_path}")
        else:
            st.info("📷 No image available for this example")
    
    with col2:
        st.subheader("Question")
        question = example_data.get('question', example_data.get('query', 'N/A'))
        st.markdown(f"**{question}**")
        
        # Gold answer with toggle
        if show_gold:
            st.subheader("Gold Answer — evaluation only, not used by system")
            st.caption("⚠️ This is the reference answer for evaluation purposes only")
            answers = example_data.get('answers', [example_data.get('answer', 'N/A')])
            if not isinstance(answers, list):
                answers = [answers]
            for ans in answers:
                st.code(ans, language=None)


def render_ocr_panel(example_data: Dict, text_prediction: str = '', expanded: bool = True):
    """Render OCR evidence panel with optional highlighting."""
    st.header("🔍 OCR Evidence")
    
    ocr_text = example_data.get('ocr_text', 'N/A')
    question = example_data.get('question', '')
    
    # Show OCR with highlighting hints
    if len(ocr_text) > 500:
        with st.expander("View OCR Text", expanded=expanded):
            st.text_area("", ocr_text, height=200, label_visibility="collapsed")
            if text_prediction:
                st.caption(f"💡 Text prediction '{text_prediction}' may appear in OCR above")
    else:
        st.text_area("", ocr_text, height=150, label_visibility="collapsed")
        if text_prediction:
            st.caption(f"💡 Text prediction '{text_prediction}' may appear in OCR above")
    
    st.caption(f"Model: **Azure OCR**")
    if ocr_text != 'N/A':
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Character Count", len(ocr_text))
        with col2:
            st.metric("Token Count", len(ocr_text.split()))


def render_text_answer_panel(text_run: Dict, costs: Dict, latency: Dict):
    """Render text-only answer panel."""
    st.header("📝 Text-Only Answer")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Prediction")
        prediction = text_run.get('prediction', 'N/A')
        st.code(prediction, language=None)
        
        st.caption(f"Model: **gpt-5-mini**")
    
    with col2:
        st.metric("Cost", f"${costs['text_cost']:.6f}")
        st.caption("per example")
    
    with col3:
        st.metric("Latency", f"{latency['text_latency']}ms")
        st.caption("⚠️ estimated")


def render_router_panel(router_pred: Dict, threshold: float, decision: str, explanation: str):
    """Render router decision panel."""
    st.header("🤖 Router Decision")
    
    probability = router_pred.get('probability', 0.0)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("Escalation Probability")
        st.progress(probability)
        # Improved probability wording
        if probability < 0.3:
            st.caption(f"**{probability:.2f}** — Low probability of needing VLM")
        elif probability > 0.7:
            st.caption(f"**{probability:.2f}** — High probability of needing VLM")
        else:
            st.caption(f"**{probability:.2f}** — Escalation probability")
        
        st.subheader("Decision")
        if decision == "ESCALATE_VLM":
            st.error(f"🔴 **{decision}**")
        else:
            st.success(f"✅ **{decision}**")
        
        st.info(explanation)
    
    with col2:
        st.metric("Threshold (θ)", f"{threshold:.2f}")
        st.caption("Current setting")
    
    with col3:
        st.metric("Model", "RandomForest")
        st.caption("Trained on 1,750 examples")


def render_threshold_slider():
    """Render threshold slider."""
    threshold = st.slider(
        "Adjust Threshold (θ)",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        format="%.2f",
        help="Lower threshold = more VLM calls, higher accuracy. Higher threshold = fewer VLM calls, lower cost."
    )
    return threshold


def render_vlm_answer_panel(vlm_run: Dict, text_run: Dict, escalated: bool, costs: Dict, latency: Dict):
    """Render VLM/final answer panel."""
    st.header("🎯 Final Answer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if escalated:
            st.subheader("VLM Prediction")
            prediction = vlm_run.get('prediction', 'N/A')
            st.code(prediction, language=None)
            st.caption(f"Model: **gemini-2.5-flash-image**")
        else:
            st.subheader("Final Answer (Text-Only)")
            prediction = text_run.get('prediction', 'N/A')
            st.code(prediction, language=None)
            st.caption("✅ Stopped at text-only. VLM not invoked.")
            st.caption(f"Model: **gpt-5-mini**")
    
    with col2:
        st.metric("Total Cost", f"${costs['total_cost']:.6f}")
        if costs['savings_pct'] > 0:
            st.metric("Savings vs Full VLM", f"{costs['savings_pct']:.1f}%", delta=f"${costs['savings']:.6f}")
        
        st.metric("Total Latency", f"{latency['total_latency']}ms")
        st.caption("⚠️ estimated")


def render_operating_frontier(threshold_sweep: pd.DataFrame, current_threshold: float):
    """Render operating frontier panel."""
    st.header("📈 Operating Frontier")
    
    # Create plot
    fig = go.Figure()
    
    # Add accuracy vs VLM rate curve
    fig.add_trace(go.Scatter(
        x=threshold_sweep['vlm_invocation_rate'],
        y=threshold_sweep['unified_accuracy'],
        mode='lines+markers',
        name='SAFE-Cascade',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=10)
    ))
    
    # Highlight current threshold
    current_row = threshold_sweep.iloc[(threshold_sweep['threshold'] - current_threshold).abs().argsort()[:1]]
    fig.add_trace(go.Scatter(
        x=current_row['vlm_invocation_rate'],
        y=current_row['unified_accuracy'],
        mode='markers',
        name=f'Current (θ={current_threshold:.2f})',
        marker=dict(size=20, color='red', symbol='star')
    ))
    
    # Add full VLM baseline
    fig.add_trace(go.Scatter(
        x=[100],
        y=[67.73],
        mode='markers',
        name='Full VLM Baseline',
        marker=dict(size=15, color='#d62728', symbol='triangle-up')
    ))
    
    fig.update_layout(
        title="Accuracy vs VLM Invocation Rate",
        xaxis_title="VLM Invocation Rate (%)",
        yaxis_title="Accuracy (%)",
        hovermode='closest',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Threshold table
    st.subheader("Threshold Performance")
    display_df = threshold_sweep[['threshold', 'unified_accuracy', 'vlm_invocation_rate', 'missed_escalation_rate']].copy()
    display_df.columns = ['Threshold (θ)', 'Accuracy (%)', 'VLM Rate (%)', 'Missed Escalation (%)']
    
    # Find highest accuracy before formatting
    max_accuracy = threshold_sweep['unified_accuracy'].max()
    
    # Format values properly
    display_df['Threshold (θ)'] = display_df['Threshold (θ)'].apply(lambda x: f"{x:.2f}")
    display_df['Accuracy (%)'] = display_df['Accuracy (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['VLM Rate (%)'] = display_df['VLM Rate (%)'].apply(lambda x: f"{x:.1f}%")
    display_df['Missed Escalation (%)'] = display_df['Missed Escalation (%)'].apply(lambda x: f"{x:.1f}%")
    
    # Highlight current threshold and highest accuracy
    def highlight_row(row):
        threshold_val = float(row['Threshold (θ)'])
        accuracy_val = float(row['Accuracy (%)'].rstrip('%'))
        
        # Current threshold - green
        if abs(threshold_val - current_threshold) < 0.01:
            return ['background-color: #90EE90; color: #000000; font-weight: bold'] * len(row)
        # Highest accuracy - light blue
        elif abs(accuracy_val - max_accuracy) < 0.1:
            return ['background-color: #ADD8E6; color: #000000; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Legend
    col1, col2 = st.columns(2)
    with col1:
        st.caption("🟢 Green = Current threshold")
    with col2:
        st.caption("🔵 Blue = Highest accuracy")


def render_demo_script_panel():
    """Render demo script panel with preset examples."""
    st.header("🎬 Demo Script")
    
    st.markdown("""
    **3-Minute Video Script:**
    
    1. **Introduction (30s):** Explain cost-adaptive routing for chart QA
    2. **Text-Solvable Example (45s):** Show low probability, text answer correct
    3. **VLM-Needed Example (45s):** Show high probability, VLM required
    4. **Threshold Adjustment (30s):** Move slider, show tradeoff
    5. **Summary (30s):** 69.1% accuracy, 73.1% VLM rate, 26.9% fewer calls
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📗 Load Text-Solvable", use_container_width=True):
            st.session_state.demo_mode = 'text_solvable'
            st.info("**Presenter Note:** Highlight that router correctly identifies simple questions that don't need visual reasoning.")
    
    with col2:
        if st.button("📕 Load VLM-Needed", use_container_width=True):
            st.session_state.demo_mode = 'vlm_needed'
            st.info("**Presenter Note:** Show that router escalates when chart analysis is required.")
    
    with col3:
        if st.button("📙 Load Borderline", use_container_width=True):
            st.session_state.demo_mode = 'borderline'
            st.info("**Presenter Note:** Demonstrate threshold sensitivity - small changes affect routing.")


def render_key_metrics_sidebar():
    """Render key metrics in sidebar."""
    st.sidebar.header("📊 System Performance")
    
    st.sidebar.metric("Router Accuracy", "69.1%", help="Test set N=375")
    st.sidebar.metric("VLM Baseline", "67.7%", delta="+1.4pp")
    st.sidebar.metric("VLM Invocation Rate", "73.1%", delta="-26.9pp vs baseline")
    st.sidebar.metric("Cost Savings", "9.5%", help="vs full VLM")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ **Note:** Latency values are estimated from typical API response times.")
    st.sidebar.caption("🔒 **Mode:** Cache-only (no live API calls)")
