# Additional UI components for CIKM demo improvements

import streamlit as st
from typing import Dict, Optional


def render_routing_signals_panel(example_data: Dict, router_pred: Dict, text_run: Dict, threshold: float, decision: str):
    """Render routing signals panel showing features used by router."""
    st.header("🔍 Routing Signals")
    
    probability = router_pred.get('probability', 0.0)
    text_pred = text_run.get('prediction', '')
    ocr_text = example_data.get('ocr_text', '')
    question = example_data.get('question', '')
    
    # Calculate signals
    text_in_ocr = text_pred.lower() in ocr_text.lower() if text_pred and ocr_text else False
    is_numeric = text_pred.replace('.', '').replace(',', '').replace('%', '').replace('$', '').strip().replace('-', '').isdigit() if text_pred else False
    ocr_token_count = len(ocr_text.split()) if ocr_text else 0
    
    # Question type flags (simple heuristics)
    question_lower = question.lower()
    is_comparison = any(word in question_lower for word in ['compare', 'difference', 'more', 'less', 'higher', 'lower'])
    is_superlative = any(word in question_lower for word in ['most', 'least', 'highest', 'lowest', 'maximum', 'minimum'])
    is_visual = any(word in question_lower for word in ['trend', 'pattern', 'shape', 'color', 'visual', 'chart'])
    
    with st.expander("View Routing Signals", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Escalation Probability", f"{probability:.2f}")
            st.metric("Threshold (θ)", f"{threshold:.2f}")
            st.metric("OCR Token Count", ocr_token_count)
            
        with col2:
            st.metric("Text Answer in OCR", "✅" if text_in_ocr else "❌")
            st.metric("Text Answer is Numeric", "✅" if is_numeric else "❌")
            st.metric("Question Type", 
                     "Visual" if is_visual else "Comparison" if is_comparison else "Superlative" if is_superlative else "Numeric")
        
        # Explanation
        st.markdown("**Routing Explanation:**")
        if decision == "STOP_TEXT":
            st.success(f"Stopped because the text answer appears reliable (prob={probability:.2f} < threshold={threshold:.2f}) and escalation probability is below threshold.")
        else:
            st.warning(f"Escalated because visual/chart reasoning appears likely (prob={probability:.2f} ≥ threshold={threshold:.2f}) and escalation probability exceeds threshold.")


def render_presenter_notes(category: str):
    """Render presenter notes for demo category."""
    notes = {
        'text_solvable': """
        **📗 Presenter Note: Text-Solvable Case**
        
        This case is answered from OCR/text, so SAFE-Cascade avoids a VLM call.
        The router correctly identifies that the question can be answered using only the extracted text,
        saving cost without sacrificing accuracy.
        """,
        'vlm_needed': """
        **📕 Presenter Note: VLM-Needed Case**
        
        This case needs visual grounding; the router escalates and the VLM supplies the final answer.
        The text-only model cannot reliably answer this question, so the system escalates to ensure accuracy.
        """,
        'borderline': """
        **📙 Presenter Note: Borderline Case**
        
        This case is near the threshold; moving θ changes whether the system stops or escalates.
        This demonstrates how the threshold parameter controls the accuracy-cost tradeoff.
        Adjust the slider to see the routing decision change in real-time.
        """
    }
    
    note = notes.get(category, "")
    if note:
        st.info(note)


def render_vlm_baseline_answer(vlm_run: Dict, escalated: bool):
    """Render VLM baseline answer in collapsible section."""
    vlm_prediction = vlm_run.get('prediction', 'N/A')
    
    with st.expander("VLM Baseline Answer (for comparison)", expanded=escalated):
        st.caption("This is what the VLM would predict if called (for evaluation purposes)")
        
        # Handle UNKNOWN predictions
        if vlm_prediction == 'UNKNOWN':
            st.code(vlm_prediction, language=None)
            st.warning("⚠️ VLM returned 'UNKNOWN' - the model could not extract a clear answer from the chart")
            st.info("💡 **Tip:** Try selecting test_00702, test_00721, or test_01271 from the dropdown for examples with valid VLM predictions")
        else:
            st.code(vlm_prediction, language=None)
        
        st.caption("Model: **gemini-2.5-flash-image**")


def render_final_answer_with_source(vlm_run: Dict, text_run: Dict, escalated: bool, costs: Dict, latency: Dict):
    """Render final answer panel with clear source indication."""
    st.header("🎯 Final Answer")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if escalated:
            st.subheader("Final Answer (from VLM)")
            prediction = vlm_run.get('prediction', 'N/A')
            st.code(prediction, language=None)
            st.caption("✅ **Final answer source: VLM**")
            st.caption(f"Model: **gemini-2.5-flash-image**")
        else:
            st.subheader("Final Answer (from Text-Only)")
            prediction = text_run.get('prediction', 'N/A')
            st.code(prediction, language=None)
            st.caption("✅ **Final answer source: Text-only model**")
            st.caption("VLM not invoked (cost savings achieved)")
            st.caption(f"Model: **gpt-5-mini**")
    
    with col2:
        st.metric("Total Cost", f"${costs['total_cost']:.6f}")
        st.caption("per example")
        
        if costs['savings_pct'] > 0:
            st.metric("This Example's Savings", f"{costs['savings_pct']:.1f}%", 
                     delta=f"${costs['savings']:.6f}")
            st.caption("vs full VLM")
        
        st.metric("Total Latency", f"{latency['total_latency']}ms")
        st.caption("⚠️ estimated")


def update_sidebar_metrics():
    """Render improved sidebar metrics with clearer labels."""
    st.sidebar.header("📊 System Performance")
    
    st.sidebar.metric("Router Accuracy", "69.1%", help="Test set N=375")
    st.sidebar.metric("VLM Baseline", "67.7%", delta="+1.4pp advantage")
    st.sidebar.metric("VLM Invocation Rate", "73.1%", delta="-26.9pp reduction")
    st.sidebar.metric("Aggregate Test-Set Cost Savings", "9.5%", help="vs full VLM baseline")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ **Note:** Latency values are estimated from typical API response times.")
    st.sidebar.caption("🔒 **Mode:** Cache-only (no live API calls)")
