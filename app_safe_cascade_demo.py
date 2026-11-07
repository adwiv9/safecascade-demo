"""
SAFE-Cascade Demo UI

Interactive demonstration of cost-adaptive chart question answering.

Run with: streamlit run app_safe_cascade_demo.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from safe_cascade_demo.data_loader import DemoDataLoader
from safe_cascade_demo.routing import RouterDecisionEngine
from safe_cascade_demo.ui_components import (
    render_input_panel,
    render_ocr_panel,
    render_text_answer_panel,
    render_router_panel,
    render_threshold_slider,
    render_vlm_answer_panel,
    render_operating_frontier
)
from safe_cascade_demo.ui_components_new import (
    render_routing_signals_panel,
    render_vlm_baseline_answer,
    render_final_answer_with_source,
    update_sidebar_metrics
)
from safe_cascade_demo.demo_examples import DemoExamples


# Page config
st.set_page_config(
    page_title="SAFE-Cascade Demo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
LIVE_API_ENABLED = False  # Set to True to enable live API calls


def initialize_session_state():
    """Initialize session state variables."""
    if 'current_example_id' not in st.session_state:
        st.session_state.current_example_id = None
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = None
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.3
    if 'show_gold_answer' not in st.session_state:
        st.session_state.show_gold_answer = False
    if 'current_category' not in st.session_state:
        st.session_state.threshold = 0.3


def load_data():
    """Load all required data."""
    try:
        loader = DemoDataLoader()
        
        # Load all datasets
        loader.load_dataset()
        loader.load_text_runs()
        loader.load_vlm_runs()
        loader.load_router_predictions()
        loader.load_test_ids()
        threshold_sweep = loader.load_threshold_sweep()
        
        return loader, threshold_sweep
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.info("Please ensure all cached artifacts are available in safe_cascade/data/")
        st.stop()


def main():
    """Main application."""
    initialize_session_state()
    
    # Title
    st.title("🎯 SAFE-Cascade: Cost-Adaptive Chart QA")
    st.caption("Interactive Demo for CIKM 2026")
    
    # Load data
    with st.spinner("Loading cached experiment data..."):
        loader, threshold_sweep = load_data()
    
    # Sidebar
    update_sidebar_metrics()
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎮 Controls")
    
    # Gold answer toggle
    st.sidebar.checkbox(
        "Show gold answer (evaluation only)",
        key='show_gold_answer',
        value=False,
        help="Show reference answer for evaluation purposes only"
    )
    
    # Example selector
    test_ids = loader.get_test_examples()
    
    # Demo mode selector
    demo_examples = DemoExamples.get_curated_examples()
    
    example_source = st.sidebar.radio(
        "Example Source",
        ["Curated Demo", "Browse Test Set"],
        help="Choose between curated examples or browse all test examples"
    )
    
    if example_source == "Curated Demo":
        category = st.sidebar.selectbox(
            "Demo Category",
            ["text_solvable", "vlm_needed", "borderline"],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        st.session_state.current_category = category
        
        available_examples = demo_examples[category]
        selected_id = st.sidebar.selectbox(
            "Select Example",
            available_examples,
            index=0
        )
        
        # Show category description
        with st.sidebar.expander("ℹ️ About This Category"):
            st.markdown(DemoExamples.get_example_description(category))
    else:
        selected_id = st.sidebar.selectbox(
            "Select Example ID",
            test_ids[:50],  # Limit to first 50 for performance
            index=0
        )
    
    st.session_state.current_example_id = selected_id
    
    # Load example
    example = loader.get_example(selected_id)
    
    if example is None:
        st.error(f"❌ Example {selected_id} not found")
        st.stop()
    
    # Get image path
    image_path = loader.get_image_path(selected_id)
    
    # Main content
    st.markdown("---")
    
    # 1. Input Panel (with gold answer toggle)
    render_input_panel(example, image_path, show_gold=st.session_state.show_gold_answer)
    
    st.markdown("---")
    
    # 2. OCR Panel (expanded for curated demos, with text prediction hint)
    text_run = example.get('text_run', {})
    text_pred = text_run.get('prediction', '')
    is_curated = example_source == "Curated Demo"
    render_ocr_panel(example, text_prediction=text_pred, expanded=is_curated)
    
    st.markdown("---")
    
    # 3. Text Answer Panel
    router_pred = example.get('router_pred', {})
    
    # Get threshold from slider
    st.markdown("---")
    st.header("⚙️ Routing Configuration")
    threshold = render_threshold_slider()
    st.session_state.threshold = threshold
    
    # Create router engine
    router = RouterDecisionEngine(threshold=threshold)
    
    # Calculate costs and latency (text-only first)
    escalate_decision, decision_label, explanation = router.make_decision(
        router_pred.get('probability', 0.0)
    )
    costs = router.calculate_cost(escalate_decision)
    latency = router.calculate_latency(escalate_decision)
    
    render_text_answer_panel(text_run, costs, latency)
    
    st.markdown("---")
    
    # 4. Router Decision Panel
    render_router_panel(router_pred, threshold, decision_label, explanation)
    
    st.markdown("---")
    
    # 4b. Routing Signals Panel
    render_routing_signals_panel(example, router_pred, text_run, threshold, decision_label)
    
    st.markdown("---")
    
    # 5. Final Answer Panel (with clear source indication)
    vlm_run = example.get('vlm_run', {})
    render_final_answer_with_source(vlm_run, text_run, escalate_decision, costs, latency)
    
    st.markdown("---")
    
    # 5b. VLM Baseline Answer (collapsible)
    render_vlm_baseline_answer(vlm_run, escalate_decision)
    
    st.markdown("---")
    
    # 6. Operating Frontier Panel
    render_operating_frontier(threshold_sweep, threshold)
    
    # Footer
    st.markdown("---")
    st.caption("🔒 **Cache-Only Mode:** Using pre-computed results. No live API calls.")
    st.caption("📊 **Data:** ChartQA-2500 Test Set (N=375)")
    st.caption("🤖 **Models:** gpt-5-mini (text), gemini-2.5-flash-image (VLM), RandomForest (router)")
    
    # Debug info (collapsible)
    with st.expander("🔧 Debug Info"):
        st.json({
            'example_id': selected_id,
            'probability': router_pred.get('probability', 0.0),
            'threshold': threshold,
            'decision': decision_label,
            'escalated': escalate_decision,
            'text_prediction': text_run.get('prediction', 'N/A'),
            'vlm_prediction': vlm_run.get('prediction', 'N/A') if escalate_decision else 'Not invoked',
            'gold_answer': example.get('answers', example.get('answer', 'N/A')),
            'total_cost': f"${costs['total_cost']:.6f}",
            'savings_vs_vlm': f"{costs['savings_pct']:.1f}%"
        })


if __name__ == "__main__":
    main()
