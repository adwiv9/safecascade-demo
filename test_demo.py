#!/usr/bin/env python3
"""
Smoke Test for SAFE-Cascade Demo

Validates that all components work correctly before demo.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules import successfully."""
    print("="*80)
    print("TEST 1: Module Imports")
    print("="*80)
    
    try:
        from safe_cascade_demo.data_loader import DemoDataLoader
        print("✅ data_loader imported")
        
        from safe_cascade_demo.routing import RouterDecisionEngine
        print("✅ routing imported")
        
        from safe_cascade_demo.ui_components import (
            render_input_panel,
            render_ocr_panel,
            render_text_answer_panel,
            render_router_panel,
            render_threshold_slider,
            render_vlm_answer_panel,
            render_operating_frontier,
            render_demo_script_panel,
            render_key_metrics_sidebar
        )
        print("✅ ui_components imported")
        
        from safe_cascade_demo.demo_examples import DemoExamples
        print("✅ demo_examples imported")
        
        print("\n✅ All imports successful\n")
        return True
    except Exception as e:
        print(f"\n❌ Import failed: {e}\n")
        return False


def test_data_loading():
    """Test data loading."""
    print("="*80)
    print("TEST 2: Data Loading")
    print("="*80)
    
    try:
        from safe_cascade_demo.data_loader import DemoDataLoader
        
        loader = DemoDataLoader()
        
        # Load test IDs
        test_ids = loader.load_test_ids()
        print(f"✅ Loaded {len(test_ids)} test examples")
        assert len(test_ids) > 0, "No test examples found"
        
        # Load dataset
        dataset, dataset_lookup = loader.load_dataset()
        print(f"✅ Loaded dataset: {len(dataset)} examples")
        
        # Load text runs
        text_runs, text_lookup = loader.load_text_runs()
        print(f"✅ Loaded text runs: {len(text_runs)} predictions")
        
        # Load VLM runs
        vlm_runs, vlm_lookup = loader.load_vlm_runs()
        print(f"✅ Loaded VLM runs: {len(vlm_runs)} predictions")
        
        # Load router predictions
        router_preds, router_lookup = loader.load_router_predictions()
        print(f"✅ Loaded router predictions: {len(router_preds)} predictions")
        
        # Load threshold sweep
        threshold_sweep = loader.load_threshold_sweep()
        print(f"✅ Loaded threshold sweep: {len(threshold_sweep)} thresholds")
        
        # Load main results
        main_results = loader.load_main_results()
        print(f"✅ Loaded main results: {len(main_results)} methods")
        
        print("\n✅ All data loaded successfully\n")
        return True, loader, test_ids
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_example_loading(loader, test_ids):
    """Test loading individual examples."""
    print("="*80)
    print("TEST 3: Example Loading")
    print("="*80)
    
    try:
        # Load first test example
        example_id = test_ids[0]
        example = loader.get_example(example_id)
        
        assert example is not None, f"Failed to load example {example_id}"
        print(f"✅ Loaded example: {example_id}")
        
        # Check required fields
        assert 'query' in example or 'question' in example, "No question field"
        print(f"✅ Example has question")
        
        assert 'ocr_text' in example, "No OCR text"
        print(f"✅ Example has OCR text")
        
        assert 'text_run' in example, "No text run"
        print(f"✅ Example has text prediction")
        
        assert 'vlm_run' in example, "No VLM run"
        print(f"✅ Example has VLM prediction")
        
        assert 'router_pred' in example, "No router prediction"
        print(f"✅ Example has router prediction")
        
        print("\n✅ Example structure validated\n")
        return True, example
    except Exception as e:
        print(f"\n❌ Example loading failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False, None


def test_routing(example):
    """Test routing logic."""
    print("="*80)
    print("TEST 4: Routing Logic")
    print("="*80)
    
    try:
        from safe_cascade_demo.routing import RouterDecisionEngine
        
        router_pred = example.get('router_pred', {})
        probability = router_pred.get('probability', 0.5)
        
        print(f"Example probability: {probability:.3f}")
        
        # Test at threshold 0.3
        router = RouterDecisionEngine(threshold=0.3)
        escalate1, label1, explanation1 = router.make_decision(probability)
        print(f"✅ Decision at θ=0.3: {label1}")
        
        # Test at threshold 0.6
        router2 = RouterDecisionEngine(threshold=0.6)
        escalate2, label2, explanation2 = router2.make_decision(probability)
        print(f"✅ Decision at θ=0.6: {label2}")
        
        # Verify threshold sensitivity
        if 0.3 < probability < 0.6:
            assert escalate1 != escalate2, "Threshold should change decision for borderline cases"
            print(f"✅ Threshold sensitivity verified (decision changed)")
        else:
            print(f"✅ Threshold sensitivity not applicable (probability outside range)")
        
        # Test cost calculation
        costs = router.calculate_cost(escalate1)
        assert 'total_cost' in costs, "Missing total_cost"
        assert 'savings_pct' in costs, "Missing savings_pct"
        print(f"✅ Cost calculation: ${costs['total_cost']:.6f} per example")
        
        # Test latency calculation
        latency = router.calculate_latency(escalate1)
        assert 'total_latency' in latency, "Missing total_latency"
        print(f"✅ Latency calculation: {latency['total_latency']}ms")
        
        print("\n✅ Routing logic validated\n")
        return True
    except Exception as e:
        print(f"\n❌ Routing test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_demo_examples():
    """Test demo examples."""
    print("="*80)
    print("TEST 5: Demo Examples")
    print("="*80)
    
    try:
        from safe_cascade_demo.demo_examples import DemoExamples
        
        curated = DemoExamples.get_curated_examples()
        
        assert 'text_solvable' in curated, "Missing text_solvable category"
        assert 'vlm_needed' in curated, "Missing vlm_needed category"
        assert 'borderline' in curated, "Missing borderline category"
        
        print(f"✅ Text-solvable examples: {len(curated['text_solvable'])}")
        print(f"✅ VLM-needed examples: {len(curated['vlm_needed'])}")
        print(f"✅ Borderline examples: {len(curated['borderline'])}")
        
        # Test descriptions
        desc = DemoExamples.get_example_description('text_solvable')
        assert len(desc) > 0, "Empty description"
        print(f"✅ Category descriptions available")
        
        # Test presenter notes
        notes = DemoExamples.get_presenter_notes('text_solvable')
        assert 'intro' in notes, "Missing intro note"
        print(f"✅ Presenter notes available")
        
        print("\n✅ Demo examples validated\n")
        return True
    except Exception as e:
        print(f"\n❌ Demo examples test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_no_live_api_calls():
    """Verify no live API calls in default mode."""
    print("="*80)
    print("TEST 6: No Live API Calls")
    print("="*80)
    
    try:
        # Check app configuration
        import app_safe_cascade_demo
        
        live_api = getattr(app_safe_cascade_demo, 'LIVE_API_ENABLED', None)
        
        if live_api is False:
            print("✅ LIVE_API_ENABLED = False (correct)")
        elif live_api is None:
            print("⚠️  LIVE_API_ENABLED not found (assuming False)")
        else:
            print(f"❌ LIVE_API_ENABLED = {live_api} (should be False for demo)")
            return False
        
        print("\n✅ No live API calls will occur\n")
        return True
    except Exception as e:
        print(f"\n❌ API check failed: {e}\n")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SAFE-CASCADE DEMO SMOKE TEST")
    print("="*80 + "\n")
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: Data Loading
    success, loader, test_ids = test_data_loading()
    results.append(("Data Loading", success))
    
    if not success:
        print("\n❌ Cannot continue without data. Please check cached artifacts.\n")
        return False
    
    # Test 3: Example Loading
    success, example = test_example_loading(loader, test_ids)
    results.append(("Example Loading", success))
    
    if not success:
        print("\n❌ Cannot continue without example data.\n")
        return False
    
    # Test 4: Routing
    results.append(("Routing Logic", test_routing(example)))
    
    # Test 5: Demo Examples
    results.append(("Demo Examples", test_demo_examples()))
    
    # Test 6: No Live API
    results.append(("No Live API", test_no_live_api_calls()))
    
    # Summary
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED - DEMO IS READY!")
    else:
        print("❌ SOME TESTS FAILED - PLEASE FIX BEFORE DEMO")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
