"""
Data Loader for SAFE-Cascade Demo

Loads cached experiment artifacts for demo UI.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st


class DemoDataLoader:
    """Load cached experiment data for demo."""
    
    def __init__(self, experiment_dir: str = "safe_cascade/data/experiments/chartqa_full"):
        self.exp_dir = Path(experiment_dir)
        self.assets_dir = Path("safe_cascade/data/paper_assets_v3")
        
        # Cache loaded data
        self._dataset = None
        self._text_runs = None
        self._vlm_runs = None
        self._router_predictions = None
        self._test_ids = None
        self._threshold_sweep = None
        self._main_results = None
        
    @st.cache_data
    def load_dataset(_self):
        """Load main dataset with OCR."""
        if _self._dataset is None:
            dataset_path = _self.exp_dir / "chartqa_full_with_ocr.jsonl"
            _self._dataset = _self._load_jsonl(dataset_path)
            _self._dataset_lookup = {ex['id']: ex for ex in _self._dataset}
        return _self._dataset, _self._dataset_lookup
    
    @st.cache_data
    def load_text_runs(_self):
        """Load text model predictions."""
        if _self._text_runs is None:
            text_path = _self.exp_dir / "runs" / "text_gpt-5-mini.jsonl"
            _self._text_runs = _self._load_jsonl(text_path)
            _self._text_lookup = {r['id']: r for r in _self._text_runs}
        return _self._text_runs, _self._text_lookup
    
    @st.cache_data
    def load_vlm_runs(_self):
        """Load VLM predictions."""
        if _self._vlm_runs is None:
            vlm_path = _self.exp_dir / "runs" / "vlm_gemini-2.5-flash-image.jsonl"
            _self._vlm_runs = _self._load_jsonl(vlm_path)
            _self._vlm_lookup = {r['id']: r for r in _self._vlm_runs}
        return _self._vlm_runs, _self._vlm_lookup
    
    @st.cache_data
    def load_router_predictions(_self):
        """Load router predictions."""
        if _self._router_predictions is None:
            router_path = _self.exp_dir / "router" / "test_predictions.jsonl"
            _self._router_predictions = _self._load_jsonl(router_path)
            _self._router_lookup = {r['id']: r for r in _self._router_predictions}
        return _self._router_predictions, _self._router_lookup
    
    @st.cache_data
    def load_test_ids(_self):
        """Load test set IDs."""
        if _self._test_ids is None:
            test_ids_path = _self.exp_dir / "router" / "test_ids.csv"
            _self._test_ids = pd.read_csv(test_ids_path)['id'].tolist()
        return _self._test_ids
    
    @st.cache_data
    def load_threshold_sweep(_self):
        """Load threshold sweep table."""
        if _self._threshold_sweep is None:
            sweep_path = _self.assets_dir / "tables" / "table_threshold_sweep.csv"
            _self._threshold_sweep = pd.read_csv(sweep_path)
        return _self._threshold_sweep
    
    @st.cache_data
    def load_main_results(_self):
        """Load main results table."""
        if _self._main_results is None:
            results_path = _self.assets_dir / "tables" / "table_main_results.csv"
            _self._main_results = pd.read_csv(results_path)
        return _self._main_results
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file."""
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def get_example(self, example_id: str) -> Optional[Dict]:
        """Get complete example data."""
        dataset, dataset_lookup = self.load_dataset()
        text_runs, text_lookup = self.load_text_runs()
        vlm_runs, vlm_lookup = self.load_vlm_runs()
        router_preds, router_lookup = self.load_router_predictions()
        
        if example_id not in dataset_lookup:
            return None
        
        example = dataset_lookup[example_id].copy()
        example['text_run'] = text_lookup.get(example_id, {})
        example['vlm_run'] = vlm_lookup.get(example_id, {})
        example['router_pred'] = router_lookup.get(example_id, {})
        
        return example
    
    def get_test_examples(self) -> List[str]:
        """Get all test example IDs."""
        return self.load_test_ids()
    
    def find_interesting_examples(self) -> Dict[str, List[str]]:
        """Find examples for demo script."""
        test_ids = self.load_test_ids()
        dataset, dataset_lookup = self.load_dataset()
        text_runs, text_lookup = self.load_text_runs()
        vlm_runs, vlm_lookup = self.load_vlm_runs()
        router_preds, router_lookup = self.load_router_predictions()
        
        text_solvable = []  # Low prob, text correct
        vlm_needed = []     # High prob, text wrong, vlm correct
        borderline = []     # Mid prob (0.25-0.35)
        
        for ex_id in test_ids[:100]:  # Check first 100 for speed
            if ex_id not in router_lookup:
                continue
            
            router = router_lookup[ex_id]
            prob = router.get('probability', 0)
            
            # Get correctness
            ex = dataset_lookup[ex_id]
            gold = ex.get('answers', [ex.get('answer', '')])
            if not isinstance(gold, list):
                gold = [gold]
            
            text_pred = text_lookup.get(ex_id, {}).get('prediction', '')
            vlm_pred = vlm_lookup.get(ex_id, {}).get('prediction', '')
            
            text_correct = self._check_correct(text_pred, gold)
            vlm_correct = self._check_correct(vlm_pred, gold)
            
            # Categorize
            if prob < 0.25 and text_correct:
                text_solvable.append(ex_id)
            elif prob > 0.6 and not text_correct and vlm_correct:
                vlm_needed.append(ex_id)
            elif 0.25 <= prob <= 0.35:
                borderline.append(ex_id)
        
        return {
            'text_solvable': text_solvable[:5],
            'vlm_needed': vlm_needed[:5],
            'borderline': borderline[:5]
        }
    
    def _check_correct(self, pred: str, gold: List[str]) -> bool:
        """Simple correctness check."""
        pred_norm = str(pred).strip().lower()
        for ans in gold:
            ans_norm = str(ans).strip().lower()
            if pred_norm == ans_norm or pred_norm in ans_norm or ans_norm in pred_norm:
                return True
        return False
    
    def get_image_path(self, example_id: str) -> Optional[Path]:
        """Get path to chart image."""
        dataset, dataset_lookup = self.load_dataset()
        if example_id not in dataset_lookup:
            return None
        
        ex = dataset_lookup[example_id]
        
        # Try image_path field first (from data)
        image_path = ex.get('image_path', '')
        if image_path:
            # Try relative to different base directories
            possible_paths = [
                Path(image_path),  # Try as-is first
                Path("safe_cascade/data") / image_path,
                Path("data") / image_path,
            ]
            for path in possible_paths:
                if path.exists():
                    return path
        
        # Fallback to image field
        image_filename = ex.get('image', '')
        if image_filename:
            possible_paths = [
                Path("safe_cascade/data/chartqa/images") / image_filename,
                Path("data/chartqa/images") / image_filename,
                Path("chartqa/images") / image_filename,
            ]
            for path in possible_paths:
                if path.exists():
                    return path
        
        return None
