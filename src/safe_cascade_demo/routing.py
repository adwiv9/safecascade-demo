"""
Routing Logic for SAFE-Cascade Demo

Handles threshold-based routing decisions.
"""

from typing import Dict, Tuple


class RouterDecisionEngine:
    """Make routing decisions based on threshold."""
    
    # Cost constants (per 1k examples)
    OCR_COST = 0.0012 * 1000  # $1.20 per 1k
    TEXT_COST = 0.03          # $0.03 per 1k
    VLM_COST = 0.80           # $0.80 per 1k
    
    # Latency estimates (ms)
    OCR_LATENCY = 1000
    TEXT_LATENCY = 2050
    VLM_LATENCY = 5500
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def make_decision(self, probability: float) -> Tuple[bool, str, str]:
        """
        Make routing decision.
        
        Returns:
            (escalate, decision_label, explanation)
        """
        escalate = probability >= self.threshold
        
        if escalate:
            decision_label = "ESCALATE_VLM"
            explanation = self._get_escalation_explanation(probability)
        else:
            decision_label = "STOP_TEXT"
            explanation = self._get_stop_explanation(probability)
        
        return escalate, decision_label, explanation
    
    def _get_escalation_explanation(self, prob: float) -> str:
        """Generate explanation for escalation."""
        confidence = int(prob * 100)
        
        if prob >= 0.7:
            return f"🔴 High confidence ({confidence}%) that visual reasoning is needed. Text-only answer likely unreliable."
        elif prob >= 0.5:
            return f"🟡 Moderate confidence ({confidence}%) that chart analysis required. Escalating to VLM for accuracy."
        else:
            return f"🟢 Borderline case ({confidence}%). Escalating to VLM to be safe (threshold={self.threshold})."
    
    def _get_stop_explanation(self, prob: float) -> str:
        """Generate explanation for stopping."""
        confidence = int((1 - prob) * 100)
        
        if prob < 0.2:
            return f"✅ High confidence ({confidence}%) that text-only answer is reliable. No visual reasoning needed."
        elif prob < 0.3:
            return f"✅ Good confidence ({confidence}%) in text answer. Stopping to save cost."
        else:
            return f"⚠️ Moderate confidence ({confidence}%). Stopping at threshold={self.threshold}, but close to escalation."
    
    def calculate_cost(self, escalate: bool) -> Dict[str, float]:
        """Calculate costs."""
        base_cost = self.OCR_COST + self.TEXT_COST
        
        if escalate:
            vlm_cost = self.VLM_COST
            total_cost = base_cost + vlm_cost
        else:
            vlm_cost = 0.0
            total_cost = base_cost
        
        # Full VLM baseline
        full_vlm_cost = self.OCR_COST + self.TEXT_COST + self.VLM_COST
        savings = full_vlm_cost - total_cost
        savings_pct = (savings / full_vlm_cost) * 100
        
        return {
            'ocr_cost': self.OCR_COST / 1000,  # Per example
            'text_cost': self.TEXT_COST / 1000,
            'vlm_cost': vlm_cost / 1000 if escalate else 0.0,
            'total_cost': total_cost / 1000,
            'full_vlm_cost': full_vlm_cost / 1000,
            'savings': savings / 1000,
            'savings_pct': savings_pct
        }
    
    def calculate_latency(self, escalate: bool) -> Dict[str, float]:
        """Calculate estimated latency."""
        base_latency = self.OCR_LATENCY + self.TEXT_LATENCY
        
        if escalate:
            total_latency = base_latency + self.VLM_LATENCY
        else:
            total_latency = base_latency
        
        full_vlm_latency = self.OCR_LATENCY + self.TEXT_LATENCY + self.VLM_LATENCY
        speedup = full_vlm_latency / total_latency if total_latency > 0 else 1.0
        
        return {
            'ocr_latency': self.OCR_LATENCY,
            'text_latency': self.TEXT_LATENCY,
            'vlm_latency': self.VLM_LATENCY if escalate else 0,
            'total_latency': total_latency,
            'full_vlm_latency': full_vlm_latency,
            'speedup': speedup
        }
    
    def get_threshold_metrics(self, threshold: float) -> Dict[str, float]:
        """Get metrics for a specific threshold from sweep data."""
        # Hardcoded from table_threshold_sweep.csv
        threshold_data = {
            0.3: {'accuracy': 69.07, 'vlm_rate': 73.07, 'missed_esc': 5.33},
            0.4: {'accuracy': 64.27, 'vlm_rate': 57.07, 'missed_esc': 10.67},
            0.5: {'accuracy': 62.13, 'vlm_rate': 43.20, 'missed_esc': 14.93},
            0.6: {'accuracy': 54.67, 'vlm_rate': 23.20, 'missed_esc': 22.93},
            0.7: {'accuracy': 52.00, 'vlm_rate': 10.40, 'missed_esc': 27.20}
        }
        
        # Find closest threshold
        closest = min(threshold_data.keys(), key=lambda x: abs(x - threshold))
        return threshold_data[closest]
