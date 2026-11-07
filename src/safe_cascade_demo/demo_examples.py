"""
Demo Examples for SAFE-Cascade

Curated examples for demonstration purposes.
"""

from typing import Dict, List


class DemoExamples:
    """Manage demo examples."""
    
    @staticmethod
    def get_curated_examples() -> Dict[str, List[str]]:
        """
        Get curated example IDs for different scenarios.
        
        These are actual examples from the test set with verified probabilities.
        15 total examples (5 per category) for diverse demonstration.
        """
        return {
            'text_solvable': [
                # Low probability (< 0.15), text-only works
                'test_00121',  # prob=0.010 - Extremely low, trivial question
                'test_00036',  # prob=0.020 - Very low, text sufficient
                'test_00023',  # prob=0.030 - Very low, simple lookup
                'test_00319',  # prob=0.030 - Very low, direct answer
                'test_00029',  # prob=0.040 - Low, no visual needed
            ],
            'vlm_needed': [
                # High probability (> 0.75), VLM required
                'test_02345',  # prob=0.970 - Extremely high, complex visual
                'test_01446',  # prob=0.930 - Very high, chart analysis critical
                'test_01772',  # prob=0.920 - Very high, visual reasoning needed
                'test_00776',  # prob=0.900 - Very high, trend identification
                'test_02278',  # prob=0.890 - High, spatial reasoning required
            ],
            'borderline': [
                # Medium probability (0.25-0.40), threshold-sensitive
                'test_00702',  # prob=0.250 - Borderline, threshold-sensitive (VLM: 18.2%)
                'test_00721',  # prob=0.260 - Borderline, close call (VLM: 25)
                'test_01271',  # prob=0.260 - Borderline, marginal case (VLM: Jimmy Fallon)
                'test_01077',  # prob=0.250 - Borderline, just below threshold (VLM: UNKNOWN)
                'test_01941',  # prob=0.250 - Borderline, could go either way (VLM: UNKNOWN)
            ]
        }
    
    @staticmethod
    def get_example_description(category: str) -> str:
        """Get description for example category."""
        descriptions = {
            'text_solvable': """
            **Text-Solvable Examples**
            
            These questions can be answered using OCR text alone:
            - Simple numeric lookups
            - Direct value extraction
            - Text-based comparisons
            
            **Expected Behavior:**
            - Low escalation probability (< 0.3)
            - Router stops at text-only
            - Cost savings: ~40% vs full VLM
            """,
            'vlm_needed': """
            **VLM-Needed Examples**
            
            These questions require visual chart analysis:
            - Trend identification
            - Visual comparisons
            - Chart structure understanding
            
            **Expected Behavior:**
            - High escalation probability (> 0.6)
            - Router escalates to VLM
            - Accuracy improvement over text-only
            """,
            'borderline': """
            **Borderline Examples**
            
            These questions are threshold-sensitive:
            - Moderate visual complexity
            - Partial information in OCR
            - Ambiguous question type
            
            **Expected Behavior:**
            - Medium probability (0.25-0.35)
            - Decision changes with threshold
            - Good for demonstrating tradeoff
            """
        }
        return descriptions.get(category, "")
    
    @staticmethod
    def get_presenter_notes(category: str) -> Dict[str, str]:
        """Get presenter notes for each category."""
        notes = {
            'text_solvable': {
                'intro': "Let's start with a simple question that doesn't need visual reasoning.",
                'highlight': "Notice the low probability - the router is confident text-only will work.",
                'conclusion': "By stopping here, we save ~40% cost with no accuracy loss."
            },
            'vlm_needed': {
                'intro': "Now let's look at a question that requires chart analysis.",
                'highlight': "See the high probability - the router knows visual reasoning is needed.",
                'conclusion': "Escalating to VLM gives us the correct answer that text-only missed."
            },
            'borderline': {
                'intro': "This is an interesting borderline case.",
                'highlight': "Watch what happens when we adjust the threshold...",
                'conclusion': "This shows the accuracy-cost tradeoff we can control."
            }
        }
        return notes.get(category, {})
