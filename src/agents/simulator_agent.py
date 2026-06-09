"""
TrialAgents SimulatorAgent (Digital Twin Generator)

New agent responsible for auto-regressively simulating patient-specific
trajectories based on the final report.

This is one of the execution components for Scenario 3 (Diverse Task Completion).
"""

from typing import Dict, List, Optional
import numpy as np
from .base_agent import Agent


class SimulatorAgent(Agent):
    """
    Simulator Agent / Digital Twin Generator

    Auto-regressively generates patient-specific trajectories based on the
    final report, enabling digital twin functionality.

    Use cases:
    1. Treatment decision support: simulate outcomes of different treatments
    2. Risk prediction: predict future patient health
    3. Virtual trials: run virtual trials before real ones
    4. Personalized medicine: tailor treatment strategies per patient

    Attributes:
        prediction_horizon: Prediction horizon (days)
        num_samples: Number of samples (for uncertainty estimation)
        time_step: Time step (days)
    """

    def __init__(
        self,
        role: str = "simulator",
        llm: str = None,
        prediction_horizon: int = 30,
        num_samples: int = 10,
        time_step: int = 1,
    ) -> None:
        """
        Initialize SimulatorAgent.

        Args:
            role: Role name
            llm: LLM model name
            prediction_horizon: Prediction horizon (days)
            num_samples: Number of samples
            time_step: Time step (days)
        """
        super().__init__(role, llm)

        self.prediction_horizon = prediction_horizon
        self.num_samples = num_samples
        self.time_step = time_step

        self.state_features = [
            "vital_signs",
            "lab_results",
            "symptoms",
            "medications",
            "risk_score",
        ]

    def simulate(self, final_report: Dict, num_samples: int = None) -> Dict:
        """
        Generate digital twin trajectories.

        Generates multiple possible patient development trajectories
        based on the final report.

        Args:
            final_report: Final report from Phase 2
            num_samples: Number of samples (overrides default)

        Returns:
            Digital Twin dict containing:
            - patient_id: Patient ID
            - base_profile: Base profile
            - trajectories: List of trajectories
            - aggregated_stats: Statistical features
        """
        num_samples = num_samples or self.num_samples

        trajectories = []
        for sample_id in range(num_samples):
            trajectory = self._generate_single_trajectory(final_report)
            trajectories.append(trajectory)

        aggregated_stats = self._aggregate_trajectories(trajectories)
        base_profile = self._extract_base_profile(final_report)

        digital_twin = {
            'patient_id': final_report.get('patient_id', 'unknown'),
            'base_profile': base_profile,
            'trajectories': trajectories,
            'aggregated_stats': aggregated_stats,
        }

        return digital_twin

    def _generate_single_trajectory(self, final_report: Dict) -> List[Dict]:
        """
        Generate a single trajectory.

        Args:
            final_report: Final report

        Returns:
            List of per-timestep state dicts
        """
        trajectory = []
        current_state = self._initialize_state(final_report)

        for t in range(self.prediction_horizon):
            next_state = self._predict_next_state(
                current_state=current_state,
                final_report=final_report,
                timestep=t
            )
            trajectory.append({
                'timestep': t,
                'state': next_state
            })
            current_state = next_state

        return trajectory

    def _initialize_state(self, final_report: Dict) -> Dict:
        """
        Initialize patient state from the final report.

        Args:
            final_report: Final report

        Returns:
            Initial state dict
        """
        base_profile = self._extract_base_profile(final_report)

        predicted_risk = base_profile.get('predicted_risk', 'unknown')
        confidence_score = base_profile.get('confidence_score', 0.5)
        diagnostic_rationale = base_profile.get('diagnostic_rationale', '')
        uncertainties = base_profile.get('uncertainties', [])

        risk_factors = self._extract_risk_factors(diagnostic_rationale, uncertainties)

        symptoms = final_report.get('symptoms', [])
        vital_signs = final_report.get('vital_signs', {})
        lab_results = final_report.get('lab_results', {})

        return {
            'health_status': predicted_risk,
            'confidence': confidence_score,
            'risk_factors': risk_factors,
            'symptoms': symptoms,
            'vital_signs': vital_signs,
            'lab_results': lab_results,
            'diagnostic_rationale': diagnostic_rationale,
            'stability': 'unknown',
        }

    def _extract_risk_factors(self, diagnostic_rationale: str, uncertainties: List) -> List[str]:
        """
        Extract risk factors from diagnostic rationale and uncertainties.

        Args:
            diagnostic_rationale: Diagnostic rationale text
            uncertainties: List of uncertainties

        Returns:
            List of risk factors
        """
        risk_factors = []

        risk_keywords = [
            'high risk', 'low risk', 'severe', 'mild', 'moderate',
            'stable', 'unstable', 'critical', 'chronic', 'acute',
            'comorbidity', 'complication', 'organ dysfunction',
            'infection', 'bleeding', 'cardiac', 'renal', 'hepatic'
        ]

        if diagnostic_rationale:
            text_lower = diagnostic_rationale.lower()
            for keyword in risk_keywords:
                if keyword in text_lower:
                    if keyword not in risk_factors:
                        risk_factors.append(keyword)

        for uncertainty in uncertainties:
            if isinstance(uncertainty, str):
                for keyword in risk_keywords:
                    if keyword in uncertainty.lower() and keyword not in risk_factors:
                        risk_factors.append(keyword)

        return risk_factors

    def _predict_next_state(self, current_state: Dict, final_report: Dict, timestep: int) -> Dict:
        """
        Predict the next state.

        Args:
            current_state: Current state
            final_report: Final report (base profile)
            timestep: Current timestep

        Returns:
            Next state dict
        """
        base_profile = self._extract_base_profile(final_report)

        prompt = f"""
Based on the following patient information, predict the patient's state at timestep {timestep + 1} (day {timestep + 1}):

Patient Base Profile:
- Predicted Risk: {base_profile.get('predicted_risk', 'unknown')}
- Confidence Score: {base_profile.get('confidence_score', 'N/A')}
- Diagnostic Rationale: {base_profile.get('diagnostic_rationale', 'N/A')}
- Consensus Summary: {base_profile.get('consensus_summary', 'N/A')}

Current State (timestep {timestep}):
- Health Status: {current_state.get('health_status', 'unknown')}
- Confidence: {current_state.get('confidence', 0.0)}
- Risk Factors: {', '.join(current_state.get('risk_factors', []) or ['none identified'])}
- Symptoms: {', '.join(current_state.get('symptoms', []) or ['none'])}
- Vital Signs: {current_state.get('vital_signs', {})}
- Lab Results: {current_state.get('lab_results', {})}
- Stability: {current_state.get('stability', 'unknown')}

Predict the patient's state at the next timestep, including:
1. Changes in vital signs (if any)
2. New or resolved symptoms
3. Changes in lab results (if any)
4. Overall stability assessment (stable/improving/deteriorating)
5. Confidence level (0-1)

Respond in JSON format with fields: predicted_vitals, predicted_symptoms, predicted_labs, stability, confidence.
"""

        messages = [
            {"role": "system", "content": "You are a medical simulation expert. Predict patient trajectory changes based on the current state and medical knowledge."},
            {"role": "user", "content": prompt}
        ]

        try:
            response, prompt_tokens, completion_tokens = self.invoke(messages)
            next_state = self._parse_llm_response(response, current_state)
        except Exception as e:
            print(f"Warning: LLM prediction failed at timestep {timestep}: {e}")
            next_state = self._rule_based_prediction(current_state, timestep)

        return next_state

    def _parse_llm_response(self, response: Dict, current_state: Dict) -> Dict:
        """
        Parse LLM response to produce the next state.

        Args:
            response: LLM response dict
            current_state: Current state

        Returns:
            Next state dict
        """
        predicted_vitals = response.get('predicted_vitals', current_state.get('vital_signs', {}))
        predicted_symptoms = response.get('predicted_symptoms', current_state.get('symptoms', []))
        predicted_labs = response.get('predicted_labs', current_state.get('lab_results', {}))
        stability = response.get('stability', 'stable')
        confidence = response.get('confidence', current_state.get('confidence', 0.5))

        return {
            'health_status': current_state.get('health_status', 'unknown'),
            'confidence': confidence,
            'risk_factors': current_state.get('risk_factors', []),
            'symptoms': predicted_symptoms,
            'vital_signs': predicted_vitals,
            'lab_results': predicted_labs,
            'diagnostic_rationale': current_state.get('diagnostic_rationale', ''),
            'stability': stability,
        }

    def _rule_based_prediction(self, current_state: Dict, timestep: int) -> Dict:
        """
        Simple rule-based prediction (used when LLM is unavailable).

        Args:
            current_state: Current state
            timestep: Timestep

        Returns:
            Next state dict
        """
        confidence = max(0.1, current_state.get('confidence', 0.5) * 0.98)

        risk_factors = current_state.get('risk_factors', [])
        if any(rf in ['high risk', 'severe', 'critical', 'unstable'] for rf in risk_factors):
            stability = 'deteriorating' if timestep > 5 else 'unstable'
        else:
            stability = 'stable'

        return {
            'health_status': current_state.get('health_status', 'unknown'),
            'confidence': confidence,
            'risk_factors': risk_factors,
            'symptoms': current_state.get('symptoms', []),
            'vital_signs': current_state.get('vital_signs', {}),
            'lab_results': current_state.get('lab_results', {}),
            'diagnostic_rationale': current_state.get('diagnostic_rationale', ''),
            'stability': stability,
        }

    def _aggregate_trajectories(self, trajectories: List[List[Dict]]) -> Dict:
        """
        Aggregate multiple trajectories and compute statistics.

        Args:
            trajectories: List of trajectories

        Returns:
            Aggregated Digital Twin dict
        """
        if not trajectories:
            return {
                'mean_trajectory': [],
                'std_trajectory': [],
                'percentile_5': [],
                'percentile_95': [],
                'risk_points': [],
            }

        num_timesteps = len(trajectories[0]) if trajectories else 0
        num_samples = len(trajectories)

        confidence_sequences = []
        stability_sequences = []

        for trajectory in trajectories:
            confidences = []
            stabilities = []
            for state in trajectory:
                confidences.append(state.get('confidence', 0.5))
                stabilities.append(state.get('stability', 'unknown'))
            confidence_sequences.append(confidences)
            stability_sequences.append(stabilities)

        mean_confidences = []
        std_confidences = []
        percentile_5 = []
        percentile_95 = []
        risk_points = []

        for t in range(num_timesteps):
            t_confidences = [seq[t] for seq in confidence_sequences]
            mean_conf = np.mean(t_confidences)
            std_conf = np.std(t_confidences)

            mean_confidences.append(mean_conf)
            std_confidences.append(std_conf)
            percentile_5.append(np.percentile(t_confidences, 5))
            percentile_95.append(np.percentile(t_confidences, 95))

            if std_conf > 0.15 or mean_conf < 0.3:
                stability_dist = {}
                for seq in stability_sequences:
                    stab = seq[t]
                    stability_dist[stab] = stability_dist.get(stab, 0) + 1
                risk_points.append({
                    'timestep': t,
                    'risk_reason': 'high_uncertainty' if std_conf > 0.15 else 'low_confidence',
                    'mean_confidence': mean_conf,
                    'std_confidence': std_conf,
                    'stability_distribution': stability_dist,
                })

        risk_trend = 'increasing' if len(risk_points) > 2 and risk_points[-1]['mean_confidence'] < risk_points[0]['mean_confidence'] else 'stable'

        return {
            'mean_trajectory': mean_confidences,
            'std_trajectory': std_confidences,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'risk_points': risk_points,
            'risk_trend': risk_trend,
            'num_samples': num_samples,
            'num_timesteps': num_timesteps,
        }

    def _extract_base_profile(self, final_report: Dict) -> Dict:
        """
        Extract base profile from the final report.

        Args:
            final_report: Final report

        Returns:
            Base profile dict
        """
        return {
            'predicted_risk': final_report.get('predicted_risk'),
            'confidence_score': final_report.get('confidence_score'),
            'diagnostic_rationale': final_report.get('diagnostic_rationale'),
            'consensus_summary': final_report.get('consensus_summary'),
            'uncertainties': final_report.get('uncertainties', []),
        }

    def generate_alternative_scenarios(self, final_report: Dict, scenarios: List[str]) -> Dict:
        """
        Generate alternative scenarios.

        Simulate patient development under different treatments or conditions.

        Args:
            final_report: Final report
            scenarios: Scenario descriptions, e.g. ["standard_care", "aggressive_treatment", "no_treatment"]

        Returns:
            Simulation results per scenario
        """
        results = {}
        for scenario in scenarios:
            modified_report = self._modify_report_for_scenario(final_report, scenario)
            trajectory = self._generate_single_trajectory(modified_report)
            results[scenario] = trajectory
        return results

    def _modify_report_for_scenario(self, final_report: Dict, scenario: str) -> Dict:
        """
        Modify report based on scenario.

        Args:
            final_report: Original final report
            scenario: Scenario name

        Returns:
            Modified report
        """
        modified = final_report.copy()

        scenario_modifiers = {
            'standard_care': {
                'description': 'Standard treatment protocol',
                'risk_adjustment': 0.0,
            },
            'aggressive_treatment': {
                'description': 'More intensive treatment',
                'risk_adjustment': -0.1,
                'confidence_adjustment': -0.05,
            },
            'no_treatment': {
                'description': 'No treatment / placebo',
                'risk_adjustment': 0.15,
                'confidence_adjustment': 0.1,
            },
            'best_case': {
                'description': 'Optimal conditions',
                'risk_adjustment': -0.2,
                'confidence_adjustment': 0.1,
            },
            'worst_case': {
                'description': 'Adverse conditions',
                'risk_adjustment': 0.25,
                'confidence_adjustment': -0.1,
            },
        }

        if scenario in scenario_modifiers:
            modifier = scenario_modifiers[scenario]
            base_profile = self._extract_base_profile(final_report)

            current_risk = base_profile.get('predicted_risk', 0.5)
            if isinstance(current_risk, (int, float)):
                new_risk = max(0.0, min(1.0, current_risk + modifier.get('risk_adjustment', 0.0)))
                modified['predicted_risk'] = new_risk

            current_conf = base_profile.get('confidence_score', 0.5)
            if isinstance(current_conf, (int, float)):
                new_conf = max(0.0, min(1.0, current_conf + modifier.get('confidence_adjustment', 0.0)))
                modified['confidence_score'] = new_conf

            modified['scenario'] = scenario
            modified['scenario_description'] = modifier['description']

        return modified

    def export_trajectory(self, digital_twin: Dict, format: str = "json") -> str:
        """
        Export trajectory to a specified format.

        Args:
            digital_twin: Digital Twin dict
            format: Export format (json, csv, dict)

        Returns:
            Exported data (string or dict)
        """
        import json
        import csv
        from io import StringIO

        if format == "json":
            return json.dumps(digital_twin, indent=2, ensure_ascii=False)

        elif format == "csv":
            trajectories = digital_twin.get('trajectories', [])
            if not trajectories:
                return ""

            all_fields = set()
            for traj in trajectories:
                for state in traj:
                    all_fields.update(state.keys())
            fieldnames = sorted(all_fields)

            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for traj_idx, trajectory in enumerate(trajectories):
                for state in trajectory:
                    row = {'trajectory_id': traj_idx, **state}
                    writer.writerow(row)

            return output.getvalue()

        elif format == "summary":
            summary = {
                'patient_id': digital_twin.get('patient_id', 'unknown'),
                'num_trajectories': len(digital_twin.get('trajectories', [])),
                'prediction_horizon': len(digital_twin.get('trajectories', [[]])[0]),
                'aggregated_stats': digital_twin.get('aggregated_stats', {}),
                'base_profile': digital_twin.get('base_profile', {}),
            }
            return json.dumps(summary, indent=2, ensure_ascii=False)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'csv', or 'summary'.")

    def invoke(self, messages: List[Dict[str, str]]) -> tuple:
        """
        Call LLM API.

        Args:
            messages: Message list

        Returns:
            (response dict, prompt_tokens, completion_tokens)
        """
        import time
        from .json_parser import extract_and_parse_json

        start = time.time()

        try:
            config = self.get_generation_config()

            if 'gemini' not in self.llm.lower():
                response = self.client.chat.completions.create(
                    model=self.llm,
                    messages=messages,
                    stream=False,
                    temperature=config["temperature"],
                    top_p=config["top_p"],
                    frequency_penalty=config["frequency_penalty"],
                    presence_penalty=config["presence_penalty"],
                    max_tokens=config["max_tokens"],
                    seed=config["seed"]
                )
                content = response.choices[0].message.content
                prompt_token = response.usage.prompt_tokens
                completion_token = response.usage.completion_tokens
            else:
                response = self.client.models.generate_content(
                    model=self.llm,
                    contents=[messages[0]['content'], messages[1]['content']],
                )
                content = response.text
                prompt_token = response.usage_metadata.prompt_token_count
                completion_token = response.usage_metadata.candidates_token_count

            ans = extract_and_parse_json(content)

        except Exception as e:
            raise e

        end = time.time()
        print(f"SimulatorAgent (temp={config['temperature']:.2f}) time cost: {end - start:.2f}s.")

        return ans, prompt_token, completion_token
