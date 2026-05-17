"""
TrialAgents SimulatorAgent (数字孪生生成)

新增智能体，负责基于终版报告自回归地模拟患者特异性轨迹

这是场景三（多样化任务完成）的执行组件之一
"""

from typing import Dict, List, Optional
import numpy as np
from .base_agent import Agent


class SimulatorAgent(Agent):
    """
    Simulator Agent / Digital Twin Generator
    
    基于终版报告自回归地生成患者特异性轨迹，实现数字孪生功能。
    
    应用场景：
    1. 治疗决策支持：模拟不同治疗方案的效果
    2. 风险预测：预测患者未来健康状况
    3. 虚拟试验：在真实试验前进行虚拟试验
    4. 个性化医疗：为特定患者定制治疗策略
    
    Attributes:
        prediction_horizon: 预测时间跨度（天）
        num_samples: 采样次数（用于不确定性估计）
        time_step: 时间步长（天）
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
        初始化SimulatorAgent
        
        Args:
            role: 角色名称
            llm: LLM模型名称
            prediction_horizon: 预测时间跨度（天）
            num_samples: 采样次数
            time_step: 时间步长（天）
        """
        super().__init__(role, llm)
        
        self.prediction_horizon = prediction_horizon
        self.num_samples = num_samples
        self.time_step = time_step
        
        # 状态特征定义
        self.state_features = [
            "vital_signs",      # 生命体征
            "lab_results",      # 实验室结果
            "symptoms",         # 症状
            "medications",      # 用药
            "risk_score",       # 风险评分
        ]
    
    def simulate(self, final_report: Dict, num_samples: int = None) -> Dict:
        """
        生成数字孪生轨迹

        基于终版报告，生成多条可能的患者发展轨迹

        Args:
            final_report: 来自Phase 2的终版报告
            num_samples: 采样次数（覆盖默认配置）

        Returns:
            Digital Twin字典，包含：
            - patient_id: 患者ID
            - base_profile: 基础档案
            - trajectories: 多条轨迹列表
            - aggregated_stats: 统计特征
        """
        num_samples = num_samples or self.num_samples

        # 生成多条轨迹
        trajectories = []
        for sample_id in range(num_samples):
            trajectory = self._generate_single_trajectory(final_report)
            trajectories.append(trajectory)

        # 整合多条轨迹，计算统计特征
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
        生成单条轨迹
        
        Args:
            final_report: 终版报告
            
        Returns:
            单条轨迹列表，每个元素是一个时间步的状态
        """
        trajectory = []
        current_state = self._initialize_state(final_report)
        
        for t in range(self.prediction_horizon):
            # 基于当前状态预测下一时刻
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
        基于终版报告初始化患者状态

        Args:
            final_report: 终版报告

        Returns:
            初始状态字典
        """
        # 提取基础档案
        base_profile = self._extract_base_profile(final_report)

        # 提取当前健康状况
        predicted_risk = base_profile.get('predicted_risk', 'unknown')
        confidence_score = base_profile.get('confidence_score', 0.5)
        diagnostic_rationale = base_profile.get('diagnostic_rationale', '')
        uncertainties = base_profile.get('uncertainties', [])

        # 提取风险因素（从诊断理由中推断）
        risk_factors = self._extract_risk_factors(diagnostic_rationale, uncertainties)

        # 提取症状和体征（如果有）
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
        从诊断理由和不确定性中提取风险因素

        Args:
            diagnostic_rationale: 诊断理由文本
            uncertainties: 不确定性列表

        Returns:
            风险因素列表
        """
        risk_factors = []

        # 常见风险因素关键词
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

        # 从不确定性中推断风险因素
        for uncertainty in uncertainties:
            if isinstance(uncertainty, str):
                for keyword in risk_keywords:
                    if keyword in uncertainty.lower() and keyword not in risk_factors:
                        risk_factors.append(keyword)

        return risk_factors
    
    def _predict_next_state(self, current_state: Dict, final_report: Dict, timestep: int) -> Dict:
        """
        预测下一时刻状态

        Args:
            current_state: 当前状态
            final_report: 终版报告（基础档案）
            timestep: 当前时间步

        Returns:
            下一时刻状态字典
        """
        # 构建预测prompt
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
            # 调用LLM
            response, prompt_tokens, completion_tokens = self.invoke(messages)
            next_state = self._parse_llm_response(response, current_state)
        except Exception as e:
            print(f"Warning: LLM prediction failed at timestep {timestep}: {e}")
            # 如果LLM调用失败，使用基于规则的简单预测
            next_state = self._rule_based_prediction(current_state, timestep)

        return next_state

    def _parse_llm_response(self, response: Dict, current_state: Dict) -> Dict:
        """
        解析LLM响应，生成下一状态

        Args:
            response: LLM响应字典
            current_state: 当前状态

        Returns:
            下一状态字典
        """
        predicted_vitals = response.get('predicted_vitals', current_state.get('vital_signs', {}))
        predicted_symptoms = response.get('predicted_symptoms', current_state.get('symptoms', []))
        predicted_labs = response.get('predicted_labs', current_state.get('lab_results', {}))
        stability = response.get('stability', 'stable')
        confidence = response.get('confidence', current_state.get('confidence', 0.5))

        # 继承不变的部分
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
        基于规则的简单预测（当LLM不可用时使用）

        Args:
            current_state: 当前状态
            timestep: 时间步

        Returns:
            下一状态字典
        """
        # 简单的基于时间的衰减/渐变模型
        confidence = max(0.1, current_state.get('confidence', 0.5) * 0.98)

        # 基于风险因素调整稳定性
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
        聚合多条轨迹，计算统计特征

        Args:
            trajectories: 多条轨迹列表

        Returns:
            聚合后的Digital Twin字典
        """
        if not trajectories:
            return {
                'mean_trajectory': [],
                'std_trajectory': [],
                'percentile_5': [],
                'percentile_95': [],
                'risk_points': [],
            }

        # 获取时间步数量
        num_timesteps = len(trajectories[0]) if trajectories else 0
        num_samples = len(trajectories)

        # 提取每条轨迹的置信度序列
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

        # 计算每个时间步的统计特征
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

            # 识别高风险点（标准差高或平均置信度低）
            if std_conf > 0.15 or mean_conf < 0.3:
                # 统计该时间点的稳定性分布
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

        # 计算风险趋势
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
        从终版报告中提取基础档案
        
        Args:
            final_report: 终版报告
            
        Returns:
            基础档案字典
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
        生成替代场景
        
        模拟不同治疗方案或条件下的患者发展
        
        Args:
            final_report: 终版报告
            scenarios: 场景描述列表，如["standard_care", "aggressive_treatment", "no_treatment"]
            
        Returns:
            各场景的模拟结果
        """
        results = {}
        for scenario in scenarios:
            # 修改基础条件，重新模拟
            modified_report = self._modify_report_for_scenario(final_report, scenario)
            trajectory = self._generate_single_trajectory(modified_report)
            results[scenario] = trajectory
        return results
    
    def _modify_report_for_scenario(self, final_report: Dict, scenario: str) -> Dict:
        """
        根据场景修改报告

        Args:
            final_report: 原始终版报告
            scenario: 场景名称

        Returns:
            修改后的报告
        """
        modified = final_report.copy()

        # 场景修改规则
        scenario_modifiers = {
            'standard_care': {
                'description': 'Standard treatment protocol',
                'risk_adjustment': 0.0,
            },
            'aggressive_treatment': {
                'description': 'More intensive treatment',
                'risk_adjustment': -0.1,  # 降低风险
                'confidence_adjustment': -0.05,  # 降低置信度（不确定性增加）
            },
            'no_treatment': {
                'description': 'No treatment / placebo',
                'risk_adjustment': 0.15,  # 增加风险
                'confidence_adjustment': 0.1,  # 增加置信度
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

            # 修改predicted_risk
            current_risk = base_profile.get('predicted_risk', 0.5)
            if isinstance(current_risk, (int, float)):
                new_risk = max(0.0, min(1.0, current_risk + modifier.get('risk_adjustment', 0.0)))
                modified['predicted_risk'] = new_risk

            # 修改confidence_score
            current_conf = base_profile.get('confidence_score', 0.5)
            if isinstance(current_conf, (int, float)):
                new_conf = max(0.0, min(1.0, current_conf + modifier.get('confidence_adjustment', 0.0)))
                modified['confidence_score'] = new_conf

            # 添加场景元数据
            modified['scenario'] = scenario
            modified['scenario_description'] = modifier['description']

        return modified
    
    def export_trajectory(self, digital_twin: Dict, format: str = "json") -> str:
        """
        导出轨迹到指定格式

        Args:
            digital_twin: Digital Twin字典
            format: 导出格式（json, csv, dict）

        Returns:
            导出数据（字符串或字典）
        """
        import json
        import csv
        from io import StringIO

        if format == "json":
            return json.dumps(digital_twin, indent=2, ensure_ascii=False)

        elif format == "csv":
            # 将轨迹转换为CSV格式
            trajectories = digital_twin.get('trajectories', [])
            if not trajectories:
                return ""

            # 获取所有字段名
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
            # 返回汇总信息
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
        调用LLM API

        Args:
            messages: 消息列表

        Returns:
            (响应字典, prompt_token数, completion_token数)
        """
        import time
        from .json_parser import extract_and_parse_json

        start = time.time()

        try:
            # 获取生成配置
            config = self.get_generation_config()

            if 'gemini' not in self.llm.lower():
                # 使用OpenAI兼容接口
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
                # 使用Gemini接口
                response = self.client.models.generate_content(
                    model=self.llm,
                    contents=[messages[0]['content'], messages[1]['content']],
                )
                content = response.text
                prompt_token = response.usage_metadata.prompt_token_count
                completion_token = response.usage_metadata.candidates_token_count

            # 解析JSON响应
            ans = extract_and_parse_json(content)

        except Exception as e:
            raise e

        end = time.time()
        print(f"SimulatorAgent (temp={config['temperature']:.2f}) time cost: {end - start:.2f}s.")

        return ans, prompt_token, completion_token
