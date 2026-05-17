from liquid import Template

# =============================================================================
# Phase 1: Clinical Pharmacovigilance Specialist Analysis
# =============================================================================

clinician_review_system = """You are an experienced clinician with extensive medical knowledge. I will provide you with multivariate time-series clinical trial data for a patient, which is a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits, represented as a triplet <event, value, time> or a quadruplet <event, value, start_time, end_time> for each variable. The mortality risk refers to the likelihood or probability of a patient dying within a specific timeframe. I will also retrieve relevant medical knowledge based on the AI model's analysis results and provide it to you. Please summarize the patient's condition based on multivariate time-series clinical trial data (especially the values at final time-point), mortality risk, feature importance weights and medical knowledge, and generate an analytical review for the patient."""

clinician_review_user = Template("""
Here is the background of this clinical trial:
{{context}}

Here is the basic clinical information and multivariate time-series clinical trial data of the patient (a structured collection of patient information comprising multiple clinical variables measured at various time points across multiple patient visits):
{{hcontext}}

**Task**: Analyze the patient's mortality risk and confidence based on the provided multivariate time-series clinical data. Use analytic reasoning to synthesize patterns, trends, and correlations across variables (adverse events, vital signs, laboratory results, etc.) to deduce physiologic or biochemical pathophysiology.

**Evaluation Principles**:
1. **Cumulative Impact**: Balance multiple high-grade/recurrent adverse events against recovery evidence and intervention response
2. **Temporal Trends**: Consider long-term effects of adverse events, vital signs trends, and biochemical markers
3. **Risk Prioritization**: Do not underestimate mortality risk based on short-term recovery if severe/repeated multi-organ toxicity exists
4. **Safety Signals**: Highlight organ dysfunction, abnormal vital signs, critical laboratory values
5. **Data Completeness**: Missing or ambiguous data increases mortality risk and uncertainty

**Confidence Levels** (select from): [0.10, 0.20), [0.20, 0.30), [0.30, 0.40), [0.40, 0.50), [0.50, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.90), [0.90, 1.00]
- **0.70-1.00**: Clear cases with complete data across all variables
- **0.40-0.69**: Typical cases with some uncertainties
- **0.10-0.39**: Complex/atypical cases or missing data

**Output Requirements**:
1. **logit**: Mortality risk (0.00-1.00, 2 decimals; 1.00=highest risk, 0.00=lowest risk)
2. **conf_logit**: Confidence level (exact values from specified ranges)
3. **analysis**: Clinical assessment (≤100 words, evidence-based synthesis)
4. **evidences**: 2-3 specific clinical indicators from provided data

**Output Format**:
{"logit": 0.35, "conf_logit": 0.75, "analysis": "Your clinical assessment focusing on key pathophysiology and risk factors.", "evidences": ["Specific evidence 1", "Specific evidence 2", ...]}

Respond in JSON format without additional content:
""")

# =============================================================================
# Phase 2: Clinical Trial Safety Director Meta-Summary
# =============================================================================

meta_summary_system = """You are an authoritative trialist in the medical field organizing a collaborative consultation. Now several clinicians have made analysis and judgments on a patient's condition. Your task is to analyze the rationality of each clinician's opinion, summarize the opinions to obtain a synthesized report for the patient and give your judgment on the patient's mortality risk. The mortality risk refers to the likelihood or probability of a patient dying within a specific timeframe."""

meta_summary_user = Template("""First, please read the patient's basic clinical trial information carefully, as follows:
{{patient_info}}

Then, all clinicians made adverse event analyses and mortality risk assessments for this patient.
In each clinician's statement, they analyzed temporal patterns and severity of adverse events, evaluated both risk signals and recovery signals, then made a prediction about the mortality risk of the patient with mortality risk scores (where higher scores indicate higher mortality risk). They also collected relevant medical literature as evidence for their judgment.
The following are their analyses:
{{clinician_info}}

**Expert Assessment Principles**:
- Balance predictive accuracy with clinical relevance, ensuring alignment with regulatory standards and patient safety priorities
- Evaluate both risk and recovery signals, intervention effects, and patient-specific factors
- Identify rare or atypical adverse reaction patterns, considering cumulative effects and long-term trends
- Transform complex safety and efficacy data into actionable clinical decisions prioritizing patient outcomes
- Ensure compliance with ICH-GCP guidelines and regulatory requirements

**Task**: Analyze all clinicians' opinions for rationality and helpfulness in assessing the patient's condition.

**Output Requirements**:
1. **Direct Assessment**: State whether the patient has high or low mortality risk
2. **Safety Assessment**: Comprehensive summary of adverse event profile and critical characteristics. Use analytic reasoning to deduce physiologic/biochemical pathophysiology underlying adverse events. List important safety signals and their impact on patient mortality risk. Focus on critical factors rather than all provided features.
3. **Supporting Evidence**: Detailed content from clinicians' analyses and evidence. Include substantial content, not just clinician names.

**Output Format** (≤120 words):
{"answer":"In my opinion, the patient has a high risk of mortality.", "report": "Comprehensive safety assessment with critical adverse event patterns...", "evidences": ["Evidence 1 with detailed content...", "Evidence 2 with detailed content..."]}
{"answer":"In my opinion, the patient has a low risk of mortality.", "report": "Comprehensive safety assessment showing manageable adverse event profile...", "evidences": ["Evidence 1 with detailed content...", "Evidence 2 with detailed content..."]}

Respond in JSON format without any additional content:
""")

# =============================================================================
# Phase 3: Clinician Collaboration and Consensus
# =============================================================================

clinician_collaboration_system = """You are an experienced medical expert and supervisor participating in a consultation with several other medical clinicians for a patient. The trialist of this consultation has generated a synthesized report based on all clinicians' analysis of the patient. Please provide your viewpoint on his opinion."""

clinician_collaboration_user = Template("""
Here is the background of this clinical trial:
{{context}}

Here is your previous analysis of the patient, which may require adjustment based on the trialist's assessment:
{{analysis}}

Here is the trialist's opinion:
{{opinion}}

Here is the synthesized report generated by the trialist:
{{report}}

**Task**: Provide your professional opinion on the trialist's synthesized report regarding the patient's mortality risk assessment.

**Response Requirements**:
1. **Stance**: Your position on the trialist's mortality risk assessment (high or low) - respond with "agree" or "disagree"
2. **Confidence Level** (1-3):
   - **3 (High)**: Subject matter expert with extensive medical domain knowledge and high confidence in assessments
   - **2 (Moderate)**: Solid understanding with familiarity in medical domain and confident assessment capability  
   - **1 (Low)**: Basic knowledge with some familiarity, understanding main points but acknowledging expertise limitations
3. **Reasoning**: Base assessment on your previously provided mortality risk and confidence level to determine consistency with trialist's opinion. If consistent, maintain original assessment.
4. **Opinion Modification** (if applicable): If you modify your opinion, update your mortality risk assessment and confidence level. Include revised mortality risk (denoted as 'new_logit': 0.00-1.00, two decimals, higher means higher mortality risk) and confidence level (denoted as 'new_conf_logit': 0.10-1.00, two decimals, higher means higher confidence). Clearly indicate assessment change with detailed rationale. Identify specific aspects where previous conclusion was incorrect or important overlooked factors.

**Output Format Examples** (reason ≤100 words):
{"answer": "agree", "confidence": 3, "opinion": "changed", "reason": "Detailed reasoning for your professional opinion.", "new_logit": 0.75, "new_conf_logit": 0.65}
{"answer": "agree", "confidence": 2, "opinion": "not_changed", "reason": "Detailed reasoning for your professional opinion."}
{"answer": "disagree", "confidence": 1, "opinion": "not_changed", "reason": "Detailed reasoning for your professional opinion."}

Respond in JSON format without any additional content:
""")

# =============================================================================
# Action and Summary Templates
# =============================================================================

action_system = """You are an authoritative expert and supervisor in the medical field. You are organizing a collaborative consultation. Now several clinicians have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each clinician."""

action_user = Template("""In response to the patient's synthesized report, several clinicians put forward their own opinions and reasons. 
In each clinician's statement, they first express whether they agree with the previous synthesized report and give the degree of confidence in their own judgment. Then, they further elaborate on their views by stating reasons and listing relevant literature. 
The following are their opinions:
{{opinions}}                

Next, you need to judge whether the next round of discussion is needed based on each clinician's statement. Considering the following four cases:
1. If all clinicians agree with the previous synthesized report, there is no need to continue the discussion.
2. If some clinicians disagree with the previous report, but they are not confident in their judgment and have not listed convincing evidence, there is no need to continue the discussion.
3. If some clinicians strongly oppose the previous report and you think their evidence is worth discussing, please continue the discussion.
4. If most clinicians disagree with the previous report, please continue the discussion.

Please output the following content in JSON format:
1. Whether to continue the discussion or not.
2. Explain the reasons for your judgment.

Here are two examples of the format you should output:
{"action": "No", "reason": "All clinicians agree with the previous synthesized report"}
{"action": "Yes", "reason": "Exist a few clinicians strongly disagree with the previous synthesized report"}

Respond in JSON format without any additional content:
""")

collaborative_summary_system = """You are an authoritative expert and supervisor in the medical field. You are organizing a collaborative consultation. Now several clinicians have made analysis and judgments on a patient's condition. Your task is to judge whether everyone has reached a consensus on the diagnosis of the patient based on the analysis statements of each clinician and then analyze the rationality of each clinician's opinion and summarize the opinions you think are reasonable to obtain a synthesized report for the patient."""

collaborative_summary_user = Template("""In the previous discussion, you took into account the opinions of all the clinicians and obtained a synthesized report about the patient, which is listed as follows:
{{latest_info}}

In response to the patient's synthesized report, several clinicians put forward their own opinions and reasons.
In each clinician's statement, they first express whether they agree with your statement in the previous synthesized report and give the confidence level on their own judgment. Then, they further elaborate on their views by stating reasons and listing relevant literatures.
The following are their opinions:
{{clinician_info}}

**Decision Framework**:
1. **All clinicians agree** with previous report → No further discussion needed
2. **Some disagree** but low confidence and weak evidence → No further discussion needed  
3. **Some strongly oppose** with compelling evidence → Continue discussion
4. **Most disagree** with previous report → Continue discussion

**If Discussion Continues**: Analyze rationality of each clinician's opinions and summarize reasonable opinions for synthesized patient report:
1. **Strong opposition** with valid reasoning → Focus on their arguments, reconsider mortality risk assessment and modify report accordingly
2. **Opposition with doubts** about own opinion → Consider their view but may maintain original opinion
3. **Agreement** → No modification needed based on their opinion

**Output Requirements**:
1. **Action**: Continue discussion? Respond with "Yes" or "No"
2. **Assessment**: Statement of patient's mortality risk level (high or low)
3. **Revised Report**: Concise summary of patient's health status, informative for mortality prediction tasks. Use analytic reasoning to deduce physiologic/biochemical pathophysiology. List important features and their impact on patient's condition. Avoid listing all clinician-provided features.
4. **Revision Reasons**: Which clinician's opinion or literature referenced, and which original opinions modified. Include detailed content, not just clinician names.

**Output Format** (answer ≤120 words):
{"action": "No", "answer": "In my opinion, the patient has a high risk of mortality.", "report": "...", "reasons": ["reason1 ...", "reason2 ..."]}
{"action": "Yes", "answer": "In my opinion, the patient has a low risk of mortality.", "report": "...", "reasons": ["reason1 ...", "reason2 ..."]}

Respond in JSON format without any additional content:
""")

revise_logits_system = """You are an authoritative expert and supervisor in the medical field organizing a collaborative consultation. After the discussion, you will get a consistent analysis report on the patient. Your task is to refer to each clinician's initial diagnosis and the conclusion of the analysis report to output your final prediction of the patient's mortality risk."""

revise_logits_user = Template("""Initially, all clinicians made predictions about the mortality risk of the patient and explained their reasons, as shown below:
{{clinician_info}}

Then, in a collaborative consultation, you organized all clinicians to get a consistent analysis report on the patient. The report includes:
1. Your judgment on mortality risk of the patient
2. Reasons and evidence to support your conclusion
As follows:
{{latest_info}}

**Task**: 
- Directly copy mortality risk predictions by clinicians based on collaborative consultation results. (Format: He thinks the mortality risk of the patient is: xx, with confidence yy.)
- Give your final prediction and confidence level of the patient's mortality risk based on your previous consultation results and predictions by clinicians. (Denote as "meta_logit" and "meta_conf".)

**Prediction Guidelines (meta_logit)**: 
- Logically sound and well-reasoned (rounded to 2 decimal places)
- 1 indicates highest mortality risk, 0 indicates lowest mortality risk
- Higher values represent higher mortality risk
                          
**Confidence Principles (meta_conf)**:
- **High confidence predictions**: Give greater weight to respecting original predictions
- **Low confidence predictions**: Greater authority to adjust based on analysis and consultation results

**Output Format** (Use clinician to refer to clinician, start from clinician0):
{"clinician0_logit": ..., "clinician0_conf": ..., "clinician1_logit": ..., "clinician1_conf": ..., "clinician2_logit": ..., "clinician2_conf": ..., "meta_logit": ..., "meta_conf": ...}

Respond in JSON format without any additional content:
""")

# =============================================================================
# Clinician Response Templates
# =============================================================================

last_round_clinician_response = Template("""In the last round, you thought the mortality risk of the patient was: {{diagnosis}}, with confidence {{conf_logit}}.""")

initial_clinician_for_revise = Template("""He thinks the mortality risk of the patient is: {{diagnosis}}, with confidence {{conf_logit}}.
Then, he briefly analyzed the patient's basic condition: {{analysis}}""")

initial_clinician_summary = Template("""He thinks the mortality risk of the patient is: {{diagnosis}}, with confidence {{conf_logit}}.
He first briefly analyzed the patient's basic condition: {{analysis}}
Then, he listed some evidence as the basis for my diagnosis of this patient: {{evidence}}""")

# Confidence-based response templates
CONFIDENCE_TEMPLATES = {
    'agree_certain': Template("""Regarding the summary report of the supervisor expert, I agree with the supervisor, with high confidence.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}"""),
    
    'agree_uncertain': Template("""Regarding the summary report of the supervisor expert, I agree with the supervisor, but am not very sure.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}"""),
    
    'disagree_certain': Template("""Regarding the summary report of the supervisor expert, I disagree with the supervisor, with high confidence.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I think my opinions and evidence are very important for accurate judgments. I hope the supervisor expert can consider them more."""),
    
    'disagree_uncertain': Template("""Regarding the summary report of the supervisor expert, I disagree with the supervisor, but am not very sure.
The reason to support my statement is announced below:
{{reason}}
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have doubts about the summary report of the supervisor expert, but I am not sure about my own opinions either. I have listed some evidence as a supplementary reference.""")
}

# Confidence level templates
CONFIDENCE_LEVEL_TEMPLATES = {
    'high': Template("""Regarding the summary report of the supervisor expert, I {{opinion}} with the supervisor.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I am an expert in the subject area and have extensive knowledge in the medical domain. I am highly confident in my ability to provide an accurate and thorough assessment. My evaluation is based on deep expertise and a comprehensive understanding of the work."""),
    
    'moderate': Template("""Regarding the summary report of the supervisor expert, I {{opinion}} with the supervisor.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have a good understanding of the subject area and is familiar with the medical domain. I feel confident in your ability to accurately assess the quality and significance of the work. My evaluation is based on a solid grasp of the content and context."""),
    
    'low': Template("""Regarding the summary report of the supervisor expert, I {{opinion}} with the supervisor.
To further support my idea, I find the extra relevant documents listed below:
{{documents}}
I have some knowledge of the subject area and is somewhat familiar with the medical domain. I understand the main points but may lack depth in certain areas. I am reasonably confident in my assessment but acknowledges some limitations in my expertise.""")
}

# Legacy aliases for backward compatibility
clinician_vote_agree_certain = CONFIDENCE_TEMPLATES['agree_certain']
clinician_vote_agree_uncertain = CONFIDENCE_TEMPLATES['agree_uncertain']
clinician_vote_disagree_certain = CONFIDENCE_TEMPLATES['disagree_certain']
clinician_vote_disagree_uncertain = CONFIDENCE_TEMPLATES['disagree_uncertain']
clinician_high_confidence = CONFIDENCE_LEVEL_TEMPLATES['high']
clinician_moderate_confidence = CONFIDENCE_LEVEL_TEMPLATES['moderate']
clinician_low_confidence = CONFIDENCE_LEVEL_TEMPLATES['low']
