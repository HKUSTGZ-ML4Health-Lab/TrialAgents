"""
JSON解析工具模块

从 m-agents/utils/framework.py 迁移
提供从LLM响应中提取和解析JSON的功能
"""

import re
import json
from typing import Dict


def extract_and_parse_json(text: str) -> Dict:
    """
    从文本中提取并解析JSON
    
    支持多种JSON格式：
    1. ```json...``` 代码块格式
    2. 不完整的JSON（尝试补全）
    3. 原始JSON对象
    4. 手动提取关键字段（针对特定错误模式）
    
    Args:
        text: 包含JSON的文本
        
    Returns:
        解析后的字典
        
    Raises:
        ValueError: 无法找到有效的JSON内容
    """
    
    def fix_common_json_errors(json_str: str) -> str:
        """修复常见的JSON错误"""
        # 移除多余的换行符和空格
        json_str = json_str.strip()
        
        # 修复缺少逗号的问题：在 "value" "key" 之间添加逗号
        json_str = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*":\s*)', r'", \1', json_str)
        
        # 修复缺少逗号的问题：在数字和字符串之间
        json_str = re.sub(r'(\d)\s+"([a-zA-Z_][a-zA-Z0-9_]*":\s*)', r'\1, "\2', json_str)
        
        # 修复多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # 确保字符串值被正确引用
        json_str = re.sub(r':\s*([^",\[\]{}]+)(?=\s*[,}])', r': "\1"', json_str)
        
        return json_str
    
    def try_parse_json(json_str: str) -> Dict:
        """尝试解析JSON，如果失败则尝试修复后再解析"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"原始JSON: {json_str}")
            
            # 尝试修复常见错误
            fixed_json = fix_common_json_errors(json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e2:
                print(f"修复后仍然无法解析: {e2}")
                print(f"修复后JSON: {fixed_json}")
                raise ValueError(f"无法解析JSON，原始错误: {e}, 修复后错误: {e2}")
    
    # 方法1: 寻找 ```json...``` 格式
    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        return try_parse_json(json_string)
    
    # 方法2: 尝试补全JSON格式
    pattern_backticks_incomplete = r'```json(.*?)$'
    match = re.search(pattern_backticks_incomplete, text.strip(), re.DOTALL)
    if match:
        json_string = match.group(1).strip() + "}"  # 尝试补全
        return try_parse_json(json_string)
    
    # 方法3: 寻找JSON对象（非贪婪）
    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        return try_parse_json(json_string)
    
    # 方法4: 寻找更完整的JSON对象（贪婪匹配）
    pattern_json_greedy = r'\{.*\}'
    match = re.search(pattern_json_greedy, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        return try_parse_json(json_string)
    
    # 方法5: 尝试从给定的错误示例中提取
    # 特别处理特定错误情况（针对TrialAgents的响应格式）
    if 'new_conf_logit' in text and 'new_logit' in text:
        # 尝试手动构建JSON
        try:
            result = {}
            
            # 寻找关键字段
            answer_match = re.search(r'"answer":\s*"([^"]*)"', text)
            confidence_match = re.search(r'"confidence":\s*(\d+)', text)
            opinion_match = re.search(r'"opinion":\s*"([^"]*)"', text)
            new_logit_match = re.search(r'"new_logit":\s*([0-9.]+)', text)
            new_conf_logit_match = re.search(r'"new_conf_logit":\s*([0-9.]+)', text)
            reason_match = re.search(r'"reason":\s*"([^"]*)"', text)
            evidences_match = re.search(r'"evidences":\s*\[(.*?)\]', text, re.DOTALL)
            
            if answer_match:
                result['answer'] = answer_match.group(1)
            if confidence_match:
                result['confidence'] = int(confidence_match.group(1))
            if opinion_match:
                result['opinion'] = opinion_match.group(1)
            if new_logit_match:
                result['new_logit'] = float(new_logit_match.group(1))
                print(f"New logit: {result['new_logit']}")
            if new_conf_logit_match:
                result['new_conf_logit'] = float(new_conf_logit_match.group(1))
                print(f"New confidence: {result['new_conf_logit']}")
            if reason_match:
                result['reason'] = reason_match.group(1)
            if evidences_match:
                evidences_str = evidences_match.group(1)
                # 简单解析evidences数组
                evidences = [e.strip(' "') for e in evidences_str.split('", "')]
                result['evidences'] = evidences
            
            if result:
                return result
        except Exception as e:
            print(f"手动解析失败: {e}")
    
    raise ValueError(f"无法找到有效的JSON内容。原始文本: {text[:200]}...")
