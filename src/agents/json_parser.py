"""
JSON Parsing Utilities

Migrated from m-agents/utils/framework.py
Extracts and parses JSON from LLM responses.
"""

import re
import json
from typing import Dict


def extract_and_parse_json(text: str) -> Dict:
    """
    Extract and parse JSON from text.

    Supports multiple JSON formats:
    1. ```json...``` code block format
    2. Incomplete JSON (attempts to complete it)
    3. Raw JSON object
    4. Manual field extraction (for specific error patterns)

    Args:
        text: Text containing JSON

    Returns:
        Parsed dictionary

    Raises:
        ValueError: No valid JSON content found
    """

    def fix_common_json_errors(json_str: str) -> str:
        """Fix common JSON formatting errors."""
        json_str = json_str.strip()

        # Fix missing commas between "value" "key"
        json_str = re.sub(r'"\s+"([a-zA-Z_][a-zA-Z0-9_]*":\s*)', r'", \1', json_str)

        # Fix missing commas between digit and string
        json_str = re.sub(r'(\d)\s+"([a-zA-Z_][a-zA-Z0-9_]*":\s*)', r'\1, "\2', json_str)

        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Quote unquoted string values
        json_str = re.sub(r':\s*([^",\[\]{}]+)(?=\s*[,}])', r': "\1"', json_str)

        return json_str

    def try_parse_json(json_str: str) -> Dict:
        """Try to parse JSON, attempt repair if it fails."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw JSON: {json_str}")

            fixed_json = fix_common_json_errors(json_str)
            try:
                return json.loads(fixed_json)
            except json.JSONDecodeError as e2:
                print(f"Still unparseable after fix: {e2}")
                print(f"Fixed JSON: {fixed_json}")
                raise ValueError(f"Cannot parse JSON, original error: {e}, after fix: {e2}")

    # Method 1: Look for ```json...``` format
    pattern_backticks = r'```json(.*?)```'
    match = re.search(pattern_backticks, text, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
        return try_parse_json(json_string)

    # Method 2: Try to complete incomplete JSON
    pattern_backticks_incomplete = r'```json(.*?)$'
    match = re.search(pattern_backticks_incomplete, text.strip(), re.DOTALL)
    if match:
        json_string = match.group(1).strip() + "}"
        return try_parse_json(json_string)

    # Method 3: Look for JSON object (non-greedy)
    pattern_json_object = r'\{.*?\}'
    match = re.search(pattern_json_object, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        return try_parse_json(json_string)

    # Method 4: Look for larger JSON object (greedy)
    pattern_json_greedy = r'\{.*\}'
    match = re.search(pattern_json_greedy, text, re.DOTALL)
    if match:
        json_string = match.group(0).strip()
        return try_parse_json(json_string)

    # Method 5: Manual extraction for TrialAgents-specific response errors
    if 'new_conf_logit' in text and 'new_logit' in text:
        try:
            result = {}

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
                evidences = [e.strip(' "') for e in evidences_str.split('", "')]
                result['evidences'] = evidences

            if result:
                return result
        except Exception as e:
            print(f"Manual parsing failed: {e}")

    raise ValueError(f"No valid JSON content found. Original text: {text[:200]}...")
