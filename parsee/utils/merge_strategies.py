import json
import re
from typing import List, Dict, Any


def merge_list_of_dict(responses: List[str]) -> str:
    """
    Merge multiple LLM responses containing JSON arrays into a single list of dictionaries.
    Handles malformed JSON like unclosed arrays, missing brackets, etc.
    """
    all_transactions = []

    for response in responses:
        transactions = extract_json_from_response(response)
        all_transactions.extend(transactions)

    return json.dumps(all_transactions)


def extract_json_from_response(response: str) -> List[Dict[str, Any]]:
    """
    Extract JSON objects from a single LLM response, handling various malformed cases.
    """
    # Remove markdown code blocks
    json_content = remove_markdown_blocks(response)

    # Try to parse as valid JSON first
    try:
        parsed = json.loads(json_content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # If direct parsing fails, try to fix common issues
    return extract_objects_from_malformed_json(json_content)


def remove_markdown_blocks(text: str) -> str:
    """Remove markdown code block markers."""
    # Remove ```json and ``` markers
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'\n?```', '', text)
    return text.strip()


def extract_objects_from_malformed_json(json_str: str) -> List[Dict[str, Any]]:
    """
    Extract individual JSON objects from malformed JSON string.
    Handles cases like unclosed arrays, missing brackets, etc.
    """
    objects = []

    # Clean up the string
    json_str = json_str.strip()

    # Remove leading/trailing array brackets if present
    json_str = re.sub(r'^\s*\[', '', json_str)
    json_str = re.sub(r'\]\s*,?\s*$', '', json_str)

    # Split by object boundaries - look for },\s*{ pattern
    # This regex finds the boundary between objects
    object_boundaries = re.split(r'}\s*,\s*(?=\{)', json_str)

    for i, obj_str in enumerate(object_boundaries):
        obj_str = obj_str.strip()

        # Ensure the object starts with {
        if not obj_str.startswith('{'):
            obj_str = '{' + obj_str

        # Ensure the object ends with }
        if not obj_str.endswith('}'):
            obj_str = obj_str + '}'

        # Remove trailing comma if present
        obj_str = re.sub(r',\s*}$', '}', obj_str)

        try:
            obj = json.loads(obj_str)
            objects.append(obj)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse object {i + 1}: {e}")
            print(f"Problematic string: {obj_str[:100]}...")
            continue

    return objects