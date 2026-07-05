"""Schemas and constants for DP-GPT query plans."""

VALID_FIELDS = {
    "sent_id", "sentence", "id", "text", "lemma", "upos", "xpos", "feats",
    "head", "deprel", "start_char", "end_char",
}

VALID_OPERATORS = {
    "equals", "not_equals", "in", "not_in", "contains", "starts_with", "ends_with",
    "regex", "exists",
}

QUERY_PLAN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "target": {"type": "string", "enum": ["sentence"]},
        "description": {"type": "string"},
        "include": {
            "type": "array",
            "items": {"$ref": "#/$defs/condition"},
        },
        "exclude": {
            "type": "array",
            "items": {"$ref": "#/$defs/condition"},
        },
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["target", "description", "include", "exclude", "notes"],
    "$defs": {
        "condition": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "type": {
                    "type": "string",
                    "enum": ["token_condition", "dependency_condition"],
                },
                "field": {"type": "string"},
                "operator": {"type": "string", "enum": sorted(VALID_OPERATORS)},
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                },
                "list_file": {"type": ["string", "null"]},
                "head": {"type": ["string", "null"]},
                "head_field": {"type": ["string", "null"]},
                "head_operator": {
                    "type": ["string", "null"],
                    "enum": sorted(VALID_OPERATORS) + [None],
                },
                "head_value": {"type": ["string", "number", "null"]},
            },
            "required": [
                "name",
                "type",
                "field",
                "operator",
                "value",
                "list_file",
                "head",
                "head_field",
                "head_operator",
                "head_value",
            ],
        }
    },
}

SYSTEM_INSTRUCTIONS = """
You convert linguistic corpus queries into DP-GPT Query Plans.
Return JSON only. Do not return prose or markdown.
The parsed corpus is a token-level DataFrame with these fields:
sent_id, sentence, id, text, lemma, upos, xpos, feats, head, deprel.
Use token_condition for conditions that can be evaluated on a single token.
Use dependency_condition when a dependent token must be attached to a head token.
For passive auxiliaries, use field='deprel', operator='equals', value='aux:pass', head='root'.
For phrasal particles, use field='deprel', operator='equals', value='compound:prt', head='root'.
If the query mentions a target verb list, use field='lemma', operator='in', list_file='verb_list.txt'.
""".strip()
