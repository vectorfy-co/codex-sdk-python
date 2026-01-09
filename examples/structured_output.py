#!/usr/bin/env python3
"""
Structured output example of the Codex Python SDK.

This example demonstrates how to:
1. Use structured output with JSON schemas
2. Parse structured responses
3. Work with schema validation
"""

import asyncio

from codex_sdk import Codex


async def main():
    """Main function demonstrating structured output usage."""
    print("Starting Codex SDK structured output example...")

    # Define a JSON schema for the expected output
    schema = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A brief summary of the analysis",
            },
            "status": {
                "type": "string",
                "enum": ["ok", "warning", "error"],
                "description": "The overall status of the analysis",
            },
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["error", "warning", "info"],
                        },
                        "message": {"type": "string"},
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                    },
                    "required": ["type", "message"],
                },
            },
            "recommendations": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "status"],
        "additionalProperties": False,
    }

    # Create a Codex instance
    codex = Codex()

    # Start a new thread
    thread = codex.start_thread()
    print(f"Started thread with ID: {thread.id}")

    # Run a command with structured output
    print(
        "Running command with structured output: 'Analyze the current Python project for code quality issues'"
    )

    result = await thread.run_json(
        "Analyze the current Python project for code quality issues",
        output_schema=schema,
    )

    structured_response = result.output
    turn = result.turn

    print("\nStructured Response:")
    print(f"Summary: {structured_response['summary']}")
    print(f"Status: {structured_response['status']}")

    if "issues" in structured_response:
        print(f"\nIssues found: {len(structured_response['issues'])}")
        for i, issue in enumerate(structured_response["issues"]):
            print(f"  {i + 1}. [{issue['type'].upper()}] {issue['message']}")
            if "file" in issue:
                print(f"     File: {issue['file']}")
            if "line" in issue:
                print(f"     Line: {issue['line']}")

    if "recommendations" in structured_response:
        print(f"\nRecommendations: {len(structured_response['recommendations'])}")
        for i, rec in enumerate(structured_response["recommendations"]):
            print(f"  {i + 1}. {rec}")

    print(f"\nTotal items processed: {len(turn.items)}")
    if turn.usage:
        print(
            f"Token usage: {turn.usage.input_tokens} input, {turn.usage.output_tokens} output"
        )


if __name__ == "__main__":
    asyncio.run(main())
