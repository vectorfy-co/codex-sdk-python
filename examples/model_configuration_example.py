#!/usr/bin/env python3
"""
Model Configuration and API Endpoints Example

This example demonstrates how to:
1. Use different Codex models (gpt-5-codex, gpt-5 variants)
2. Configure OpenAI-compatible API endpoints
3. Set up custom model providers
4. Compare different model behaviors
"""

import asyncio

from codex_sdk import Codex, ThreadOptions


async def test_different_models():
    """Test different Codex models and their characteristics."""
    print("=" * 60)
    print("TESTING DIFFERENT CODEX MODELS")
    print("=" * 60)

    # Available models
    models = [
        ("gpt-5-codex", "low", "Fastest Codex model"),
        ("gpt-5-codex", "medium", "Balanced Codex model"),
        ("gpt-5-codex", "high", "Most capable Codex model (current default)"),
        ("gpt-5", "minimal", "Fastest responses, limited reasoning"),
        ("gpt-5", "low", "Balances speed with reasoning"),
        ("gpt-5", "medium", "Default GPT-5, solid balance"),
        ("gpt-5", "high", "Maximizes reasoning depth"),
    ]

    codex = Codex()

    for model, effort, description in models:
        print(f"\n{'-' * 40}")
        print(f"Testing: {model} ({effort})")
        print(f"Description: {description}")
        print(f"{'-' * 40}")

        # Create thread with specific model
        thread_options = ThreadOptions(
            model=f"{model}-{effort}",
            sandbox_mode="danger-full-access",
            working_directory=".",
            skip_git_repo_check=True,
        )

        thread = codex.start_thread(thread_options)

        # Test with a coding task
        print("Task: Write a simple Python function to calculate fibonacci numbers")
        turn = await thread.run(
            "Write a simple Python function to calculate fibonacci numbers. Make it efficient and well-documented."
        )

        print("Response:")
        try:
            # Show first 200 characters of response
            response_preview = (
                turn.final_response[:200] + "..."
                if len(turn.final_response) > 200
                else turn.final_response
            )
            print(response_preview)
        except UnicodeEncodeError:
            safe_response = turn.final_response.encode(
                "utf-8", errors="replace"
            ).decode("utf-8")
            response_preview = (
                safe_response[:200] + "..."
                if len(safe_response) > 200
                else safe_response
            )
            print(response_preview)

        # Show performance metrics
        print(f"Items processed: {len(turn.items)}")
        if turn.usage:
            print(
                f"Token usage: {turn.usage.input_tokens} input, {turn.usage.output_tokens} output"
            )


async def test_openai_compatible_endpoints():
    """Test configuring OpenAI-compatible API endpoints."""
    print("\n" + "=" * 60)
    print("OPENAI-COMPATIBLE API ENDPOINTS")
    print("=" * 60)

    # Example configurations for different providers
    endpoint_configs = [
        {
            "name": "OpenAI Direct",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o",
            "env_key": "OPENAI_API_KEY",
        },
        {
            "name": "Azure OpenAI",
            "base_url": "https://your-resource.openai.azure.com/openai",
            "model": "gpt-4o",
            "env_key": "AZURE_OPENAI_API_KEY",
        },
        {
            "name": "Local Ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "llama3.1",
            "env_key": None,
        },
        {
            "name": "Anthropic Claude (via OpenAI-compatible proxy)",
            "base_url": "https://api.anthropic.com/v1",
            "model": "claude-3-sonnet-20240229",
            "env_key": "ANTHROPIC_API_KEY",
        },
    ]

    print("Available OpenAI-compatible endpoint configurations:")
    for i, config in enumerate(endpoint_configs, 1):
        print(f"\n{i}. {config['name']}:")
        print(f"   Base URL: {config['base_url']}")
        print(f"   Model: {config['model']}")
        print(f"   API Key: {config['env_key'] or 'Not required'}")

    print("\n" + "-" * 40)
    print("To use these endpoints, you would configure them in ~/.codex/config.toml:")
    print("-" * 40)

    config_example = """
# Example config.toml for OpenAI-compatible endpoints

# Use a custom model provider
model = "gpt-4o"
model_provider = "openai-direct"

# Define custom model providers
[model_providers.openai-direct]
name = "OpenAI Direct"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
wire_api = "chat"

[model_providers.azure-openai]
name = "Azure OpenAI"
base_url = "https://your-resource.openai.azure.com/openai"
env_key = "AZURE_OPENAI_API_KEY"
wire_api = "chat"
query_params = { "api-version" = "2024-02-15-preview" }

[model_providers.ollama-local]
name = "Local Ollama"
base_url = "http://localhost:11434/v1"
wire_api = "chat"

[model_providers.anthropic-proxy]
name = "Anthropic Claude"
base_url = "https://api.anthropic.com/v1"
env_key = "ANTHROPIC_API_KEY"
wire_api = "chat"
"""

    print(config_example)


async def test_model_comparison():
    """Compare different models on the same task."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TEST")
    print("=" * 60)

    # Test different models on the same task
    models_to_compare = [
        "gpt-5-codex-low",
        "gpt-5-codex-medium",
        "gpt-5-codex-high",
        "gpt-5-minimal",
        "gpt-5-medium",
    ]

    task = "Explain the difference between Python's list and tuple data structures. Provide examples and use cases."

    codex = Codex()

    results = {}

    for model in models_to_compare:
        print(f"\nTesting {model}...")

        thread_options = ThreadOptions(
            model=model,
            sandbox_mode="read-only",
            working_directory=".",
            skip_git_repo_check=True,
        )

        thread = codex.start_thread(thread_options)
        turn = await thread.run(task)

        results[model] = {
            "response_length": len(turn.final_response),
            "items_count": len(turn.items),
            "usage": turn.usage,
        }

        print(f"  Response length: {len(turn.final_response)} characters")
        print(f"  Items processed: {len(turn.items)}")
        if turn.usage:
            print(
                f"  Tokens: {turn.usage.input_tokens} input, {turn.usage.output_tokens} output"
            )

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"{'Model':<20} {'Response Length':<15} {'Items':<8} {'Input Tokens':<12} {'Output Tokens':<12}"
    )
    print("-" * 70)

    for model, data in results.items():
        input_tokens = data["usage"].input_tokens if data["usage"] else "N/A"
        output_tokens = data["usage"].output_tokens if data["usage"] else "N/A"
        print(
            f"{model:<20} {data['response_length']:<15} {data['items_count']:<8} {input_tokens:<12} {output_tokens:<12}"
        )


async def demonstrate_model_selection_guidelines():
    """Show guidelines for selecting the right model."""
    print(f"\n{'=' * 60}")
    print("MODEL SELECTION GUIDELINES")
    print(f"{'=' * 60}")

    guidelines = """
CHOOSING THE RIGHT MODEL:

GPT-5-CODEX MODELS (Specialized for coding):
- gpt-5-codex-low:    Fast coding tasks, simple scripts, quick fixes
- gpt-5-codex-medium: Balanced coding tasks, moderate complexity
- gpt-5-codex-high:   Complex coding tasks, architecture decisions, debugging

GPT-5 MODELS (General purpose):
- gpt-5-minimal:      Fast responses, simple tasks, basic Q&A
- gpt-5-low:          Quick explanations, straightforward queries
- gpt-5-medium:       General tasks, balanced reasoning
- gpt-5-high:         Complex problems, deep reasoning, analysis

PERFORMANCE CHARACTERISTICS:
- Speed: minimal > low > medium > high
- Reasoning: minimal < low < medium < high
- Cost: minimal < low < medium < high (typically)

USE CASE RECOMMENDATIONS:
- Quick coding fixes: gpt-5-codex-low
- Code reviews: gpt-5-codex-medium
- System architecture: gpt-5-codex-high
- Simple questions: gpt-5-minimal
- General tasks: gpt-5-medium
- Complex analysis: gpt-5-high

OPENAI-COMPATIBLE ENDPOINTS:
- Use for custom models (Claude, Llama, etc.)
- Configure in ~/.codex/config.toml
- Supports any OpenAI-compatible API
- Great for cost optimization or specific model requirements
"""

    print(guidelines)


async def main():
    """Main function to run all model configuration tests."""
    print("Codex Python SDK - Model Configuration and API Endpoints")
    print("=" * 60)

    try:
        await test_different_models()
        await test_openai_compatible_endpoints()
        await test_model_comparison()
        await demonstrate_model_selection_guidelines()

        print(f"\n{'=' * 60}")
        print("MODEL CONFIGURATION TEST COMPLETED!")
        print(f"{'=' * 60}")
        print(
            """
Key Takeaways:
- Choose models based on task complexity and speed requirements
- Codex models are optimized for coding tasks
- GPT-5 models are general purpose with different reasoning levels
- OpenAI-compatible endpoints allow using custom models
- Configure endpoints in ~/.codex/config.toml
- Test different models to find the best fit for your use case
"""
        )

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
