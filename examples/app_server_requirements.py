"""Read requirements.toml constraints via app-server."""

import asyncio

from codex_sdk import AppServerClient, AppServerOptions


async def main() -> None:
    async with AppServerClient(AppServerOptions()) as app:
        resp = await app.config_requirements_read()
        requirements = resp.get("requirements")
        if requirements is None:
            print("No requirements configured.")
            return

        print("Allowed approval policies:", requirements.get("allowedApprovalPolicies"))
        print("Allowed sandbox modes:", requirements.get("allowedSandboxModes"))


if __name__ == "__main__":
    asyncio.run(main())
