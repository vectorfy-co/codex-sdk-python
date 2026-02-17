#!/usr/bin/env python3
"""Download and install vendored Codex CLI binaries for the Python SDK."""

import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

TARGET_TO_SUFFIX = {
    "aarch64-apple-darwin": "darwin-arm64",
    "x86_64-apple-darwin": "darwin-x64",
    "aarch64-unknown-linux-musl": "linux-arm64",
    "x86_64-unknown-linux-musl": "linux-x64",
    "aarch64-pc-windows-msvc": "win32-arm64",
    "x86_64-pc-windows-msvc": "win32-x64",
}


def run_command(cmd, cwd=None, check=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=check, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        raise


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")

    # Check if npm is available
    try:
        result = run_command(["npm", "--version"], check=False)
        if result.returncode != 0:
            print("ERROR: npm is not installed. Please install Node.js and npm first.")
            print("   You can install it with: conda install nodejs")
            return False
        print(f"OK: npm version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: npm is not found. Please install Node.js and npm first.")
        print("   You can install it with: conda install nodejs")
        return False

    return True


def resolve_codex_version() -> str:
    """Resolve the Codex npm version to install."""
    requested = os.environ.get("CODEX_NPM_VERSION", "").strip()
    if requested:
        print(f"Using requested CODEX_NPM_VERSION: {requested}")
        return requested

    print("Resolving latest @openai/codex version...")
    result = run_command(["npm", "view", "@openai/codex", "version"])
    version = result.stdout.strip()
    if not version:
        raise RuntimeError("Could not resolve @openai/codex version from npm")
    print(f"Resolved @openai/codex version: {version}")
    return version


def extract_tarball(tarball_path: Path, destination: Path) -> Path:
    """Extract a tarball and return the extracted package directory."""
    destination.mkdir(parents=True, exist_ok=True)
    run_command(["tar", "-xzf", str(tarball_path), "-C", str(destination)])
    package_dir = destination / "package"
    if not package_dir.exists():
        raise RuntimeError(f"No package directory found in {tarball_path.name}")
    return package_dir


def download_codex_packages(version: str) -> Tuple[Path, List[Path]]:
    """Download package(s) that contain vendor binaries for all target platforms."""
    print("Downloading Codex npm packages...")

    temp_dir = Path(tempfile.mkdtemp(prefix="codex-setup-"))
    print(f"Using temporary directory: {temp_dir}")

    package_dirs: List[Path] = []

    # Older releases bundled all targets in @openai/codex-sdk. Keep this path for
    # backwards compatibility.
    sdk_spec = f"@openai/codex-sdk@{version}"
    sdk_pack = run_command(["npm", "pack", sdk_spec], cwd=temp_dir, check=False)
    if sdk_pack.returncode == 0:
        sdk_tgz = next(temp_dir.glob("openai-codex-sdk-*.tgz"), None)
        if sdk_tgz is not None:
            sdk_package_dir = extract_tarball(sdk_tgz, temp_dir / "codex-sdk")
            if (sdk_package_dir / "vendor").exists():
                print(f"Found vendored binaries in {sdk_spec}")
                package_dirs.append(sdk_package_dir)

    if package_dirs:
        return temp_dir, package_dirs

    print(
        "Falling back to platform-specific @openai/codex artifacts "
        "(new packaging format)."
    )
    for target, suffix in TARGET_TO_SUFFIX.items():
        spec = f"@openai/codex@{version}-{suffix}"
        print(f"Downloading {spec} for {target}")
        run_command(["npm", "pack", spec], cwd=temp_dir)
        tarball = temp_dir / f"openai-codex-{version}-{suffix}.tgz"
        if not tarball.exists():
            matches = list(temp_dir.glob(f"openai-codex-*{suffix}.tgz"))
            if len(matches) != 1:
                raise RuntimeError(f"Could not find tarball for {spec}")
            tarball = matches[0]
        package_dir = extract_tarball(tarball, temp_dir / f"platform-{suffix}")
        vendor_dir = package_dir / "vendor"
        if not vendor_dir.exists():
            raise RuntimeError(f"Vendor directory not found in {spec}")
        package_dirs.append(package_dir)

    return temp_dir, package_dirs


def setup_vendor_directory(package_dirs: List[Path], sdk_dir: Path):
    """Copy vendor directories from downloaded packages to the SDK."""
    print("Setting up vendor directory...")
    vendor_dest = sdk_dir / "src" / "codex_sdk" / "vendor"

    # Remove existing vendor directory if it exists
    if vendor_dest.exists():
        print("Removing existing vendor directory...")
        shutil.rmtree(vendor_dest)

    vendor_dest.mkdir(parents=True, exist_ok=True)

    copied_targets = set()
    for package_dir in package_dirs:
        vendor_src = package_dir / "vendor"
        if not vendor_src.exists():
            raise RuntimeError(f"Vendor directory not found in {package_dir}")
        for target_dir in sorted(p for p in vendor_src.iterdir() if p.is_dir()):
            destination = vendor_dest / target_dir.name
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(target_dir, destination)
            copied_targets.add(target_dir.name)

    # Verify the copy
    if not vendor_dest.exists():
        raise RuntimeError("Failed to copy vendor directory")

    missing_targets = sorted(set(TARGET_TO_SUFFIX.keys()) - copied_targets)
    if missing_targets:
        raise RuntimeError(
            "Missing vendored targets after copy: " + ", ".join(missing_targets)
        )

    print("SUCCESS: Vendor directory set up successfully")

    # Show what platforms are available
    platforms = sorted(d.name for d in vendor_dest.iterdir() if d.is_dir())
    print(f"Available platforms: {', '.join(platforms)}")

    return vendor_dest


def verify_binary_for_current_platform(vendor_dir):
    """Verify that the binary exists for the current platform."""
    print("Verifying binary for current platform...")

    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform to target triple
    target_triple = None
    if system == "linux":
        if machine in ["x86_64", "amd64"]:
            target_triple = "x86_64-unknown-linux-musl"
        elif machine in ["aarch64", "arm64"]:
            target_triple = "aarch64-unknown-linux-musl"
    elif system == "darwin":
        if machine in ["x86_64", "amd64"]:
            target_triple = "x86_64-apple-darwin"
        elif machine in ["aarch64", "arm64"]:
            target_triple = "aarch64-apple-darwin"
    elif system == "windows":
        if machine in ["x86_64", "amd64"]:
            target_triple = "x86_64-pc-windows-msvc"
        elif machine in ["aarch64", "arm64"]:
            target_triple = "aarch64-pc-windows-msvc"

    if not target_triple:
        raise RuntimeError(f"Unsupported platform: {system} ({machine})")

    print(f"Current platform: {system} ({machine})")
    print(f"Target triple: {target_triple}")

    binary_name = "codex.exe" if system == "windows" else "codex"
    binary_path = vendor_dir / target_triple / "codex" / binary_name

    if not binary_path.exists():
        raise RuntimeError(f"Binary not found for current platform: {binary_path}")

    # Get binary size
    size_mb = binary_path.stat().st_size / (1024 * 1024)
    print(f"SUCCESS: Binary found: {binary_path}")
    print(f"   Size: {size_mb:.1f} MB")

    return binary_path


def test_binary(binary_path):
    """Test that the binary works."""
    print("Testing binary...")

    try:
        result = run_command([str(binary_path), "--version"], check=False)
        if result.returncode == 0:
            print(f"SUCCESS: Binary works! Version: {result.stdout.strip()}")
        else:
            print(f"WARNING: Binary returned non-zero exit code: {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"WARNING: Could not test binary: {e}")


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print()
    print("Next steps:")
    print()
    print("1. Authenticate with Codex:")
    print("   Run: codex login")
    print("   (This will open a browser for authentication)")
    print()
    print("2. Test the SDK:")
    print("   python examples/basic_usage.py")
    print()
    print("3. Try other examples:")
    print("   python examples/streaming_example.py")
    print("   python examples/thread_resume.py")
    print()
    print("4. Use in your code:")
    print("   ```python")
    print("   import asyncio")
    print("   from codex_sdk import Codex")
    print()
    print("   async def main():")
    print("       codex = Codex()")
    print("       thread = codex.start_thread()")
    print("       turn = await thread.run('Hello, Codex!')")
    print("       print(turn.final_response)")
    print()
    print("   asyncio.run(main())")
    print("   ```")
    print()
    print("For more information, see README.md")
    print("=" * 60)


def main():
    """Main setup function."""
    print("Codex Python SDK Setup")
    print("=" * 40)
    print()

    # Get the SDK directory (where this script is located)
    sdk_dir = Path(__file__).resolve().parent.parent
    print(f"SDK directory: {sdk_dir}")

    temp_dir = None
    try:
        # Check dependencies
        if not check_dependencies():
            return 1

        # Resolve target version and download vendor packages
        version = resolve_codex_version()
        temp_dir, package_dirs = download_codex_packages(version)

        # Setup vendor directory
        vendor_dir = setup_vendor_directory(package_dirs, sdk_dir)

        # Verify binary for current platform
        binary_path = verify_binary_for_current_platform(vendor_dir)

        # Test the binary
        test_binary(binary_path)

        # Print next steps
        print_next_steps()

        return 0

    except Exception as e:
        print(f"\nERROR: Setup failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Node.js and npm installed")
        print("2. Check your internet connection")
        print("3. Try running: conda install nodejs")
        return 1
    finally:
        if temp_dir is not None:
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
