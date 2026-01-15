#!/usr/bin/env python3
"""
Setup script for the Codex Python SDK.

This script downloads the real codex binary from the npm package and sets it up
for use with the Python SDK.
"""

import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


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


def download_codex_package():
    """Download the codex-sdk npm package."""
    print("Downloading codex-sdk package...")

    # Create a temporary directory for the download
    temp_dir = Path(tempfile.mkdtemp(prefix="codex-setup-"))
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Download the package
        run_command(["npm", "pack", "@openai/codex-sdk"], cwd=temp_dir)

        # Find the downloaded tarball
        tarball_files = list(temp_dir.glob("*.tgz"))
        if not tarball_files:
            raise RuntimeError("No tarball found after npm pack")

        tarball_path = tarball_files[0]
        print(f"Downloaded: {tarball_path.name}")

        # Extract the tarball
        print("Extracting package...")
        run_command(["tar", "-xzf", str(tarball_path)], cwd=temp_dir)

        # Find the extracted package directory
        package_dirs = [
            d for d in temp_dir.iterdir() if d.is_dir() and d.name.startswith("package")
        ]
        if not package_dirs:
            raise RuntimeError("No package directory found after extraction")

        package_dir = package_dirs[0]
        print(f"Extracted to: {package_dir}")

        return package_dir

    except Exception as e:
        print(f"ERROR: Error downloading package: {e}")
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def setup_vendor_directory(package_dir, sdk_dir):
    """Copy the vendor directory from the package to the SDK."""
    print("Setting up vendor directory...")

    vendor_src = package_dir / "vendor"
    vendor_dest = sdk_dir / "src" / "codex_sdk" / "vendor"

    if not vendor_src.exists():
        raise RuntimeError("Vendor directory not found in downloaded package")

    # Remove existing vendor directory if it exists
    if vendor_dest.exists():
        print("Removing existing vendor directory...")
        shutil.rmtree(vendor_dest)

    # Copy the vendor directory
    print(f"Copying vendor directory from {vendor_src} to {vendor_dest}")
    shutil.copytree(vendor_src, vendor_dest)

    # Verify the copy
    if not vendor_dest.exists():
        raise RuntimeError("Failed to copy vendor directory")

    print("SUCCESS: Vendor directory set up successfully")

    # Show what platforms are available
    platforms = [d.name for d in vendor_dest.iterdir() if d.is_dir()]
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

    try:
        # Check dependencies
        if not check_dependencies():
            return 1

        # Download the package
        package_dir = download_codex_package()

        # Setup vendor directory
        vendor_dir = setup_vendor_directory(package_dir, sdk_dir)

        # Verify binary for current platform
        binary_path = verify_binary_for_current_platform(vendor_dir)

        # Test the binary
        test_binary(binary_path)

        # Clean up temporary directory
        temp_dir = package_dir.parent
        print(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)

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


if __name__ == "__main__":
    sys.exit(main())
