#!/usr/bin/env python3
"""
Setup script for the Codex Python SDK.

This script downloads the real codex binary from the npm package and sets it up
for use with the Python SDK.
"""

import json
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence


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
    """
    Download and extract the codex-sdk npm package into a temporary directory.

    Uses the resolved npm package spec, runs `npm pack` to download a tarball, extracts it, and returns the path to the extracted package directory. On error, the temporary download directory is removed and the exception is re-raised.

    Returns:
        package_dir (Path): Path to the extracted package directory named like "package".

    Raises:
        RuntimeError: If no tarball is found after `npm pack` or no package directory is found after extraction.
        Exception: Propagates other exceptions encountered during download or extraction.
    """
    print("Downloading codex-sdk package...")

    # Create a temporary directory for the download
    temp_dir = Path(tempfile.mkdtemp(prefix="codex-setup-"))
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Download the package
        package_spec = npm_pack_codex_sdk_package(
            temp_dir, resolve_codex_sdk_npm_specs()
        )
        print(f"Downloaded npm package: {package_spec}")

        # Find the downloaded tarball
        tarball_files = list(temp_dir.glob("*.tgz"))
        if not tarball_files:
            raise RuntimeError("No tarball found after npm pack")

        tarball_path = tarball_files[0]
        print(f"Downloaded: {tarball_path.name}")

        # Extract the tarball
        print("Extracting package...")
        _extract_tarball(tarball_path, temp_dir)

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
    vendor_parent = vendor_dest.parent
    vendor_parent.mkdir(parents=True, exist_ok=True)
    staged_vendor_dest = Path(tempfile.mkdtemp(prefix="vendor-new-", dir=vendor_parent))

    try:
        if vendor_src.exists():
            # Copy the vendor directory directly when present in the package.
            print(
                f"Copying vendor directory from {vendor_src} to staging area "
                f"{staged_vendor_dest}"
            )
            shutil.copytree(vendor_src, staged_vendor_dest, dirs_exist_ok=True)
        else:
            print(
                "Vendor directory not found in downloaded package; "
                "assembling vendor binaries from @openai/codex platform packages..."
            )
            _assemble_vendor_from_platform_packages(package_dir, staged_vendor_dest)

        # Verify the staged copy before replacing the current tree.
        if not staged_vendor_dest.exists():
            raise RuntimeError("Failed to stage vendor directory")

        _replace_directory(staged_vendor_dest, vendor_dest)
    except Exception:
        shutil.rmtree(staged_vendor_dest, ignore_errors=True)
        raise

    print("SUCCESS: Vendor directory set up successfully")
    platforms = [d.name for d in vendor_dest.iterdir() if d.is_dir()]
    print(f"Available platforms: {', '.join(platforms)}")

    return vendor_dest


def _replace_directory(staged_dir: Path, dest_dir: Path) -> None:
    """Replace dest_dir with a fully prepared staged_dir, restoring the old tree on failure."""
    backup_dir = None

    try:
        if dest_dir.exists():
            backup_dir = Path(
                tempfile.mkdtemp(prefix=f"{dest_dir.name}-backup-", dir=dest_dir.parent)
            )
            backup_dir.rmdir()
            dest_dir.rename(backup_dir)

        staged_dir.rename(dest_dir)
    except Exception:
        if dest_dir.exists():
            shutil.rmtree(dest_dir, ignore_errors=True)
        if backup_dir is not None and backup_dir.exists():
            backup_dir.rename(dest_dir)
        raise
    finally:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)


def _read_package_json(package_dir: Path) -> dict:
    package_json_path = package_dir / "package.json"
    if not package_json_path.exists():
        raise RuntimeError(
            f"package.json not found in downloaded package: {package_dir}"
        )
    return json.loads(package_json_path.read_text(encoding="utf-8"))


def _resolve_codex_cli_version(package_dir: Path) -> str:
    package_json = _read_package_json(package_dir)
    dependencies = package_json.get("dependencies", {})
    dep_version = dependencies.get("@openai/codex")
    if isinstance(dep_version, str) and dep_version.strip():
        return dep_version.strip().lstrip("^~")

    # Fallback if the downloaded package itself is @openai/codex.
    package_name = package_json.get("name")
    package_version = package_json.get("version")
    if package_name == "@openai/codex" and isinstance(package_version, str):
        return package_version.split("-", 1)[0]

    raise RuntimeError(
        "Could not determine @openai/codex version from downloaded package metadata"
    )


def _assemble_vendor_from_platform_packages(
    package_dir: Path, vendor_dest: Path
) -> None:
    codex_version = _resolve_codex_cli_version(package_dir)
    print(f"Resolved @openai/codex version: {codex_version}")

    # Map codex npm package suffixes to vendor target triples expected by the SDK.
    platform_matrix = {
        "linux-x64": "x86_64-unknown-linux-musl",
        "linux-arm64": "aarch64-unknown-linux-musl",
        "darwin-x64": "x86_64-apple-darwin",
        "darwin-arm64": "aarch64-apple-darwin",
        "win32-x64": "x86_64-pc-windows-msvc",
        "win32-arm64": "aarch64-pc-windows-msvc",
    }

    vendor_dest.mkdir(parents=True, exist_ok=True)
    assembly_dir = package_dir.parent / "platform-vendor-assembly"
    assembly_dir.mkdir(parents=True, exist_ok=True)

    for platform_suffix, target_triple in platform_matrix.items():
        package_spec = f"@openai/codex@{codex_version}-{platform_suffix}"
        print(
            f"Downloading platform package {package_spec} "
            f"for target {target_triple}..."
        )
        before = {p.name for p in assembly_dir.glob("*.tgz")}
        run_command(["npm", "pack", package_spec], cwd=assembly_dir)
        after = list(assembly_dir.glob("*.tgz"))
        new_tarballs = [p for p in after if p.name not in before]
        if not new_tarballs:
            raise RuntimeError(
                f"Failed to locate tarball after npm pack: {package_spec}"
            )
        tarball_path = max(new_tarballs, key=lambda p: p.stat().st_mtime)

        extract_dir = assembly_dir / f"extract-{platform_suffix}"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_tarball(tarball_path, extract_dir)

        platform_vendor_src = extract_dir / "package" / "vendor" / target_triple
        if not platform_vendor_src.exists():
            raise RuntimeError(
                f"Vendor path missing in platform package {package_spec}: {platform_vendor_src}"
            )

        platform_vendor_dest = vendor_dest / target_triple
        if platform_vendor_dest.exists():
            shutil.rmtree(platform_vendor_dest)
        shutil.copytree(platform_vendor_src, platform_vendor_dest)

        tarball_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)


def _extract_tarball(tarball_path: Path, dest_dir: Path) -> None:
    """Extract a gzip-compressed tarball with basic path traversal protection."""
    with tarfile.open(tarball_path, "r:gz") as archive:
        dest_root = dest_dir.resolve()
        for member in archive.getmembers():
            member_path = (dest_dir / member.name).resolve()
            try:
                member_path.relative_to(dest_root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Tarball contains path outside extraction dir: {member.name}"
                ) from exc
        archive.extractall(dest_dir)


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
    """
    Orchestrates the SDK binary setup workflow, performs installation steps, and reports success or failure.

    Performs dependency verification, downloads the codex-sdk npm package, installs the package's vendor files into the SDK tree, verifies and tests the platform-specific codex binary, cleans up temporary files, and prints post-setup instructions and progress/error messages to stdout.

    Returns:
        int: `0` on success, `1` on failure.
    """
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


def npm_pack_codex_sdk_package(temp_dir: Path, package_specs: Sequence[str]) -> str:
    """Run `npm pack` for the first available package spec, falling back from missing exact versions."""
    if not package_specs:
        raise RuntimeError("No npm package specs provided")

    last_error = None
    for index, package_spec in enumerate(package_specs):
        print(f"Using npm package: {package_spec}")
        try:
            run_command(["npm", "pack", package_spec], cwd=temp_dir)
            return package_spec
        except subprocess.CalledProcessError as exc:
            last_error = exc
            is_missing_exact_version = index < len(
                package_specs
            ) - 1 and is_missing_npm_version_error(exc)
            if not is_missing_exact_version:
                raise

            fallback_spec = package_specs[index + 1]
            print(
                "WARNING: "
                f"{package_spec} is not published on npm yet; falling back to "
                f"{fallback_spec}."
            )

    assert last_error is not None
    raise last_error


def is_missing_npm_version_error(error: subprocess.CalledProcessError) -> bool:
    """Return True when npm failed because the requested package version does not exist."""
    combined_output = "\n".join(
        part.strip() for part in (error.stdout or "", error.stderr or "") if part
    )
    return (
        "ETARGET" in combined_output or "No matching version found" in combined_output
    )


def resolve_codex_sdk_npm_specs() -> list[str]:
    """
    Build the npm package specs for @openai/codex-sdk.

    Reads the repository pyproject.toml to find the [project].version. When a
    version is present, the exact npm version is tried first and the unpinned
    package name is kept as a fallback for Python-only patch releases that do
    not have a matching npm publish. If no version is found, only the unpinned
    package name is returned.

    Returns:
        list[str]: Ordered npm package specs to try.
    """
    sdk_dir = Path(__file__).resolve().parent.parent
    pyproject_path = sdk_dir / "pyproject.toml"
    version = read_pyproject_version(pyproject_path)
    if version:
        return [f"@openai/codex-sdk@{version}", "@openai/codex-sdk"]
    return ["@openai/codex-sdk"]


def read_pyproject_version(pyproject_path: Path) -> str:
    """
    Extract the value of [project].version from a pyproject.toml file in a best-effort manner.

    Searches the file for a top-level [project] section and returns the value from a `version = "..."` or `version = '...'` line within that section. This is a simple text-based extraction (no TOML parser) and may not handle complex or nonstandard TOML constructs.

    Parameters:
        pyproject_path (Path): Path to the pyproject.toml file to read.

    Returns:
        str: The version string if found, otherwise an empty string.
    """
    if not pyproject_path.exists():
        return ""

    in_project = False
    version_re = re.compile(r'^version\s*=\s*["\']([^"\']+)["\']\s*$')
    for line in pyproject_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project:
            match = version_re.match(stripped)
            if match:
                return match.group(1)

    return ""


if __name__ == "__main__":
    sys.exit(main())
