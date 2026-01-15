# Codex Python SDK Setup Guide

This guide will help you set up the Codex Python SDK with the real codex binary.
PyPI distributions do not bundle the binaries, so use this script when working
from source or when you need to download the vendor assets locally.

## Quick Setup

### 1. Install Dependencies

First, make sure you have the required dependencies:

```bash
# Install Node.js and npm (required to download the binary)
conda install nodejs

```

### 2. Run the Setup Script

The setup script will automatically download and configure the real codex binary:

```bash
python scripts/setup_binary.py
```

This script will:
- ✅ Check that npm is installed
- 📦 Download the official codex-sdk npm package
- 📁 Extract the real codex binary for your platform
- 🧪 Test that the binary works
- 🧹 Clean up temporary files

### 3. Authenticate with Codex

After setup, you need to authenticate:

```bash
codex login
```

This will open a browser window for you to log in with your OpenAI account.

### 4. Test the SDK

Run the basic example to make sure everything works:

```bash
python examples/basic_usage.py
```

## Manual Setup (Alternative)

If the automated setup doesn't work, you can set up manually:

### 1. Download the Binary

```bash
# Download the npm package
npm pack @openai/codex-sdk

# Extract it
tar -xzf openai-codex-sdk-*.tgz
```

### 2. Copy the Vendor Directory

```bash
# Copy the vendor directory to your SDK
cp -r package/vendor src/codex_sdk/
```

### 3. Verify the Binary

```bash
# Test the binary for your platform
./src/codex_sdk/vendor/x86_64-pc-windows-msvc/codex/codex.exe --version
```

## Platform Support

The SDK includes binaries for these platforms:

- **Windows**: `x86_64-pc-windows-msvc`, `aarch64-pc-windows-msvc`
- **macOS**: `x86_64-apple-darwin`, `aarch64-apple-darwin`
- **Linux**: `x86_64-unknown-linux-musl`, `aarch64-unknown-linux-musl`

## Troubleshooting

### "npm is not found"
```bash
conda install nodejs
```

### "Binary not found for current platform"
Make sure you're running the setup script from the correct directory (`sdk/python/`).

### "Authentication failed"
Make sure you have a valid OpenAI account and run `codex login` again.

### "Permission denied" (Linux/macOS)
You might need to make the binary executable:
```bash
chmod +x src/codex_sdk/vendor/*/codex/codex
```

## What Gets Installed

After setup, you'll have:

```
sdk/python/
├── src/codex_sdk/
│   ├── vendor/                    # Real codex binaries
│   │   ├── x86_64-pc-windows-msvc/
│   │   ├── aarch64-apple-darwin/
│   │   └── ... (other platforms)
│   ├── codex.py                  # Main SDK code
│   ├── thread.py                 # Thread management
│   └── ...                       # Other SDK files
├── examples/                     # Working examples
├── scripts/
│   └── setup_binary.py          # Vendor download script
└── README.md                    # Main documentation
```

## Next Steps

Once setup is complete, check out:

- 📖 [README.md](README.md) - Full SDK documentation
- 🧪 [examples/](examples/) - Working code examples
- 🧪 [tests/](tests/) - Test suite

Happy coding! 🚀
