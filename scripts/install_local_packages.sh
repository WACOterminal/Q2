#!/bin/bash
# install_local_packages.sh

# This script installs all local shared libraries.
# It should be run from the root of the project.

set -e

echo "Installing local shared packages..."

# Find all setup.py files in the shared directory and install them
for setup_file in $(find ./shared -name "setup.py"); do
    dir=$(dirname "$setup_file")
    echo "Building and installing $dir..."
    # Build a wheel file in a temp directory and install it
    # This avoids issues with 'setup.py develop' and permissions.
    pushd "$dir" > /dev/null
    python3 -m build --wheel --outdir dist
    pip install --force-reinstall --no-deps dist/*.whl
    popd > /dev/null
done

echo "Local packages installed successfully." 