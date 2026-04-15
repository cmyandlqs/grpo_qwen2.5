#!/bin/bash
# Check Swift installation

echo "Checking Swift installation..."

# Check if swift command exists
if ! command -v swift &> /dev/null; then
    echo "Error: Swift is not installed or not in PATH"
    echo "Please install Swift from: https://swift.org/download/"
    exit 1
fi

# Get Swift version
echo "Swift found!"
echo ""
swift --version
echo ""

# Additional diagnostics
echo "Swift binary location: $(which swift)"
echo ""

# Check swiftc if needed
if command -v swiftc &> /dev/null; then
    echo "Swift compiler (swiftc) is also available"
fi

echo ""
echo "Swift installation check passed!"
