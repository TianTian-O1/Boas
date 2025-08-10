#!/bin/bash
# BOAS Release Script
# Creates a release package for distribution

VERSION="1.0.0"
RELEASE_DIR="boas-v${VERSION}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "======================================"
echo "BOAS Release Builder v${VERSION}"
echo "======================================"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf ${RELEASE_DIR} ${RELEASE_DIR}.tar.gz

# Create release directory structure
echo "Creating release structure..."
mkdir -p ${RELEASE_DIR}/{bin,lib,include,examples,docs}

# Build the project
echo "Building BOAS..."
if [ ! -d "build" ]; then
    mkdir build
fi
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../${RELEASE_DIR}
make -j$(nproc)
make install
cd ..

# Copy essential files
echo "Copying files..."
cp -r examples/*.bs ${RELEASE_DIR}/examples/
cp README.md ${RELEASE_DIR}/
cp LICENSE ${RELEASE_DIR}/
cp RELEASE.md ${RELEASE_DIR}/
cp package.json ${RELEASE_DIR}/

# Copy include files
cp -r include/mlirops/*.h ${RELEASE_DIR}/include/
cp -r include/frontend/*.h ${RELEASE_DIR}/include/

# Create minimal documentation
cat > ${RELEASE_DIR}/docs/QUICK_START.md << EOF
# BOAS Quick Start Guide

## Installation

1. Extract the archive:
   \`\`\`bash
   tar -xzf boas-v${VERSION}.tar.gz
   cd boas-v${VERSION}
   \`\`\`

2. Add to PATH:
   \`\`\`bash
   export PATH="\$PWD/bin:\$PATH"
   export LD_LIBRARY_PATH="\$PWD/lib:\$LD_LIBRARY_PATH"
   \`\`\`

## First Program

1. Create a file \`test.bs\`:
   \`\`\`python
   import tensor
   
   def main():
       A = tensor.random(512, 512)
       B = tensor.random(512, 512)
       C = tensor.matmul(A, B)
       print("Success!")
   
   if __name__ == "__main__":
       main()
   \`\`\`

2. Compile and run:
   \`\`\`bash
   boas compile test.bs -o test
   ./test
   \`\`\`

## Examples

See the \`examples/\` directory for more examples:
- \`hello_world.bs\` - Basic example
- \`matrix_ops.bs\` - Matrix operations
- \`npu_optimized.bs\` - NPU optimizations
EOF

# Create version file
echo "${VERSION}" > ${RELEASE_DIR}/VERSION

# Create install script
cat > ${RELEASE_DIR}/install.sh << 'EOF'
#!/bin/bash
# BOAS Installation Script

INSTALL_DIR="/usr/local/boas"

echo "Installing BOAS to ${INSTALL_DIR}..."

# Check permissions
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Create installation directory
mkdir -p ${INSTALL_DIR}

# Copy files
cp -r * ${INSTALL_DIR}/

# Create symbolic links
ln -sf ${INSTALL_DIR}/bin/boas /usr/local/bin/boas

# Set up environment
echo "export BOAS_HOME=${INSTALL_DIR}" >> /etc/profile.d/boas.sh
echo "export PATH=\${BOAS_HOME}/bin:\$PATH" >> /etc/profile.d/boas.sh
echo "export LD_LIBRARY_PATH=\${BOAS_HOME}/lib:\$LD_LIBRARY_PATH" >> /etc/profile.d/boas.sh

echo "Installation complete!"
echo "Please run: source /etc/profile.d/boas.sh"
EOF

chmod +x ${RELEASE_DIR}/install.sh

# Create uninstall script
cat > ${RELEASE_DIR}/uninstall.sh << 'EOF'
#!/bin/bash
# BOAS Uninstallation Script

INSTALL_DIR="/usr/local/boas"

echo "Uninstalling BOAS..."

if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Remove installation
rm -rf ${INSTALL_DIR}
rm -f /usr/local/bin/boas
rm -f /etc/profile.d/boas.sh

echo "Uninstallation complete!"
EOF

chmod +x ${RELEASE_DIR}/uninstall.sh

# Create archive
echo "Creating release archive..."
tar -czf ${RELEASE_DIR}.tar.gz ${RELEASE_DIR}

# Calculate checksums
echo "Calculating checksums..."
sha256sum ${RELEASE_DIR}.tar.gz > ${RELEASE_DIR}.tar.gz.sha256

# Print summary
echo ""
echo "======================================"
echo "Release package created successfully!"
echo "======================================"
echo "Package: ${RELEASE_DIR}.tar.gz"
echo "Size: $(du -h ${RELEASE_DIR}.tar.gz | cut -f1)"
echo "SHA256: $(cat ${RELEASE_DIR}.tar.gz.sha256)"
echo ""
echo "Contents:"
ls -la ${RELEASE_DIR}/
echo ""
echo "To test the release:"
echo "  tar -xzf ${RELEASE_DIR}.tar.gz"
echo "  cd ${RELEASE_DIR}"
echo "  ./install.sh"
echo ""
echo "Release ready for distribution!"