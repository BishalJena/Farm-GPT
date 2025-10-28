#!/bin/bash

# Docker MCP Gateway Setup Script for Agricultural AI
# This script sets up Docker MCP Gateway locally for development and testing

set -e

echo "🤖 Docker MCP Gateway Setup for Agricultural AI"
echo "=============================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    print_success "Docker is running"
}

# Install Docker MCP Gateway (if not already installed)
install_mcp_gateway() {
    print_status "Checking Docker MCP Gateway installation..."
    
    if docker mcp --help > /dev/null 2>&1; then
        print_success "Docker MCP Gateway is already installed"
        return 0
    fi
    
    print_warning "Docker MCP Gateway not found. Installation required."
    print_status "Please install Docker MCP Gateway manually:"
    echo ""
    echo "1. Clone the repository:"
    echo "   git clone https://github.com/docker/mcp-gateway.git"
    echo "   cd mcp-gateway"
    echo ""
    echo "2. Build and install:"
    echo "   mkdir -p \"\$HOME/.docker/cli-plugins/\""
    echo "   make docker-mcp"
    echo ""
    echo "3. Verify installation:"
    echo "   docker mcp --help"
    echo ""
    read -p "Press Enter after installing Docker MCP Gateway..."
    
    if ! docker mcp --help > /dev/null 2>&1; then
        print_error "Docker MCP Gateway installation failed or not found"
        exit 1
    fi
    
    print_success "Docker MCP Gateway is now available"
}

# Setup MCP Gateway features and catalogs
setup_gateway() {
    print_status "Setting up MCP Gateway configuration..."
    
    # Enable configured catalogs feature
    print_status "Enabling configured-catalogs feature..."
    if docker mcp feature enable configured-catalogs; then
        print_success "Configured catalogs feature enabled"
    else
        print_warning "Could not enable configured-catalogs feature (may already be enabled)"
    fi
    
    # Create agricultural AI catalog
    print_status "Creating agricultural AI catalog..."
    if docker mcp catalog create agricultural-ai-dev; then
        print_success "Agricultural AI catalog created"
    else
        print_warning "Catalog may already exist"
    fi
    
    # Add our server to the catalog
    print_status "Adding agricultural AI server to catalog..."
    if docker mcp catalog add agricultural-ai-dev agricultural-ai-unified ./agricultural-ai-catalog.yaml --force; then
        print_success "Server added to catalog"
    else
        print_warning "Could not add server to catalog"
    fi
    
    # List catalogs to verify
    print_status "Current catalogs:"
    docker mcp catalog ls || print_warning "Could not list catalogs"
    
    print_success "MCP Gateway configuration completed"
}

# Build the Docker image
build_image() {
    print_status "Building Agricultural AI MCP Server Docker image..."
    
    if docker build -t agricultural-ai-mcp:latest .; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Test the setup
test_setup() {
    print_status "Testing the setup..."
    
    # Start the services
    print_status "Starting services with Docker Compose..."
    if docker-compose up -d; then
        print_success "Services started"
    else
        print_error "Failed to start services"
        exit 1
    fi
    
    # Wait for services to be ready
    print_status "Waiting for services to be ready..."
    sleep 15
    
    # Test direct API
    print_status "Testing direct API..."
    if curl -f http://localhost:10000/health > /dev/null 2>&1; then
        print_success "Direct API is responding"
    else
        print_warning "Direct API may not be ready yet"
    fi
    
    # Test MCP Gateway
    print_status "Testing MCP Gateway..."
    if curl -f http://localhost:8811 > /dev/null 2>&1; then
        print_success "MCP Gateway is responding"
    else
        print_warning "MCP Gateway may not be ready yet"
    fi
    
    # Test tool calling through gateway
    print_status "Testing tool calling through MCP Gateway..."
    
    # Show available tools
    print_status "Available tools through gateway:"
    docker mcp tools --gateway-arg="--servers=agricultural-ai-unified" list || print_warning "Could not list tools"
    
    print_success "Setup testing completed"
}

# Show usage information
show_usage() {
    echo ""
    echo "🎉 Agricultural AI MCP Gateway Setup Complete!"
    echo "============================================="
    echo ""
    echo "🌐 Services Running:"
    echo "   • MCP Gateway: http://localhost:8811"
    echo "   • Direct API: http://localhost:10000"
    echo "   • Health Check: http://localhost:10000/health"
    echo ""
    echo "🧪 Test Commands:"
    echo ""
    echo "1. List available tools:"
    echo "   docker mcp tools list"
    echo ""
    echo "2. Test crop price tool:"
    echo "   docker mcp tools call crop-price state=Punjab commodity=Wheat"
    echo ""
    echo "3. Test search tool:"
    echo "   docker mcp tools call search query=\"Indian agriculture news\""
    echo ""
    echo "4. Test via HTTP (direct API):"
    echo "   curl -X POST http://localhost:10000/tools/crop-price \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"state\": \"Punjab\", \"commodity\": \"Wheat\"}'"
    echo ""
    echo "5. Test via MCP Protocol:"
    echo "   curl -X POST http://localhost:10000/mcp \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"jsonrpc\": \"2.0\", \"id\": 1, \"method\": \"tools/list\"}'"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • View logs: docker-compose logs -f"
    echo "   • Stop services: docker-compose down"
    echo "   • Restart: docker-compose restart"
    echo ""
    echo "📚 MCP Gateway Commands:"
    echo "   • List catalogs: docker mcp catalog ls"
    echo "   • Show catalog: docker mcp catalog show agricultural-ai-dev"
    echo "   • Gateway help: docker mcp gateway --help"
    echo ""
    echo "🚀 Ready for development and hackathon demo!"
}

# Main setup flow
main() {
    print_status "Starting MCP Gateway setup..."
    
    check_docker
    install_mcp_gateway
    build_image
    setup_gateway
    test_setup
    show_usage
    
    print_success "Setup completed successfully! 🎉"
}

# Handle script arguments
case "${1:-setup}" in
    "setup")
        main
        ;;
    "test")
        test_setup
        ;;
    "clean")
        print_status "Cleaning up..."
        docker-compose down
        docker mcp catalog rm agricultural-ai-dev || print_warning "Could not remove catalog"
        print_success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 {setup|test|clean}"
        exit 1
        ;;
esac