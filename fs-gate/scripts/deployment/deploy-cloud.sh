#!/bin/bash

# Agricultural AI MCP Server - Cloud Deployment Script
# Supports DigitalOcean, AWS EC2, GCP Compute Engine

set -e

echo "🚀 Agricultural AI MCP Server - Cloud Deployment"
echo "================================================"

# Configuration
DOCKER_MCP_VERSION="latest"
SERVER_NAME="agricultural-ai-mcp"
DOMAIN=${DOMAIN:-"your-domain.com"}
EMAIL=${EMAIL:-"your-email@example.com"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Check required environment variables
check_env_vars() {
    print_status "Checking environment variables..."
    
    if [[ -z "$DATAGOVIN_API_KEY" ]]; then
        print_error "DATAGOVIN_API_KEY environment variable is required"
        exit 1
    fi
    
    if [[ -z "$EXA_API_KEY" ]]; then
        print_error "EXA_API_KEY environment variable is required"
        exit 1
    fi
    
    print_success "Environment variables validated"
}

# Install Docker and Docker Compose if not present
install_docker() {
    if ! command -v docker &> /dev/null; then
        print_status "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
        print_success "Docker installed"
    else
        print_success "Docker already installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_status "Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
        print_success "Docker Compose installed"
    else
        print_success "Docker Compose already installed"
    fi
}

# Install Docker MCP CLI
install_docker_mcp() {
    print_status "Installing Docker MCP CLI..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker must be installed first"
        exit 1
    fi
    
    # Check if docker mcp is available
    if ! docker mcp --help &> /dev/null; then
        print_status "Docker MCP CLI not found. Installing..."
        
        # Create plugins directory
        mkdir -p "$HOME/.docker/cli-plugins/"
        
        # Download and install docker-mcp plugin
        # Note: In production, you'd build this from source or download from releases
        print_warning "Docker MCP CLI installation requires manual setup"
        print_warning "Please follow: https://github.com/docker/mcp-gateway#installation"
    else
        print_success "Docker MCP CLI already available"
    fi
}

# Setup MCP Gateway configuration
setup_mcp_gateway() {
    print_status "Setting up MCP Gateway configuration..."
    
    # Enable configured catalogs feature
    docker mcp feature enable configured-catalogs || print_warning "Could not enable configured-catalogs feature"
    
    # Create production catalog
    docker mcp catalog create agricultural-ai-production || print_warning "Catalog may already exist"
    
    # Add our server to the catalog
    docker mcp catalog add agricultural-ai-production agricultural-ai-unified ./agricultural-ai-catalog.yaml --force || print_warning "Could not add server to catalog"
    
    print_success "MCP Gateway configured"
}

# Create necessary directories and files
setup_directories() {
    print_status "Setting up directories..."
    
    mkdir -p logs
    mkdir -p secrets
    mkdir -p nginx/ssl
    mkdir -p monitoring
    
    # Create nginx configuration
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream mcp_gateway {
        server mcp-gateway:8811;
    }
    
    upstream ai_server {
        server agricultural-ai-server:10000;
    }
    
    server {
        listen 80;
        server_name _;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }
    
    server {
        listen 443 ssl;
        server_name _;
        
        # SSL configuration (add your certificates)
        # ssl_certificate /etc/nginx/ssl/cert.pem;
        # ssl_certificate_key /etc/nginx/ssl/key.pem;
        
        # MCP Gateway
        location /mcp/ {
            proxy_pass http://mcp_gateway/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Direct API access
        location /api/ {
            proxy_pass http://ai_server/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://ai_server/health;
        }
    }
}
EOF
    
    print_success "Directories and configuration files created"
}

# Build and deploy
deploy() {
    print_status "Building and deploying services..."
    
    # Build the application
    print_status "Building Agricultural AI MCP Server..."
    docker-compose -f docker-compose.cloud.yml build
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.cloud.yml up -d
    
    # Wait for services to be healthy
    print_status "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if docker-compose -f docker-compose.cloud.yml ps | grep -q "Up (healthy)"; then
        print_success "Services are running and healthy"
    else
        print_warning "Some services may not be fully healthy yet"
    fi
    
    print_success "Deployment completed!"
}

# Show deployment information
show_info() {
    echo ""
    echo "🎉 Agricultural AI MCP Server Deployed Successfully!"
    echo "=================================================="
    echo ""
    echo "🌐 Services:"
    echo "   • MCP Gateway: http://$(curl -s ifconfig.me):8811"
    echo "   • Direct API: http://$(curl -s ifconfig.me):10000"
    echo "   • Health Check: http://$(curl -s ifconfig.me):10000/health"
    echo ""
    echo "🔧 Management Commands:"
    echo "   • View logs: docker-compose -f docker-compose.cloud.yml logs -f"
    echo "   • Stop services: docker-compose -f docker-compose.cloud.yml down"
    echo "   • Restart: docker-compose -f docker-compose.cloud.yml restart"
    echo ""
    echo "🧪 Test MCP Gateway:"
    echo "   curl -X POST http://$(curl -s ifconfig.me):8811/tools/call \\"
    echo "     -H 'Content-Type: application/json' \\"
    echo "     -d '{\"name\": \"crop-price\", \"arguments\": {\"state\": \"Punjab\", \"commodity\": \"Wheat\"}}'"
    echo ""
    echo "🎯 Ready for hackathon demonstration!"
}

# Main deployment flow
main() {
    print_status "Starting deployment process..."
    
    check_env_vars
    install_docker
    install_docker_mcp
    setup_directories
    setup_mcp_gateway
    deploy
    show_info
    
    print_success "Deployment completed successfully! 🚀"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        print_status "Stopping services..."
        docker-compose -f docker-compose.cloud.yml down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting services..."
        docker-compose -f docker-compose.cloud.yml restart
        print_success "Services restarted"
        ;;
    "logs")
        docker-compose -f docker-compose.cloud.yml logs -f
        ;;
    "status")
        docker-compose -f docker-compose.cloud.yml ps
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status}"
        exit 1
        ;;
esac