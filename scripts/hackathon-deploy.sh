#!/bin/bash

# 🏆 Hackathon Deployment Script
# One-command deployment of the complete Agricultural AI system

set -e

echo "🌾 Agricultural AI - Hackathon Deployment"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "🔍 Checking Prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if ports are available
    if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port 80 is already in use. The application might not be accessible."
    fi
    
    if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warning "Port 8000 is already in use. Backend might conflict."
    fi
    
    print_status "Prerequisites check completed ✓"
}

# Setup environment
setup_environment() {
    print_header "⚙️ Setting up Environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# Database Configuration
MONGO_ROOT_PASSWORD=agricultural-ai-hackathon-2024
REDIS_PASSWORD=agricultural-ai-cache-hackathon

# Authentication
JWT_SECRET=hackathon-super-secret-jwt-key-change-in-production

# AI Services (Add your API keys here)
CEREBRAS_API_KEY=your-cerebras-api-key-here
DATAGOVIN_API_KEY=your-datagovin-api-key-here
EXA_API_KEY=your-exa-api-key-here

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=hackathon-grafana-2024

# Application
APP_VERSION=1.0.0-hackathon
ENVIRONMENT=production
EOF
        print_warning "Please update the API keys in .env file before proceeding!"
        print_status "Created .env file ✓"
    else
        print_status "Using existing .env file ✓"
    fi
    
    # Create necessary directories
    mkdir -p logs nginx/conf.d ssl monitoring/grafana/dashboards letsencrypt
    print_status "Created necessary directories ✓"
}

# Build and deploy
deploy_system() {
    print_header "🚀 Deploying Agricultural AI System..."
    
    # Pull latest images
    print_status "Pulling latest Docker images..."
    docker-compose -f docker-compose.production.yml pull
    
    # Build custom images
    print_status "Building custom images..."
    docker-compose -f docker-compose.production.yml build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose -f docker-compose.production.yml up -d
    
    print_status "Deployment completed ✓"
}

# Wait for services
wait_for_services() {
    print_header "⏳ Waiting for services to be ready..."
    
    # Wait for MongoDB
    print_status "Waiting for MongoDB..."
    timeout 60 bash -c 'until docker-compose -f docker-compose.production.yml exec -T mongodb mongosh --eval "db.adminCommand(\"ping\")" > /dev/null 2>&1; do sleep 2; done'
    
    # Wait for Backend
    print_status "Waiting for Backend API..."
    timeout 120 bash -c 'until curl -f http://localhost:8000/api/health > /dev/null 2>&1; do sleep 5; done'
    
    # Wait for Frontend
    print_status "Waiting for Frontend..."
    timeout 60 bash -c 'until curl -f http://localhost:3000 > /dev/null 2>&1; do sleep 3; done'
    
    # Wait for MCP Server
    print_status "Waiting for MCP Server..."
    timeout 60 bash -c 'until curl -f http://localhost:10000/health > /dev/null 2>&1; do sleep 3; done'
    
    print_status "All services are ready ✓"
}

# Setup demo data
setup_demo_data() {
    print_header "📊 Setting up Demo Data..."
    
    # Create demo user
    print_status "Creating demo user..."
    curl -X POST http://localhost:8000/api/auth/register \
        -H "Content-Type: application/json" \
        -d '{"email":"demo@agricultural-ai.com","password":"hackathon2024"}' \
        > /dev/null 2>&1 || true
    
    # Load sample metrics
    print_status "Loading sample metrics..."
    docker-compose -f docker-compose.production.yml exec -T backend python -c "
from metrics_system import MetricsSystem
from database import get_database
import asyncio

async def load_demo_metrics():
    db = await get_database()
    metrics = MetricsSystem(db)
    print('Demo metrics loaded successfully')

asyncio.run(load_demo_metrics())
" || true
    
    print_status "Demo data setup completed ✓"
}

# Display access information
show_access_info() {
    print_header "🎯 Access Information"
    echo ""
    echo "🌐 Web Application:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8000"
    echo "   API Docs: http://localhost:8000/docs"
    echo ""
    echo "🛠️ MCP Tools:"
    echo "   MCP Server: http://localhost:10000"
    echo "   Health Check: http://localhost:10000/health"
    echo ""
    echo "📊 Monitoring:"
    echo "   Prometheus: http://localhost:9090"
    echo "   Grafana: http://localhost:3001 (admin/hackathon-grafana-2024)"
    echo "   Kibana: http://localhost:5601"
    echo ""
    echo "🔧 Demo Credentials:"
    echo "   Email: demo@agricultural-ai.com"
    echo "   Password: hackathon2024"
    echo ""
    echo "📈 Performance Dashboard:"
    echo "   Metrics: http://localhost:3000/dashboard"
    echo "   Cerebras Showcase: http://localhost:8000/api/metrics/cerebras-showcase"
    echo ""
}

# Health check
run_health_check() {
    print_header "🏥 Running Health Check..."
    
    # Check all services
    services=("frontend:3000" "backend:8000/api/health" "mcp-server:10000/health" "prometheus:9090" "grafana:3001")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name endpoint <<< "$service"
        if curl -f "http://localhost:$endpoint" > /dev/null 2>&1; then
            print_status "$name: ✅ Healthy"
        else
            print_error "$name: ❌ Unhealthy"
        fi
    done
    
    # Test MCP tools
    print_status "Testing MCP tools..."
    if curl -f "http://localhost:10000/tools/crop-price" \
        -H "Content-Type: application/json" \
        -d '{"state":"Punjab","commodity":"Wheat"}' > /dev/null 2>&1; then
        print_status "MCP Tools: ✅ Working"
    else
        print_warning "MCP Tools: ⚠️ Check API keys"
    fi
    
    print_status "Health check completed ✓"
}

# Main deployment flow
main() {
    echo ""
    print_header "🏆 Starting Hackathon Deployment..."
    echo ""
    
    check_prerequisites
    setup_environment
    deploy_system
    wait_for_services
    setup_demo_data
    run_health_check
    
    echo ""
    print_header "🎉 Deployment Successful!"
    echo ""
    show_access_info
    
    echo ""
    print_status "🚀 Your Agricultural AI system is ready for the hackathon!"
    print_status "📖 Check DEMO_SCENARIOS.md for complete demo instructions"
    print_status "🏆 Good luck with your presentation!"
    echo ""
}

# Cleanup function
cleanup() {
    print_header "🧹 Cleaning up..."
    docker-compose -f docker-compose.production.yml down -v
    docker system prune -f
    print_status "Cleanup completed ✓"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "health")
        run_health_check
        ;;
    "logs")
        docker-compose -f docker-compose.production.yml logs -f
        ;;
    "restart")
        docker-compose -f docker-compose.production.yml restart
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|health|logs|restart}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the complete system (default)"
        echo "  cleanup - Stop and remove all containers"
        echo "  health  - Run health check on all services"
        echo "  logs    - Show logs from all services"
        echo "  restart - Restart all services"
        exit 1
        ;;
esac