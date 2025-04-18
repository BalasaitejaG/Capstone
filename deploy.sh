#!/bin/bash

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running. Please start Docker and try again."
        exit 1
    fi
}

# Function to check if required ports are available
check_ports() {
    if lsof -Pi :32181 -sTCP:LISTEN -t >/dev/null ; then
        echo "Error: Port 32181 is already in use"
        exit 1
    fi
    if lsof -Pi :39092 -sTCP:LISTEN -t >/dev/null ; then
        echo "Error: Port 39092 is already in use"
        exit 1
    fi
}

# Main deployment function
deploy() {
    echo "Starting deployment..."
    
    # Check Docker status
    check_docker
    
    # Check if ports are available
    check_ports
    
    # Create network if it doesn't exist
    if ! docker network ls | grep -q kafka_network; then
        echo "Creating kafka network..."
        docker network create kafka_network
    fi
    
    # Start the services using docker-compose
    echo "Starting services with docker-compose..."
    docker-compose up -d
    
    # Wait for services to be healthy
    echo "Waiting for services to be ready..."
    sleep 10
    
    # Verify services are running
    if docker-compose ps | grep -q "Up"; then
        echo "Deployment successful! Services are running."
        echo "Zookeeper is available at localhost:32181"
        echo "Kafka is available at localhost:39092"
    else
        echo "Error: Services failed to start properly"
        docker-compose logs
        exit 1
    fi
}

# Cleanup function
cleanup() {
    echo "Cleaning up resources..."
    docker-compose down
    docker network rm kafka_network 2>/dev/null || true
}

# Command line argument handling
case "$1" in
    "up")
        deploy
        ;;
    "down")
        cleanup
        ;;
    *)
        echo "Usage: $0 {up|down}"
        echo "  up   - Deploy the infrastructure"
        echo "  down - Clean up all resources"
        exit 1
        ;;
esac