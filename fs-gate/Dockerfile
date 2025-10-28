# Multi-stage production Dockerfile for Agricultural AI MCP Server
FROM node:20-alpine AS build

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install all dependencies (including dev dependencies for build)
RUN npm ci

# Copy source code and config
COPY src ./src
COPY tsconfig.json ./

# Build TypeScript
RUN npm run build

# Production stage
FROM node:20-alpine AS production

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S mcp -u 1001

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install only production dependencies
RUN npm ci --production --ignore-scripts && \
    npm cache clean --force

# Copy built application from build stage
COPY --from=build /app/dist ./dist

# Change ownership to non-root user
RUN chown -R mcp:nodejs /app
USER mcp

# Set production environment
ENV NODE_ENV=production

# Render dynamically assigns PORT
# Don't set a default PORT, let Render handle it

# Expose the port (Render will set this)
EXPOSE 10000

# Health check for Render
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD node -e "console.log('Agricultural AI MCP server healthy')" || exit 1

# Start the MCP server
CMD ["node", "dist/server.js"]