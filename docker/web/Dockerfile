# Use the latest Node.js version
FROM node:latest AS base

# Install dependencies only when needed
FROM base AS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Copy package files from the correct path
COPY ../../frontend/kgsum-frontend/package.json ../../frontend/kgsum-frontend/package-lock.json* ./
# Install dependencies using npm only
RUN npm ci --only=production --ignore-scripts && npm cache clean --force

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
# Copy the entire frontend application
COPY ../../frontend/kgsum-frontend/ ./

# Set environment variables for build (Next.js 15)
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG CLERK_SECRET_KEY

# Validate that required environment variables are provided
RUN if [ -z "$NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" ]; then \
        echo "ERROR: NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY is required but not provided"; \
        exit 1; \
    fi

# Export as environment variables for Next.js build
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}
ENV CLERK_SECRET_KEY=${CLERK_SECRET_KEY}


# Verify environment variables are set (debug info)
RUN echo "=== Environment Variables Check ==="
RUN echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY: ${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}"
RUN echo "CLERK_SECRET_KEY: ${CLERK_SECRET_KEY}"
RUN echo "Build Date: 2025-06-28 13:53:17"
RUN echo "Build User: mariocosenza"
RUN echo "==================================="

# Install all dependencies for build
RUN npm ci --ignore-scripts
# Build the application using npm
RUN npm run build

# Production image for Next.js 15
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

# Create user mariocosenza for security
RUN groupadd --system --gid 1001 nodejs
RUN useradd --system --uid 1001 --gid nodejs mariocosenza

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown mariocosenza:nodejs .next

# Copy Next.js 15 standalone output
COPY --from=builder --chown=mariocosenza:nodejs /app/.next/standalone ./
COPY --from=builder --chown=mariocosenza:nodejs /app/.next/static ./.next/static

# Set runtime environment variables (only public ones)
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ENV NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY}


USER mariocosenza

# Expose port 80
EXPOSE 80
EXPOSE 433

# Set port to 80 for Next.js 15
ENV PORT=80
ENV HOSTNAME="0.0.0.0"

# Start the server
CMD ["node", "server.js"]