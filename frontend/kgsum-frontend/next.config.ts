import type { NextConfig } from "next";

const nextConfig : NextConfig = {
  output: 'standalone',
  experimental: {
    serverActions: {
      bodySizeLimit: '500mb',
    },
  },
  // Environment configuration
  env: {
    PORT: process.env.PORT || '80',
    BUILD_DATE: '2025-06-28 13:42:21',
    BUILD_USER: 'mariocosenza'
  },
  // Performance optimizations
  poweredByHeader: false,
  compress: true,
  images: {
    formats: ['image/webp', 'image/avif'],
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
    ],
  },
}

module.exports = nextConfig