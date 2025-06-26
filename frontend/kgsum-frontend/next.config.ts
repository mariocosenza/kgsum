import type { NextConfig } from "next";

/** @type {import('next').NextConfig} */
const nextConfig : NextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '500mb',
    },
  },
  // For API routes (if you're using them)
  api: {
    bodyParser: {
      sizeLimit: '500mb',
    },
    responseLimit: false,
  },
}

module.exports = nextConfig