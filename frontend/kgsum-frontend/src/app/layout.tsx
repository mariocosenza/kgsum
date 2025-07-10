import type {Metadata, Viewport} from "next";
import {Geist, Geist_Mono} from "next/font/google";
import "./globals.css";
import {NavBar} from "@/components/navbar";
import {Footer} from "@/components/footer";
import {ThemeProvider} from "next-themes";
import {ClerkProvider} from "@clerk/nextjs";
import React from "react";
import {itIT} from "@clerk/localizations";

export const viewport: Viewport = {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
    userScalable: false,
    // Also supported but less commonly used
    // interactiveWidget: 'resizes-visual',
}

const geistSans = Geist({
    variable: "--font-geist-sans",
    subsets: ["latin"],
});

const geistMono = Geist_Mono({
    variable: "--font-geist-mono",
    subsets: ["latin"],
});

export const metadata: Metadata = {
    title: "HomePage KgSum",
    description: "Progetto di tesi sulla summarization di Knowledge Graph"
};

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <ClerkProvider localization={itIT}
                       appearance={{
                           cssLayerName: 'clerk',
                       }}>
            <html lang="it" suppressHydrationWarning>
            <body
                className={`
          ${geistSans.variable} ${geistMono.variable} antialiased
          min-h-screen flex flex-col
        `}
            >
            <ThemeProvider
                attribute="class"
                defaultTheme="system"
                enableSystem
                disableTransitionOnChange
            >
                <NavBar/>
                {/* This div takes up all available space between NavBar and Footer */}
                <div className="flex-1 flex flex-col">
                    {children}
                </div>
                <Footer/>
            </ThemeProvider>
            </body>
            </html>
        </ClerkProvider>
    );
}