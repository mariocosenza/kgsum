import {redirect} from "next/navigation";
import {DocumentTemplate} from "@/components/document-template";
import fs from 'fs/promises';
import path from 'path';
import {VERSIONS} from "@/components/VERSIONS";

export async function generateStaticParams() {
    return VERSIONS.map((versione) => ({
        versione: versione,
    }));
}

function checkVersion(version: string): boolean {
    return VERSIONS.includes(version);
}

export default async function SelectedVersionPage({params}: { params: Promise<{ versione: string }> }) {
    // Await the params Promise
    const resolvedParams = await params;

    // 1. Basic version check and redirect
    if (resolvedParams.versione !== 'latest') {
        if (!checkVersion(resolvedParams.versione)) {
            redirect("/documentation/latest");
        }
    } else {
        resolvedParams.versione = VERSIONS[VERSIONS.length - 1];
    }

    let content: string;
    try {
        // More robust path resolution - try multiple possible locations
        const possiblePaths = [
            // Original path
            path.join(process.cwd(), 'src', 'app', 'documentation', '[versione]', `${resolvedParams.versione}.md`),
            // Alternative: files in public directory
            path.join(process.cwd(), 'public', 'docs', `${resolvedParams.versione}.md`),
            // Alternative: files in docs directory at root
            path.join(process.cwd(), 'docs', `${resolvedParams.versione}.md`),
            // Alternative: files in same directory without brackets
            path.join(process.cwd(), 'src', 'app', 'documentation', `${resolvedParams.versione}.md`),
        ];

        console.log(`Working directory: ${process.cwd()}`);
        console.log(`Looking for version: ${resolvedParams.versione}`);
        console.log(`Available versions: ${VERSIONS.join(', ')}`);

        let filePath: string | null = null;

        // Try each possible path
        for (const possiblePath of possiblePaths) {
            try {
                await fs.access(possiblePath);
                filePath = possiblePath;
                console.log(`Found file at: ${filePath}`);
                break;
            } catch {
                console.log(`File not found at: ${possiblePath}`);
            }
        }

        if (!filePath) {
            // If no file found, try to list available files for debugging
            for (const dir of [
                path.join(process.cwd(), 'src', 'app', 'documentation', '[versione]'),
                path.join(process.cwd(), 'public', 'docs'),
                path.join(process.cwd(), 'docs'),
                path.join(process.cwd(), 'src', 'app', 'documentation'),
            ]) {
                try {
                    const files = await fs.readdir(dir);
                    console.log(`Files in ${dir}:`, files);
                } catch {
                    console.log(`Directory does not exist: ${dir}`);
                }
            }

            console.error(`No markdown file found for version: ${resolvedParams.versione}`);
            redirect("/documentation/latest");
            return; // This won't execute, but TypeScript needs it
        }

        content = await fs.readFile(filePath, 'utf8');
        console.log(`Successfully read file: ${filePath}`);

    } catch (error) {
        console.error(`Error reading file:`, error);

        // Additional debugging: check if it's a permission issue or file doesn't exist
        if (error instanceof Error) {
            console.error(`Error details: ${error.message}`);
        }

        redirect("/documentation/latest");
        return; // This won't execute, but TypeScript needs it
    }

    return (
        <DocumentTemplate content={content}/>
    );
}