// src/app/documentazione/[versione]/page.tsx

import {redirect} from "next/navigation";
import {DocumentTemplate} from "@/components/document-template";
import fs from 'fs/promises';
import path from 'path';
import {VERSIONS} from "@/components/VERSIONS";


export async function generateStaticParams() {
    return VERSIONS.map((versione) => ({
        versione: versione, // The key 'versione' must match your dynamic segment name '[versione]'
    }));
}

function checkVersion(version: string): boolean {
    return VERSIONS.includes(version);
}

export default async function SelectedVersionPage({ params }: { params: { versione: string } }) {
    // 1. Basic version check and redirect
    if (params.versione !== 'latest') {
        if (!checkVersion(params.versione)) {
            redirect("/documentazione/latest");
        }
    }  else {
        params.versione = VERSIONS[VERSIONS.length - 1]
    }

    let content: string;
    try {
        // --- CRITICAL CORRECTION HERE ---
        // process.cwd() is your project root.
        // Assuming your markdown files are in `src/docs/`
        const markdownDocsDirectory = path.join(process.cwd(), 'src', 'app', 'documentazione', '[versione]');
        const filePath = path.join(markdownDocsDirectory, `${params.versione}.md`);

        // Add a console.log to debug the exact path being attempted
        console.log(`Attempting to read file: ${filePath}`);

        content = await fs.readFile(filePath, 'utf8');
    } catch {
        redirect("/documentazione/latest");
    }

    return (
        <DocumentTemplate content={content} />
    );
}