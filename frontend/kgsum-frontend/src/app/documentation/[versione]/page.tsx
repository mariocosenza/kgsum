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
        const markdownDocsDirectory = path.join(process.cwd(), 'src', 'app', 'documentation', '[versione]');
        const filePath = path.join(markdownDocsDirectory, `${resolvedParams.versione}.md`);

        // Add a console.log to debug the exact path being attempted
        console.log(`Attempting to read file: ${filePath}`);

        content = await fs.readFile(filePath, 'utf8');
    } catch {
        redirect("/documentation/latest");
    }

    return (
        <DocumentTemplate content={content}/>
    );
}