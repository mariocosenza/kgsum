"use client";
import React, {useActionState, useMemo, useRef, useState} from "react";
import {createPost} from "@/app/classify/action";
import {Tabs, TabsContent, TabsList, TabsTrigger,} from "@/components/ui/tabs";
import {Label} from "@/components/ui/label";
import {Input} from "@/components/ui/input";
import {Card, CardContent} from "@/components/ui/card";
import {Separator} from "@/components/ui/separator";
import {Checkbox} from "@/components/ui/checkbox";
import {Button} from "@/components/ui/button";
import {Textarea} from "@/components/ui/textarea";
import Link from "next/link";

function FileIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>
            <path d="M14 2v4a2 2 0 0 0 2 2h4"/>
        </svg>
    );
}

function CopyIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <rect width="14" height="14" x="8" y="8" rx="2" ry="2"/>
            <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/>
        </svg>
    );
}

function CheckIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="M20 6 9 17l-5-5"/>
        </svg>
    );
}

function AlertTriangleIcon(props: React.SVGProps<SVGSVGElement>) {
    return (
        <svg
            {...props}
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
        >
            <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>
            <path d="M12 9v4"/>
            <path d="m12 17 .01 0"/>
        </svg>
    );
}

// Infinite Progress Component with proper animation and dark mode support
function InfiniteProgress({isVisible}: { isVisible: boolean }) {
    if (!isVisible) return null;

    return (
        <div className="w-full mt-4">
            <div className="flex items-center space-x-3 mb-2">
                <div className="flex-1 relative h-2 bg-muted rounded-full overflow-hidden">
                    <div className="absolute inset-0 h-full bg-primary rounded-full animate-progress-infinite"></div>
                </div>
                <span className="text-sm text-muted-foreground font-medium">Elaborando...</span>
            </div>
            <style jsx>{`
        @keyframes progress-infinite {
          0% {
            transform: translateX(-100%);
            width: 30%;
          }
          50% {
            width: 50%;
          }
          100% {
            transform: translateX(400%);
            width: 30%;
          }
        }
        .animate-progress-infinite {
          animation: progress-infinite 2s ease-in-out infinite;
        }
      `}</style>
        </div>
    );
}

// File size formatter utility
function formatFileSize(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
}

export const Form = () => {
    // Use useActionState properly with the server action
    const [state, formAction, isPending] = useActionState(
        createPost,
        {message: "", data: undefined}
    );

    const [tab, setTab] = useState<"SPARQL" | "DUMP">("SPARQL");
    const [sparqlUrl, setSparqlUrl] = useState("");
    const [hasFile, setHasFile] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [privacyConsent, setPrivacyConsent] = useState(false);
    const [isCopied, setIsCopied] = useState(false);
    const [fileSizeWarning, setFileSizeWarning] = useState("");
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Constants
    const MAX_FILE_SIZE = 524288000; // 500MB in bytes
    const LARGE_FILE_THRESHOLD = 104857600; // 100MB in bytes

    // Calculate if form is valid based on current tab and inputs
    const isFormValid = useMemo(() => {
        if (!privacyConsent) return false;

        if (tab === "SPARQL") {
            return sparqlUrl.trim() !== "";
        }

        if (tab === "DUMP") {
            return hasFile && selectedFile && selectedFile.size <= MAX_FILE_SIZE;
        }

        return false;
    }, [tab, sparqlUrl, hasFile, selectedFile, privacyConsent]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];

        if (file) {
            // Reset warnings
            setFileSizeWarning("");

            // Check file size (500MB limit)
            if (file.size > MAX_FILE_SIZE) {
                const sizeMB = Math.round(file.size / (1024 * 1024));
                setFileSizeWarning(`Error: The file (${sizeMB}MB) is over the 500MB size threshold. Upload a smaller file.`);
                e.target.value = ''; // Clear the input
                setHasFile(false);
                setSelectedFile(null);
                return;
            }

            // Warn for large files
            if (file.size > LARGE_FILE_THRESHOLD) {
                const sizeMB = Math.round(file.size / (1024 * 1024));
                setFileSizeWarning(`Large file (${sizeMB}MB) - the processing could take a few minutes.`);
            }

            // Validate file extension
            const allowedExtensions = ['.rdf', '.ttl', '.nq', '.nt', '.xml', '.json'];
            const fileName = file.name.toLowerCase();
            const fileExtension = '.' + fileName.split('.').pop();

            if (!allowedExtensions.includes(fileExtension)) {
                setFileSizeWarning(`Error: File format "${fileExtension}" not support. Accepted format: ${allowedExtensions.join(', ')}`);
                e.target.value = ''; // Clear the input
                setHasFile(false);
                setSelectedFile(null);
                return;
            }

            setHasFile(true);
            setSelectedFile(file);
        } else {
            setHasFile(false);
            setSelectedFile(null);
            setFileSizeWarning("");
        }
    };

    // Format JSON response for display
    const formatResponse = (response: Record<string, unknown> | undefined): string => {
        if (!response) return "";

        try {
            return JSON.stringify(response, null, 2);
        } catch {
            return String(response);
        }
    };

    const displayResponse = state?.data ? formatResponse(state.data) : "";

    // Copy to clipboard functionality
    const handleCopy = async () => {
        if (!displayResponse) return;

        try {
            await navigator.clipboard.writeText(displayResponse);
            setIsCopied(true);
            setTimeout(() => setIsCopied(false), 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    };

    // Reset form when switching tabs
    const handleTabChange = (newTab: string) => {
        setTab(newTab as "SPARQL" | "DUMP");
        // Reset form state when switching tabs
        setSparqlUrl("");
        setHasFile(false);
        setSelectedFile(null);
        setFileSizeWarning("");
        if (fileInputRef.current) {
            fileInputRef.current.value = "";
        }
    };

    return (
        <div className="min-h-screen bg-background py-8">
            <div className="container mx-auto px-4 max-w-[120rem]">
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                    {/* Form Section */}
                    <div className="bg-card border rounded-xl shadow-xl p-8 min-h-[50rem]">
                        <h2 className="text-xl font-bold text-foreground mb-6 text-center">
                            Classify RDF data
                        </h2>

                        <form
                            action={formAction}
                            className="space-y-6"
                            autoComplete="off"
                        >
                            {/* Hidden input to specify the mode */}
                            <input type="hidden" name="mode" value={tab}/>

                            <div className="space-y-5">
                                <Tabs
                                    defaultValue="SPARQL"
                                    value={tab}
                                    onValueChange={handleTabChange}
                                    className="w-full"
                                >
                                    <div className="flex justify-center mb-6">
                                        <TabsList className="grid w-full max-w-md grid-cols-2 h-10">
                                            <TabsTrigger value="SPARQL" id="tab-sparql" className="text-center text-sm">
                                                SPARQL
                                            </TabsTrigger>
                                            <TabsTrigger value="DUMP" id="tab-dump" className="text-center text-sm">
                                                DUMP
                                            </TabsTrigger>
                                        </TabsList>
                                    </div>

                                    {/* Fixed height container for consistent tab content height */}
                                    <div className="min-h-[220px]">
                                        <TabsContent value="SPARQL" className="space-y-4 mt-0">
                                            <div className="space-y-2">
                                                <Label htmlFor="sparql-url" className="text-sm font-medium">
                                                    SPARQL Endpoint
                                                </Label>
                                                <Input
                                                    type="url"
                                                    id="sparql-url"
                                                    name="sparqlUrl"
                                                    placeholder="https://example.com/sparql"
                                                    autoComplete="off"
                                                    value={sparqlUrl}
                                                    onChange={(e) => setSparqlUrl(e.target.value)}
                                                    required={tab === "SPARQL"}
                                                    disabled={tab !== "SPARQL" || isPending}
                                                    className="w-full h-10 text-sm"
                                                />
                                                <p className="text-xs text-muted-foreground">
                                                    Input the full URL of the SPARQL endpoint to analyze
                                                </p>
                                            </div>
                                        </TabsContent>

                                        <TabsContent value="DUMP" className="space-y-4 mt-0">
                                            <Card
                                                className="border-2 border-dashed border-muted hover:border-muted-foreground/50 transition-colors min-h-[180px]">
                                                <CardContent
                                                    className="p-6 space-y-4 flex flex-col justify-center min-h-[180px]">
                                                    <div className="text-center space-y-3">
                                                        <div className="flex justify-center">
                                                            <FileIcon className="w-12 h-12 text-muted-foreground"/>
                                                        </div>
                                                        <div className="space-y-1">
                                                            <p className="text-sm font-medium text-foreground">
                                                                Drag an RDF file or click here to select
                                                            </p>
                                                            <p className="text-xs text-muted-foreground">
                                                                Supported format: RDF, TTL, NQ, NT, XML, JSON
                                                            </p>
                                                            <p className="text-xs text-muted-foreground font-medium dark:text-blue-400">
                                                                Max size: 500MB
                                                            </p>
                                                        </div>
                                                    </div>

                                                    <div className="space-y-2">
                                                        <Label htmlFor="file-upload" className="text-sm font-medium">
                                                            Select RDF File
                                                        </Label>
                                                        <Input
                                                            id="file-upload"
                                                            name="file"
                                                            type="file"
                                                            accept=".rdf,.ttl,.nq,.nt,.xml,.json"
                                                            ref={fileInputRef}
                                                            onChange={handleFileChange}
                                                            required={tab === "DUMP"}
                                                            disabled={tab !== "DUMP" || isPending}
                                                            className="w-full h-10 text-sm"
                                                        />
                                                        <p className="text-xs text-muted-foreground">
                                                            Estensioni: .rdf, .ttl, .nq, .nt, .xml, .json (max 500MB)
                                                        </p>

                                                        {/* File info display */}
                                                        {selectedFile && (
                                                            <div className="mt-3 p-3 bg-muted/30 rounded-lg border">
                                                                <div className="flex items-center justify-between">
                                                                    <div className="flex-1 min-w-0">
                                                                        <p className="text-sm font-medium text-foreground truncate">
                                                                            {selectedFile.name}
                                                                        </p>
                                                                        <p className="text-xs text-muted-foreground">
                                                                            {formatFileSize(selectedFile.size)} • {selectedFile.type || 'Tipo sconosciuto'}
                                                                        </p>
                                                                    </div>
                                                                    <CheckIcon
                                                                        className="w-4 h-4 text-green-600 dark:text-green-400 ml-2 flex-shrink-0"/>
                                                                </div>
                                                            </div>
                                                        )}
                                                    </div>
                                                </CardContent>
                                            </Card>

                                            {/* File size warning */}
                                            {fileSizeWarning && (
                                                <div
                                                    className={`mt-4 p-3 rounded-lg border flex items-start space-x-2 ${
                                                        fileSizeWarning.includes('supera il limite') || fileSizeWarning.includes('non supportato')
                                                            ? 'border-destructive/50 bg-destructive/10'
                                                            : 'border-orange-500/50 bg-orange-500/10'
                                                    }`}>
                                                    <AlertTriangleIcon className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                                                        fileSizeWarning.includes('supera il limite') || fileSizeWarning.includes('non supportato')
                                                            ? 'text-destructive'
                                                            : 'text-orange-600 dark:text-orange-400'
                                                    }`}/>
                                                    <p className={`text-sm ${
                                                        fileSizeWarning.includes('supera il limite') || fileSizeWarning.includes('non supportato')
                                                            ? 'text-destructive'
                                                            : 'text-orange-700 dark:text-orange-300'
                                                    }`}>
                                                        {fileSizeWarning}
                                                    </p>
                                                </div>
                                            )}
                                        </TabsContent>
                                    </div>
                                </Tabs>
                            </div>

                            <Separator className="my-6"/>

                            <div className="space-y-4">
                                {/* Save Profile Checkbox - First */}
                                <div className="flex items-start space-x-3 p-4 bg-muted/30 rounded-lg border">
                                    <Checkbox
                                        id="save-profile"
                                        name="saveProfile"
                                        disabled={isPending}
                                        className="mt-1"
                                    />
                                    <div className="flex-1">
                                        <Label htmlFor="save-profile" className="text-sm font-medium cursor-pointer">
                                            Save profile
                                        </Label>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Make the generated profile public
                                        </p>
                                    </div>
                                </div>

                                {/* Privacy Consent Checkbox - Second */}
                                <div className="flex items-start space-x-3 p-4 bg-muted/50 rounded-lg border">
                                    <Checkbox
                                        id="privacy-consent"
                                        name="privacyConsent"
                                        checked={privacyConsent}
                                        onCheckedChange={(checked) => setPrivacyConsent(!!checked)}
                                        required
                                        disabled={isPending}
                                        className="mt-1"
                                    />
                                    <div className="flex-1">
                                        <Label htmlFor="privacy-consent" className="text-sm font-medium cursor-pointer">
                                            Accept<Link href="/privacy" className="underline ml-0">terms and
                                            condition</Link>*
                                        </Label>
                                        <p className="text-xs text-muted-foreground mt-1">
                                            Obligatory to proceed
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <Separator className="my-6"/>

                            <div className="flex justify-center pt-2">
                                <Button
                                    disabled={isPending || !isFormValid}
                                    id="submit"
                                    type="submit"
                                    aria-disabled={isPending || !isFormValid}
                                    className="w-full max-w-lg py-3 text-base font-semibold h-12"
                                    size="lg"
                                >
                                    {isPending ? (
                                        <div className="flex items-center space-x-2">
                                            <div
                                                className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin"></div>
                                            <span>
                        {selectedFile && selectedFile.size > LARGE_FILE_THRESHOLD
                            ? 'Computing large file..'
                            : 'Computing...'
                        }
                      </span>
                                        </div>
                                    ) : (
                                        "Classify"
                                    )}
                                </Button>
                            </div>

                            {/* Display server validation errors */}
                            {state?.message && state.message.includes("Errore") && (
                                <div className="mt-6 p-4 rounded-lg border border-destructive/50 bg-destructive/10">
                                    <div className="flex items-start space-x-2">
                                        <AlertTriangleIcon className="w-5 h-5 text-destructive mt-0.5 flex-shrink-0"/>
                                        <p className="text-destructive text-sm">{state.message}</p>
                                    </div>
                                </div>
                            )}
                        </form>
                    </div>

                    {/* Response Section */}
                    <div className="bg-card border rounded-xl shadow-xl p-8 min-h-[50rem]">
                        <h2 className="text-xl font-bold text-foreground mb-6 text-center">
                            Result JSON
                        </h2>

                        <div className="space-y-4">
                            <div className="space-y-3">
                                <div className="flex items-center justify-between">
                                    <Label htmlFor="message" className="text-sm font-medium">
                                        System response
                                    </Label>
                                    {displayResponse && (
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={handleCopy}
                                            disabled={isPending}
                                            className="flex items-center space-x-2 h-8 text-xs"
                                        >
                                            {isCopied ? (
                                                <>
                                                    <CheckIcon className="w-3 h-3 text-green-600 dark:text-green-400"/>
                                                    <span className="text-green-600 dark:text-green-400">Copiato!</span>
                                                </>
                                            ) : (
                                                <>
                                                    <CopyIcon className="w-3 h-3"/>
                                                    <span>Copy</span>
                                                </>
                                            )}
                                        </Button>
                                    )}
                                </div>
                                <Textarea
                                    id="message"
                                    value={displayResponse}
                                    placeholder="The JSON response will appear here after processing..."
                                    readOnly
                                    className="w-full h-[35rem] resize-none bg-muted/30 font-mono text-xs leading-relaxed focus:ring-2 focus:ring-ring focus:ring-offset-2"
                                />

                                {/* Progress bar below the textarea */}
                                <InfiniteProgress isVisible={isPending}/>
                            </div>

                            {displayResponse && !isPending && (
                                <div className="mt-4 p-4 rounded-lg border border-green-500/50 bg-green-500/10">
                                    <div className="flex items-start space-x-2">
                                        <CheckIcon
                                            className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 flex-shrink-0"/>
                                        <div className="flex-1">
                                            <p className="text-green-700 dark:text-green-300 text-sm font-medium">
                                                Classificazione completata con successo!
                                            </p>
                                            {selectedFile && (
                                                <p className="text-green-600 dark:text-green-400 text-xs mt-1">
                                                    File
                                                    elaborato: {selectedFile.name} ({formatFileSize(selectedFile.size)})
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};