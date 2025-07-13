'use server'

import {auth} from "@clerk/nextjs/server"; // For server actions
// If you use client components, use: import { useAuth } from "@clerk/nextjs";

export type FormState = {
    message: string;
    data?: Record<string, unknown>;
}

export async function createPost(
    prevState: FormState,
    formData: FormData
): Promise<FormState> {
    try {
        let token: string | null = "";
        const enabled: boolean = process.env.CLERK_MIDDLEWARE_ENABLED === 'true';
        const mode = formData.get('mode') as string;
        const privacyConsent = formData.get('privacyConsent');
        const saveProfile = formData.get('saveProfile');

        // Constants
        const MAX_FILE_SIZE = 524288000; // 500MB in bytes
        const API_BASE_URL = process.env.CLASSIFICATION_API_URL || 'http://localhost:5000';
        // Get Clerk JWT token for authenticated user
        if (enabled) {
            const {getToken} = await auth();
            token = await getToken();
            if (!token) {
                return {
                    message: "Error: You need to be authenticated to use this function.",
                };
            }
        }


        // Check privacy consent requirement
        if (!privacyConsent) {
            return {
                message: "Error: You need to accept terms and condition to precede.",
            };
        }

        // Process based on selected mode
        if (mode === 'SPARQL') {
            const sparqlUrl = formData.get('sparqlUrl') as string;

            if (!sparqlUrl || !sparqlUrl.trim()) {
                return {
                    message: "Error: The SPARQL URL is obligatory for this mode.",
                };
            }

            try {
                // Validate URL format
                new URL(sparqlUrl);
            } catch {
                return {
                    message: "Error: The SPARQL URL is not valid.",
                };
            }

            try {
                // Create request payload for SPARQL endpoint - matching your API
                const requestBody = {
                    endpoint: sparqlUrl, // Your API expects 'endpoint', not 'sparqlUrl'
                    store: !!saveProfile, // Boolean value as expected by your API
                };

                // Send to your Flask API SPARQL endpoint - NO TIMEOUT
                const response = await fetch(`${API_BASE_URL}/api/v1/profile/sparql`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json',
                        'Authorization': `Bearer ${token}`,
                    },
                    body: JSON.stringify(requestBody),
                    // No signal/timeout - let it run as long as needed
                });

                if (!response.ok) {
                    let errorMessage = `Errore API (${response.status})`;

                    try {
                        const errorData = await response.json();
                        if (errorData.error) {
                            errorMessage = errorData.error;

                            // Handle specific Flask API errors
                            if (response.status === 429) {
                                errorMessage = "Overloaded Server: too many active request. Retry later.";
                            } else if (response.status === 500 && errorData.cpu_usage) {
                                errorMessage = `Overloaded Server: CPU ${errorData.cpu_usage}%, RAM ${errorData.ram_usage}%`;
                            }
                        }
                    } catch {
                        // If we can't parse the error, use the generic message
                    }

                    return {
                        message: `Errore: ${errorMessage}`,
                    };
                }

                const apiResult = await response.json();

                return {
                    message: "SPARQL endpoint profiled successfully!",
                    data: {
                        source: 'SPARQL',
                        endpoint: sparqlUrl,
                        processedAt: new Date().toISOString(),
                        profileSaved: !!saveProfile,
                        result: apiResult,
                    },
                };

            } catch (error) {
                if (error instanceof Error) {
                    if (error.message.includes('fetch') || error.message.includes('network')) {
                        return {
                            message: "Error: Can not reach classification API. Verify that the server is active on port 5000.",
                        };
                    }
                }

                console.error('SPARQL classification error:', error);
                return {
                    message: "Error: An error occurred during SPARQL classification.",
                };
            }

        } else if (mode === 'DUMP') {
            const file = formData.get('file') as File;

            if (!file || file.size === 0) {
                return {
                    message: "Error: Select a valid RDF file.",
                };
            }

            // Check file size (500MB limit)
            if (file.size > MAX_FILE_SIZE) {
                const sizeMB = Math.round(file.size / (1024 * 1024));
                return {
                    message: `Error: The file (${sizeMB}MB) is over the 500MB size threshold. Upload a smaller file.`,
                };
            }

            // Validate file extension
            const allowedExtensions = ['.rdf', '.ttl', '.nq', '.nt', '.xml', '.json'];
            const fileName = file.name.toLowerCase();
            const fileExtension = '.' + fileName.split('.').pop();

            if (!allowedExtensions.includes(fileExtension)) {
                return {
                    message: `Error: File format "${fileExtension}" not support. Accepted format: ${allowedExtensions.join(', ')}`,
                };
            }

            try {
                // Create FormData for file upload to your Flask API
                const apiFormData = new FormData();
                apiFormData.append('file', file);

                // Build URL with store parameter as query parameter (as your API expects)
                const storeParam = saveProfile ? 'true' : 'false';
                const apiUrl = `${API_BASE_URL}/api/v1/profile/file?store=${storeParam}`;

                // Send file to your Flask API file endpoint - NO TIMEOUT
                const response = await fetch(apiUrl, {
                    method: 'POST',
                    body: apiFormData, // multipart/form-data as expected by your API
                    headers: {
                        'Accept': 'application/json',
                        'Authorization': `Bearer ${token}`,
                        // Don't set Content-Type - let browser set it with boundary for multipart
                    },
                    // No signal/timeout - let it run as long as needed for large files
                });

                // Check if response is successful FIRST
                if (response.ok) {
                    // Success case - parse JSON and return success
                    try {
                        const apiResult = await response.json();
                        const fileSizeMB = Math.round(file.size / (1024 * 1024) * 100) / 100;

                        return {
                            message: "RDF file profiled successfully!",
                            data: {
                                source: 'FILE',
                                file: {
                                    name: file.name,
                                    size: file.size,
                                    sizeMB: fileSizeMB,
                                    type: file.type,
                                    extension: fileExtension,
                                },
                                processedAt: new Date().toISOString(),
                                profileSaved: !!saveProfile,
                                result: apiResult,
                            },
                        };
                    } catch (jsonError) {
                        console.error('Error parsing successful API response:', jsonError);
                        return {
                            message: "Error parsing successful API response.",
                        };
                    }
                } else {
                    // Error case - handle API errors
                    let errorMessage = `Errore API (${response.status})`;

                    try {
                        const errorData = await response.json();
                        if (errorData.error) {
                            errorMessage = errorData.error;

                            // Handle specific Flask API errors
                            if (response.status === 400) {
                                if (errorData.error.includes("No file part")) {
                                    errorMessage = "Error: File non received from API.";
                                } else if (errorData.error.includes("File type not allowed")) {
                                    errorMessage = "Error: File format not supported by the API.";
                                }
                            } else if (response.status === 429) {
                                errorMessage = "Overloaded Server: too many active request. Retry later.";
                            } else if (response.status === 500) {
                                if (errorData.cpu_usage) {
                                    errorMessage = `Overloaded Server: CPU ${errorData.cpu_usage}%, RAM ${errorData.ram_usage}%`;
                                } else if (errorData.error.includes("Profile generation failed")) {
                                    errorMessage = "Error during RDF profile generation.";
                                }
                            }
                        }
                    } catch {
                        // If we can't parse the error, use the generic message
                    }

                    return {
                        message: `Errore: ${errorMessage}`,
                    };
                }

            } catch (error) {
                console.error('File classification error:', error);

                if (error instanceof Error) {
                    if (error.message.includes('fetch') || error.message.includes('network')) {
                        return {
                            message: "Error: Can not reach classification API. Verify that the server is active on port 5000.",
                        };
                    }
                }

                return {
                    message: "Error: An error occurred during RDF classification process.",
                };
            }
        }

        return {
            message: "Error: Elaboration mode not supported.",
        };

    } catch (error) {
        console.error('Server action error:', error);

        // Handle specific error types
        if (error instanceof Error) {
            if (error.name === 'PayloadTooLargeError' || error.message.includes('PayloadTooLarge')) {
                return {
                    message: "Error: File too large. Size limit is 500MB.",
                };
            }
            if (error.message.includes('network') || error.message.includes('fetch')) {
                return {
                    message: "Error: Can not reach classification API. Verify that the server is active on port 5000.",
                };
            }
        }

        return {
            message: "Error: An unknown error occurred during the profiling. Retry later.",
        };
    }
}