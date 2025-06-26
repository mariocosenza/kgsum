'use server'

export type FormState = {
  message: string;
  data?: Record<string, unknown>;
}

export async function createPost(
  prevState: FormState,
  formData: FormData
): Promise<FormState> {
  try {
    const mode = formData.get('mode') as string;
    const privacyConsent = formData.get('privacyConsent');
    const saveProfile = formData.get('saveProfile');

    // Constants
    const MAX_FILE_SIZE = 524288000; // 500MB in bytes
    const API_BASE_URL = process.env.CLASSIFICATION_API_URL || 'http://localhost:5000';

    // Check privacy consent requirement
    if (!privacyConsent) {
      return {
        message: "Errore: Devi accettare i termini e condizioni per procedere.",
      };
    }

    // Process based on selected mode
    if (mode === 'SPARQL') {
      const sparqlUrl = formData.get('sparqlUrl') as string;

      if (!sparqlUrl || !sparqlUrl.trim()) {
        return {
          message: "Errore: L'URL SPARQL è obbligatorio per questa modalità.",
        };
      }

      try {
        // Validate URL format
        new URL(sparqlUrl);
      } catch {
        return {
          message: "Errore: L'URL SPARQL fornito non è valido.",
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
                errorMessage = "Server sovraccarico: troppe richieste attive. Riprova tra qualche minuto.";
              } else if (response.status === 500 && errorData.cpu_usage) {
                errorMessage = `Server sovraccarico: CPU ${errorData.cpu_usage}%, RAM ${errorData.ram_usage}%`;
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
          message: "SPARQL endpoint classificato con successo!",
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
              message: "Errore: Impossibile raggiungere l'API di classificazione. Verifica che il server sia attivo su porta 5000.",
            };
          }
        }

        console.error('SPARQL classification error:', error);
        return {
          message: "Errore: Si è verificato un errore durante la classificazione dell'endpoint SPARQL.",
        };
      }

    } else if (mode === 'DUMP') {
      const file = formData.get('file') as File;

      if (!file || file.size === 0) {
        return {
          message: "Errore: Devi selezionare un file RDF valido.",
        };
      }

      // Check file size (500MB limit)
      if (file.size > MAX_FILE_SIZE) {
        const sizeMB = Math.round(file.size / (1024 * 1024));
        return {
          message: `Errore: Il file (${sizeMB}MB) supera il limite di 500MB. Seleziona un file più piccolo.`,
        };
      }

      // Validate file extension
      const allowedExtensions = ['.rdf', '.ttl', '.nq', '.nt', '.xml', '.json'];
      const fileName = file.name.toLowerCase();
      const fileExtension = '.' + fileName.split('.').pop();

      if (!allowedExtensions.includes(fileExtension)) {
        return {
          message: `Errore: Formato file "${fileExtension}" non supportato. Formati accettati: ${allowedExtensions.join(', ')}`,
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
              message: "File RDF classificato con successo!",
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
              message: "Errore: Risposta API non valida nonostante il successo.",
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
                  errorMessage = "Errore: File non ricevuto dall'API.";
                } else if (errorData.error.includes("File type not allowed")) {
                  errorMessage = "Errore: Tipo di file non supportato dall'API.";
                }
              } else if (response.status === 429) {
                errorMessage = "Server sovraccarico: troppe richieste attive. Riprova tra qualche minuto.";
              } else if (response.status === 500) {
                if (errorData.cpu_usage) {
                  errorMessage = `Server sovraccarico: CPU ${errorData.cpu_usage}%, RAM ${errorData.ram_usage}%`;
                } else if (errorData.error.includes("Profile generation failed")) {
                  errorMessage = "Errore durante la generazione del profilo RDF.";
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
              message: "Errore: Impossibile raggiungere l'API di classificazione. Verifica che il server sia attivo su porta 5000.",
            };
          }
        }

        return {
          message: "Errore: Si è verificato un errore durante la classificazione del file RDF.",
        };
      }
    }

    return {
      message: "Errore: Modalità di elaborazione non riconosciuta.",
    };

  } catch (error) {
    console.error('Server action error:', error);

    // Handle specific error types
    if (error instanceof Error) {
      if (error.name === 'PayloadTooLargeError' || error.message.includes('PayloadTooLarge')) {
        return {
          message: "Errore: File troppo grande. Il limite massimo è di 500MB.",
        };
      }
      if (error.message.includes('network') || error.message.includes('fetch')) {
        return {
          message: "Errore: Problema di connessione con l'API di classificazione su porta 5000.",
        };
      }
    }

    return {
      message: "Errore: Si è verificato un errore imprevisto durante l'elaborazione. Riprova più tardi.",
    };
  }
}