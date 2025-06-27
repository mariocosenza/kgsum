import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Mail, Shield, Database, Brain, Users, FileText, Cookie, Lock } from "lucide-react"

export default function PrivacyPolicyPage() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-4xl">
      <div className="space-y-8">
        {/* Header */}
        <div className="text-center space-y-4">
          <div className="flex items-center justify-center gap-2 mb-4">
            <Shield className="h-8 w-8 text-primary" />
            <h1 className="text-4xl font-bold">Privacy Policy</h1>
          </div>
          <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
            <Badge variant="outline">KgSum</Badge>
            <span>•</span>
            <span>Ultima modifica: 27 giugno 2025</span>
            <span>•</span>
            <span>Versione 1.0</span>
          </div>
        </div>

        {/* Introduzione */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-5 w-5" />
              Introduzione
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <p>
              Benvenuto su <strong>KgSum</strong>, un'applicazione web per la ricerca sui Knowledge Graph
              sviluppata come progetto di tesi. La presente Privacy Policy descrive come raccogliamo,
              utilizziamo e proteggiamo le tue informazioni personali quando utilizzi il nostro servizio.
            </p>
            <div className="bg-muted p-4 rounded-lg">
              <p className="font-semibold">Responsabile del trattamento:</p>
              <div className="flex items-center gap-2 mt-2">
                <Mail className="h-4 w-4" />
                <span>Mario Cosenza - cosenzamario@proton.me</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Informazioni raccolte */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Database className="h-5 w-5" />
              Informazioni che raccogliamo
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <h3 className="font-semibold mb-2">2.1 Dati di autenticazione</h3>
              <p className="mb-3">Tramite il servizio Clerk, raccogliamo i seguenti dati per l'autenticazione:</p>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li><strong>GitHub/GitLab:</strong> Nome utente, email, avatar pubblico</li>
                <li><strong>Email:</strong> Indirizzo email e password (crittografata)</li>
                <li><strong>Dati di sessione:</strong> Token di accesso e informazioni di autenticazione</li>
              </ul>
            </div>

            <Separator />

            <div>
              <h3 className="font-semibold mb-2">2.2 Dati di utilizzo del sito</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li><strong>Cookie di GraphDB:</strong> Per il funzionamento del database grafico</li>
                <li><strong>Dati di navigazione:</strong> Pagine visitate, tempo di permanenza, interazioni</li>
                <li><strong>Informazioni tecniche:</strong> Indirizzo IP, browser, sistema operativo</li>
              </ul>
            </div>

            <Separator />

            <div>
              <h3 className="font-semibold mb-2">2.3 Contenuti caricati dall'utente</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li><strong>File caricati:</strong> Documenti e dataset per l'elaborazione</li>
                <li><strong>Query SPARQL:</strong> Le query eseguite sui Knowledge Graph</li>
                <li><strong>Risultati delle elaborazioni:</strong> Output delle analisi e classificazioni</li>
              </ul>
            </div>
          </CardContent>
        </Card>

        {/* Come utilizziamo le informazioni */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5" />
              Come utilizziamo le tue informazioni
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div>
              <h3 className="font-semibold mb-2">3.1 Finalità del trattamento</h3>
              <p className="mb-3">Le tue informazioni vengono utilizzate per:</p>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li>Fornire l'accesso autenticato alla piattaforma</li>
                <li>Elaborare i file caricati tramite algoritmi di Machine Learning locali</li>
                <li>Classificare Knowledge Graph tramite endpoint SPARQL</li>
                <li>Migliorare il servizio e condurre ricerca accademica</li>
                <li>Garantire la sicurezza e il corretto funzionamento della piattaforma</li>
              </ul>
            </div>

            <Separator />

            <div>
              <h3 className="font-semibold mb-2">3.2 Elaborazione dei dati</h3>
              <div className="space-y-2">
                <p><strong>ML locale:</strong> I file vengono processati principalmente con soluzioni di Machine Learning eseguite sui nostri server</p>
                <p><strong>Servizi esterni:</strong> In alcuni casi, i dati potrebbero essere inviati a:</p>
                <ul className="list-disc list-inside ml-4 space-y-1 text-sm">
                  <li><strong>Google Gemini API</strong> per elaborazioni avanzate</li>
                  <li><strong>Linked Open Vocabularies API</strong> per arricchimento semantico</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Base giuridica */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="h-5 w-5" />
              Base giuridica del trattamento
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3">Il trattamento dei tuoi dati si basa su:</p>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li><strong>Consenso:</strong> Per l'utilizzo di cookie non essenziali e l'elaborazione dei file caricati</li>
              <li><strong>Interesse legittimo:</strong> Per la ricerca accademica e il miglioramento del servizio</li>
              <li><strong>Esecuzione del contratto:</strong> Per fornire i servizi richiesti</li>
            </ul>
          </CardContent>
        </Card>

        {/* Cookie */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cookie className="h-5 w-5" />
              Cookie e tecnologie di tracciamento
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Tipi di cookie utilizzati</h3>
              <ul className="list-disc list-inside space-y-1 text-sm">
                <li><strong>Cookie essenziali:</strong> Per l'autenticazione e il funzionamento del sito</li>
                <li><strong>Cookie di GraphDB:</strong> Per la gestione del database grafico</li>
                <li><strong>Cookie di sessione:</strong> Per mantenere la sessione utente attiva</li>
              </ul>
            </div>
            <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
              <p className="text-sm">
                <strong>Gestione dei cookie:</strong> Puoi gestire le preferenze sui cookie tramite le impostazioni del tuo browser.
                La disabilitazione di alcuni cookie potrebbe limitare le funzionalità del sito.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* I tuoi diritti */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              I tuoi diritti
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3">Secondo il GDPR, hai diritto a:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="space-y-2">
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Accesso</p>
                  <p className="text-xs text-muted-foreground">Richiedere una copia dei tuoi dati personali</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Rettifica</p>
                  <p className="text-xs text-muted-foreground">Correggere dati inesatti o incompleti</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Cancellazione</p>
                  <p className="text-xs text-muted-foreground">Richiedere la rimozione dei tuoi dati</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Limitazione</p>
                  <p className="text-xs text-muted-foreground">Limitare il trattamento in determinate circostanze</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Portabilità</p>
                  <p className="text-xs text-muted-foreground">Ricevere i tuoi dati in formato strutturato</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Opposizione</p>
                  <p className="text-xs text-muted-foreground">Opporti al trattamento per motivi legittimi</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <p className="font-semibold text-sm">Revoca del consenso</p>
                  <p className="text-xs text-muted-foreground">Revocare il consenso in qualsiasi momento</p>
                </div>
              </div>
            </div>
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <p className="text-sm">
                <strong>Per esercitare questi diritti:</strong> Contatta cosenzamario@proton.me
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Sicurezza */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              Sicurezza dei dati
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-3">Implementiamo misure di sicurezza appropriate per proteggere i tuoi dati:</p>
            <ul className="list-disc list-inside space-y-1 text-sm">
              <li>Crittografia dei dati in transito e a riposo</li>
              <li>Accesso limitato ai dati del personale autorizzato</li>
              <li>Monitoraggio regolare della sicurezza</li>
              <li>Backup sicuri e procedure di ripristino</li>
            </ul>
          </CardContent>
        </Card>

        {/* Contatti */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Mail className="h-5 w-5" />
              Contatti
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="mb-4">
              Per qualsiasi domanda riguardo questa Privacy Policy o per esercitare i tuoi diritti, contatta:
            </p>
            <div className="bg-muted p-4 rounded-lg">
              <p className="font-semibold">Mario Cosenza</p>
              <div className="flex items-center gap-2 mt-2">
                <Mail className="h-4 w-4" />
                <a href="mailto:cosenzamario@proton.me" className="text-primary hover:underline">
                  cosenzamario@proton.me
                </a>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Footer della policy */}
        <div className="text-center text-sm text-muted-foreground border-t pt-6">
          <p>
            <strong>Ultima modifica:</strong> 27 giugno 2025 | <strong>Versione:</strong> 1.0
          </p>
          <p className="mt-2">
            Hai il diritto di presentare reclamo al{" "}
            <a href="https://www.gpdp.it" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
              Garante per la protezione dei dati personali
            </a>
          </p>
        </div>
      </div>
    </div>
  )
}