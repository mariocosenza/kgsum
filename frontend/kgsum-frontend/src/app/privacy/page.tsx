import {Card, CardContent, CardHeader, CardTitle} from "@/components/ui/card"
import {Separator} from "@/components/ui/separator"
import {Badge} from "@/components/ui/badge"
import {Brain, Cookie, Database, FileText, Lock, Mail, Shield, Users} from "lucide-react"

export default function PrivacyPolicyPage() {
    return (
        <div className="container mx-auto px-4 py-8 max-w-4xl">
            <div className="space-y-8">
                {/* Header */}
                <div className="text-center space-y-4">
                    <div className="flex items-center justify-center gap-2 mb-4">
                        <Shield className="h-8 w-8 text-primary"/>
                        <h1 className="text-4xl font-bold">Privacy Policy</h1>
                    </div>
                    <div className="flex items-center justify-center gap-4 text-sm text-muted-foreground">
                        <Badge variant="outline">KgSum</Badge>
                        <span>•</span>
                        <span>Last updated: June 27, 2025</span>
                        <span>•</span>
                        <span>Version 1.0</span>
                    </div>
                </div>

                {/* Introduction */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Users className="h-5 w-5"/>
                            Introduction
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <p>
                            Welcome to <strong>KgSum</strong>, a web application for Knowledge Graph research
                            developed as a thesis project. This Privacy Policy describes how we collect,
                            use, and protect your personal information when you use our service.
                        </p>
                        <div className="bg-muted p-4 rounded-lg">
                            <p className="font-semibold">Data Controller:</p>
                            <div className="flex items-center gap-2 mt-2">
                                <Mail className="h-4 w-4"/>
                                <span>Mario Cosenza - cosenzamario@proton.me</span>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Information we collect */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Database className="h-5 w-5"/>
                            Information we collect
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div>
                            <h3 className="font-semibold mb-2">2.1 Authentication data</h3>
                            <p className="mb-3">Through the Clerk service, we collect the following data for
                                authentication:</p>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                                <li><strong>GitHub/GitLab:</strong> Username, email, public avatar</li>
                                <li><strong>Email:</strong> Email address and password (encrypted)</li>
                                <li><strong>Session data:</strong> Access tokens and authentication information</li>
                            </ul>
                        </div>

                        <Separator/>

                        <div>
                            <h3 className="font-semibold mb-2">2.2 Site usage data</h3>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                                <li><strong>GraphDB cookies:</strong> For the operation of the graph database</li>
                                <li><strong>Navigation data:</strong> Pages visited, time spent, interactions</li>
                                <li><strong>Technical information:</strong> IP address, browser, operating system</li>
                            </ul>
                        </div>

                        <Separator/>

                        <div>
                            <h3 className="font-semibold mb-2">2.3 User-uploaded content</h3>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                                <li><strong>Uploaded files:</strong> Documents and datasets for processing</li>
                                <li><strong>SPARQL queries:</strong> Queries executed on Knowledge Graphs</li>
                                <li><strong>Processing results:</strong> Output of analyses and classifications</li>
                            </ul>
                        </div>
                    </CardContent>
                </Card>

                {/* How we use your information */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Brain className="h-5 w-5"/>
                            How we use your information
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div>
                            <h3 className="font-semibold mb-2">3.1 Purpose of processing</h3>
                            <p className="mb-3">Your information is used to:</p>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                                <li>Provide authenticated access to the platform</li>
                                <li>Process uploaded files through local Machine Learning algorithms</li>
                                <li>Classify Knowledge Graphs through SPARQL endpoints</li>
                                <li>Improve the service and conduct academic research</li>
                                <li>Ensure security and proper functioning of the platform</li>
                            </ul>
                        </div>

                        <Separator/>

                        <div>
                            <h3 className="font-semibold mb-2">3.2 Data processing</h3>
                            <div className="space-y-2">
                                <p><strong>Local ML:</strong> Files are processed primarily with Machine Learning
                                    solutions executed on our servers</p>
                                <p><strong>External services:</strong> In some cases, data may be sent to:</p>
                                <ul className="list-disc list-inside ml-4 space-y-1 text-sm">
                                    <li><strong>Google Gemini API</strong> for advanced processing</li>
                                    <li><strong>Linked Open Vocabularies API</strong> for semantic enrichment</li>
                                </ul>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Legal basis */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <FileText className="h-5 w-5"/>
                            Legal basis for processing
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="mb-3">The processing of your data is based on:</p>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                            <li><strong>Consent:</strong> For the use of non-essential cookies and processing of
                                uploaded files
                            </li>
                            <li><strong>Legitimate interest:</strong> For academic research and service improvement</li>
                            <li><strong>Contract performance:</strong> To provide the requested services</li>
                        </ul>
                    </CardContent>
                </Card>

                {/* Cookies */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Cookie className="h-5 w-5"/>
                            Cookies and tracking technologies
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div>
                            <h3 className="font-semibold mb-2">Types of cookies used</h3>
                            <ul className="list-disc list-inside space-y-1 text-sm">
                                <li><strong>Essential cookies:</strong> For authentication and site functionality</li>
                                <li><strong>GraphDB cookies:</strong> For graph database management</li>
                                <li><strong>Session cookies:</strong> To maintain active user sessions</li>
                            </ul>
                        </div>
                        <div
                            className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
                            <p className="text-sm">
                                <strong>Cookie management:</strong> You can manage cookie preferences through your
                                browser settings.
                                Disabling some cookies may limit site functionality.
                            </p>
                        </div>
                    </CardContent>
                </Card>

                {/* Your rights */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Shield className="h-5 w-5"/>
                            Your rights
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="mb-3">Under GDPR, you have the right to:</p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            <div className="space-y-2">
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Access</p>
                                    <p className="text-xs text-muted-foreground">Request a copy of your personal data</p>
                                </div>
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Rectification</p>
                                    <p className="text-xs text-muted-foreground">Correct inaccurate or incomplete data</p>
                                </div>
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Erasure</p>
                                    <p className="text-xs text-muted-foreground">Request removal of your data</p>
                                </div>
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Restriction</p>
                                    <p className="text-xs text-muted-foreground">Limit processing under certain
                                        circumstances</p>
                                </div>
                            </div>
                            <div className="space-y-2">
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Portability</p>
                                    <p className="text-xs text-muted-foreground">Receive your data in structured format</p>
                                </div>
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Objection</p>
                                    <p className="text-xs text-muted-foreground">Object to processing for legitimate
                                        reasons</p>
                                </div>
                                <div className="p-3 bg-muted rounded-lg">
                                    <p className="font-semibold text-sm">Withdraw consent</p>
                                    <p className="text-xs text-muted-foreground">Revoke consent at any time</p>
                                </div>
                            </div>
                        </div>
                        <div
                            className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
                            <p className="text-sm">
                                <strong>To exercise these rights:</strong> Contact cosenzamario@proton.me
                            </p>
                        </div>
                    </CardContent>
                </Card>

                {/* Security */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Lock className="h-5 w-5"/>
                            Data security
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="mb-3">We implement appropriate security measures to protect your data:</p>
                        <ul className="list-disc list-inside space-y-1 text-sm">
                            <li>Encryption of data in transit and at rest</li>
                            <li>Limited access to data by authorized personnel</li>
                            <li>Regular security monitoring</li>
                            <li>Secure backups and recovery procedures</li>
                        </ul>
                    </CardContent>
                </Card>

                {/* Contact */}
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Mail className="h-5 w-5"/>
                            Contact
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <p className="mb-4">
                            For any questions regarding this Privacy Policy or to exercise your rights,
                            contact:
                        </p>
                        <div className="bg-muted p-4 rounded-lg">
                            <p className="font-semibold">Mario Cosenza</p>
                            <div className="flex items-center gap-2 mt-2">
                                <Mail className="h-4 w-4"/>
                                <a href="mailto:cosenzamario@proton.me" className="text-primary hover:underline">
                                    cosenzamario@proton.me
                                </a>
                            </div>
                        </div>
                    </CardContent>
                </Card>

                {/* Policy footer */}
                <div className="text-center text-sm text-muted-foreground border-t pt-6">
                    <p>
                        <strong>Last updated:</strong> June 27, 2025 | <strong>Version:</strong> 1.0
                    </p>
                    <p className="mt-2">
                        You have the right to file a complaint with the{" "}
                        <a href="https://www.gpdp.it" target="_blank" rel="noopener noreferrer"
                           className="text-primary hover:underline">
                            Personal Data Protection Authority
                        </a>
                    </p>
                </div>
            </div>
        </div>
    )
}