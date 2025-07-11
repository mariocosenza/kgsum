'use client'
import Link from 'next/link'
import {Button} from '@/components/ui/button'
import {Card, CardContent, CardDescription, CardHeader, CardTitle} from '@/components/ui/card'
import {ArrowLeft, Home} from 'lucide-react'

export default function NotFound() {
    return (
        <div
            className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 p-4">
            <Card className="w-full max-w-md text-center shadow-lg">
                <CardHeader className="space-y-4">
                    <div
                        className="mx-auto w-20 h-20 bg-red-100 dark:bg-red-900/20 rounded-full flex items-center justify-center">
                        <span className="text-4xl font-bold text-red-600 dark:text-red-400">404</span>
                    </div>
                    <div className="space-y-2">
                        <CardTitle className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                            Page Not Found
                        </CardTitle>
                        <CardDescription className="text-gray-600 dark:text-gray-400">
                            Sorry, we can&#39;t find the page you&#39;re looking for. It might have been moved, deleted, or you
                            might have entered an incorrect URL.
                        </CardDescription>
                    </div>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex flex-col sm:flex-row gap-3 justify-center">
                        <Button asChild className="flex items-center gap-2">
                            <Link href="/">
                                <Home className="w-4 h-4"/>
                                Go to Home
                            </Link>
                        </Button>
                        <Button variant="outline" onClick={() => window.history.back()}
                                className="flex items-center gap-2">
                            <ArrowLeft className="w-4 h-4"/>
                            Go Back
                        </Button>
                    </div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">
                        If you think this is an error, please{' '}
                        <Link href="mailto:cosenzamario@proton.me"
                              className="text-blue-600 dark:text-blue-400 hover:underline">
                            contact support
                        </Link>
                    </div>
                </CardContent>
            </Card>
        </div>
    )
}