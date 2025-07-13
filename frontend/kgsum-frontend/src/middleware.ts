import {clerkMiddleware, createRouteMatcher} from '@clerk/nextjs/server';

// Optional protection based on env, default true
const CLERK_MIDDLEWARE_ENABLED = process.env.CLERK_MIDDLEWARE_ENABLED ?? 'true';
const isClerkMiddlewareEnabled = CLERK_MIDDLEWARE_ENABLED.toLowerCase() !== 'false';

const isProtectedRoute = createRouteMatcher(['/classify(.*)', '/api(.*)']);

export default clerkMiddleware(async (auth, req) => {
    // If disabled, do nothing (always allow)
    if (!isClerkMiddlewareEnabled) return;

    if (isProtectedRoute(req)) {
        const baseUrl = new URL(req.url).origin;

        await auth.protect({
            // Use absolute URLs to avoid Clerk errors
            unauthenticatedUrl: `${baseUrl}/auth/sign-in`,
            unauthorizedUrl: `${baseUrl}/auth/sign-in`,
        });
    }
});

export const config = {
    matcher: [
        // Skip Next.js internals and all static files, unless found in search params
        '/((?!_next|[^?]*\\.(?:html?|css|js(?!on)|jpe?g|webp|png|gif|svg|ttf|woff2?|ico|csv|docx?|xlsx?|zip|webmanifest)).*)',
        // Always run for API routes
        '/(api|trpc)(.*)',
    ],
};