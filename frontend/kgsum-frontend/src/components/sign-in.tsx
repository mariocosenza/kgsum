'use client'
import {ReactNode} from "react";
import {SignedOut, SignInButton} from "@clerk/nextjs";
import {usePathname} from "next/navigation";
import {Button} from "@/components/ui/button";

export default function CustomSignIn(): ReactNode {
    const path = usePathname()
    return (
        path.includes('auth') ?
            <div></div>
            :
            <SignedOut>
                <SignInButton mode="modal">
                    <Button>Login</Button>
                </SignInButton>
            </SignedOut>

    )
}