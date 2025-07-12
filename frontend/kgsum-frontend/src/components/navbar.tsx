import {ReactNode} from "react";
import Link from "next/link";
import {
    NavigationMenu,
    NavigationMenuContent,
    NavigationMenuItem,
    NavigationMenuLink,
    NavigationMenuList,
    NavigationMenuTrigger
} from "@/components/ui/navigation-menu";
import {Avatar, AvatarFallback, AvatarImage} from "@/components/ui/avatar";
import Image from "next/image";
import {ModeToggle} from "@/components/theme-toggle";
import {SignedIn, UserButton} from "@clerk/nextjs";

import CustomSignIn from "@/components/sign-in";

// Consistent styling for all main navigation links
const menuLinkClass = "font-normal px-3 py-2 transition-colors hover:bg-accent rounded-md";
const menuLinkBoldClass = "font-bold px-3 py-2 transition-colors hover:bg-accent rounded-md";

export const NavBar = (): ReactNode => {
    return (
        <nav className="pt-4 pb-2 px-5 border-b-2 border-dotted flex items-center">
            <Link href="/">
                <Image
                    src="/logo.png"
                    className="hidden md:block"
                    width={80}
                    height={80}
                    alt="KgSum Logo"
                />
            </Link>
            <NavigationMenu>
                <NavigationMenuList>
                    <NavigationMenuItem>
                        <NavigationMenuLink asChild>
                            <Link href="/" className={menuLinkClass}>Home</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>
                    <NavigationMenuItem>
                        <NavigationMenuTrigger className={menuLinkClass}>
                            Try KgSum
                        </NavigationMenuTrigger>
                        <NavigationMenuContent className="font-normal p-2 min-w-[30vw]">
                            <div className="flex flex-row">
                                <div className="basis-1/4 relative min-h-[200px]">
                                    <Image
                                        src="/banner-menu.png"
                                        alt="Menu Banner"
                                        fill
                                        sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 300px"
                                        className="object-cover rounded-md"
                                        priority={false}
                                    />
                                </div>
                                <div className="basis-3/4 pl-2">
                                    <NavigationMenuLink
                                        href="/explore"
                                        className={menuLinkBoldClass}
                                    >
                                        Explore Profiles
                                        <p className="text-muted-foreground text-sm font-normal">
                                            Browse available Knowledge Graphs and discover their metadata.
                                        </p>
                                    </NavigationMenuLink>
                                    <NavigationMenuLink
                                        href="/classify"
                                        className={menuLinkBoldClass}
                                    >
                                        Classify Online
                                        <p className="text-muted-foreground text-sm font-normal">
                                            Access the Knowledge Graph classification service in real-time.
                                        </p>
                                    </NavigationMenuLink>
                                    <NavigationMenuLink
                                        href="/documentation/latest"
                                        className={menuLinkBoldClass}
                                    >
                                        API Documentation
                                        <p className="text-muted-foreground text-sm font-normal">
                                            Consult guides and examples to integrate our classification APIs.
                                        </p>
                                    </NavigationMenuLink>

                                </div>
                            </div>
                        </NavigationMenuContent>
                    </NavigationMenuItem>
                    <NavigationMenuItem>
                        <NavigationMenuLink asChild>
                            <Link href="/statistics" className={menuLinkClass}>Statistics</Link>
                        </NavigationMenuLink>
                    </NavigationMenuItem>
                </NavigationMenuList>
            </NavigationMenu>
            <ModeToggle/>
            <Avatar className="ml-3 mr-3" aria-label="GitHub KgSum">
                <Link
                    aria-label="https://github.com/mariocosenza/kgsum/"
                    href={"https://github.com/mariocosenza/kgsum/"}
                >
                    <AvatarImage src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png"/>
                    <AvatarFallback>GH</AvatarFallback>
                </Link>
            </Avatar>
            <SignedIn>
                <UserButton/>
            </SignedIn>
            <CustomSignIn/>
        </nav>
    );
};