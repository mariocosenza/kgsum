import Link from "next/link";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export const Footer = () => {
    return (
        <footer className="w-full mt-2">
            <div className="mx-5 lg:mx-20 border-t-2 border-dotted flex flex-row justify-between items-center h-12">
                <div>
                    <p className="text-gray-600 text-sm ml-2">
                        KgSum 2025
                    </p>
                </div>
                <div className="flex items-center gap-2 mr-5">
                    <Link
                        aria-label="GitHub KgSum"
                        href="https://github.com/mariocosenza/kgsum/"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center"
                    >
                        <Avatar className="hidden md:block h-6 w-6 mr-2 ">
                            <AvatarImage src="https://avatars.githubusercontent.com/u/61911701?v=4" />
                            <AvatarFallback>MC</AvatarFallback>
                        </Avatar>
                    </Link>
                    <span className="text-sm text-gray-600">
                        Un progetto di{" "}
                        <Link
                            href="https://github.com/mariocosenza"
                            className="underline"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            @mariocosenza
                        </Link>
                    </span>
                </div>
            </div>
        </footer>
    );
};