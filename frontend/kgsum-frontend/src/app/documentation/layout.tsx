import type {Metadata} from "next";
import {SideTags} from "@/components/side-tags";

export const metadata: Metadata = {
    title: "Documentation",
    description: "Thesis project on knowledge graph summarization"
};

export default function RootLayout({
                                       children,
                                   }: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <div className="flex flex-col md:flex-row w-full">
            <SideTags/>
            <div className="flex-1">
                {children}
            </div>
        </div>
    );
}