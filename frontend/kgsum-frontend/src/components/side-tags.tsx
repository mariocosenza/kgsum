import React, {ReactNode} from "react";
import {ScrollArea} from "@/components/ui/scroll-area";
import {Separator} from "@/components/ui/separator";
import Link from "next/link";
import {VERSIONS} from "@/components/VERSIONS";

const tags = VERSIONS

export const SideTags = (): ReactNode => {
    return (
        <div className="mt-6 mx-4 md:mx-6">
            <ScrollArea className="w-95 md:h-70 md:w-56 rounded-lg border border-border shadow-sm">
                <div className="p-4">
                    <h4 className="mb-4 text-sm font-semibold text-foreground tracking-tight">
                        Tags
                    </h4>
                    <div className="space-y-1">
                        {tags.map((tag, index) => (
                            <React.Fragment key={tag}>
                                <Link
                                    href={`/documentation/${tag}`}
                                    className="block text-sm text-muted-foreground hover:text-foreground transition-colors duration-200 py-2 px-2 rounded-md hover:bg-muted/50"
                                >
                                    {tag}
                                </Link>
                                {index < tags.length - 1 && (
                                    <Separator className="my-1"/>
                                )}
                            </React.Fragment>
                        ))}
                    </div>
                </div>
            </ScrollArea>
        </div>
    )
}