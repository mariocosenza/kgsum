import Markdown from "react-markdown";
import { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export type ReadmeFile = {
    content: string;
    className?: string;
}

export function DocumentTemplate({ content, className }: ReadmeFile): ReactNode {
    return (
        <div className={cn("max-w-none prose prose-slate dark:prose-invert mx-4 md:mx-6 lg:mx-8 mt-4", className)}>
            <Markdown
                components={{
                    // Headers
                    h1: ({ children }) => (
                        <div className="space-y-4 mb-8">
                            <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl">
                                {children}
                            </h1>
                            <Separator className="my-4" />
                        </div>
                    ),
                    h2: ({ children }) => (
                        <h2 className="scroll-m-20 border-b pb-2 text-3xl font-semibold tracking-tight first:mt-0 mt-10 mb-4">
                            {children}
                        </h2>
                    ),
                    h3: ({ children }) => (
                        <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight mt-8 mb-4">
                            {children}
                        </h3>
                    ),
                    h4: ({ children }) => (
                        <h4 className="scroll-m-20 text-xl font-semibold tracking-tight mt-6 mb-3">
                            {children}
                        </h4>
                    ),

                    // Paragraphs
                    p: ({ children }) => (
                        <p className="leading-7 [&:not(:first-child)]:mt-6">
                            {children}
                        </p>
                    ),

                    // Lists
                    ul: ({ children }) => (
                        <ul className="my-6 ml-6 list-disc [&>li]:mt-2">
                            {children}
                        </ul>
                    ),
                    ol: ({ children }) => (
                        <ol className="my-6 ml-6 list-decimal [&>li]:mt-2">
                            {children}
                        </ol>
                    ),
                    li: ({ children }) => (
                        <li className="leading-7">
                            {children}
                        </li>
                    ),

                    // Code blocks
                    code: ({ className, children, ...props }) => {
                        const match = /language-(\w+)/.exec(className || '');
                        const language = match ? match[1] : '';
                        const isInline = !match;

                        if (!isInline && language) {
                            return (
                                <div className="my-6">
                                    <div className="flex items-center justify-between px-4 py-2 bg-slate-800 dark:bg-slate-900 rounded-t-lg">
                                        <Badge variant="secondary" className="text-xs font-mono">
                                            {language}
                                        </Badge>
                                    </div>
                                    <pre className="p-4 bg-slate-900 dark:bg-slate-800 text-slate-100 overflow-x-auto rounded-b-lg">
                                        <code className="font-mono text-sm leading-relaxed" {...props}>
                                            {children}
                                        </code>
                                    </pre>
                                </div>
                            );
                        }

                        return (
                            <code className="relative rounded bg-muted/60 px-[0.3rem] py-[0.2rem] font-mono text-sm font-medium text-foreground" {...props}>
                                {children}
                            </code>
                        );
                    },

                    // Blockquotes
                    blockquote: ({ children }) => (
                        <blockquote className="mt-6 border-l-2 border-primary pl-6 italic text-muted-foreground">
                            {children}
                        </blockquote>
                    ),

                    // Tables
                    table: ({ children }) => (
                        <Card className="my-6 border-border">
                            <CardContent className="p-0">
                                <div className="overflow-x-auto">
                                    <table className="w-full">
                                        {children}
                                    </table>
                                </div>
                            </CardContent>
                        </Card>
                    ),
                    thead: ({ children }) => (
                        <thead className="bg-muted/50">
                            {children}
                        </thead>
                    ),
                    th: ({ children }) => (
                        <th className="border-b border-border px-4 py-3 text-left font-medium [&[align=center]]:text-center [&[align=right]]:text-right">
                            {children}
                        </th>
                    ),
                    td: ({ children }) => (
                        <td className="border-b border-border px-4 py-3 [&[align=center]]:text-center [&[align=right]]:text-right">
                            {children}
                        </td>
                    ),

                    // Horizontal rule
                    hr: () => <Separator className="my-8" />,

                    // Strong and emphasis
                    strong: ({ children }) => (
                        <strong className="font-semibold text-foreground">{children}</strong>
                    ),
                    em: ({ children }) => (
                        <em className="italic">{children}</em>
                    ),

                    // Links
                    a: ({ href, children }) => (
                        <a
                            href={href}
                            className="font-medium text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            {children}
                        </a>
                    ),
                }}
            >
                {content}
            </Markdown>
        </div>
    );
}