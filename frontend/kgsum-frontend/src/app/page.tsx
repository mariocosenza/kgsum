import {Button} from "@/components/ui/button";
import Image from "next/image";
import Link from "next/link";

export default function Home() {
    return (
        <main className="flex h-[78vh] lg:h-[85vh] px-10 w-full items-center justify-evenly">
            <div className="w-full lg:w-1/2 h-[90%] flex flex-col justify-center">
                <h1 className="scroll-m-20 text-center text-5xl font-extrabold tracking-tight text-balance mb-8">
                    Automatic Classification and Profiling of Knowledge Graphs
                </h1>
                <p className="leading-7 text-center mb-10 max-w-3xl mx-auto">
                    A system that automates the semantic analysis of Knowledge Graphs, identifies their thematic domains
                    and generates detailed profiles to facilitate their integration and use.
                </p>

                {/* Button row always matches the width of the h1/p, and buttons fully fill the row */}
                <div className="max-w-3xl w-full mx-auto flex gap-6 justify-center">
                    <div className="flex-1 cursor-pointer">
                        <Link href={"/classify"} className="cursor-pointer">
                            <Button className="w-full h-14 text-lg font-semibold  cursor-pointer">
                                Classify
                            </Button>
                        </Link>
                    </div>
                    <div className="flex-1">
                        <Link href={"/explore"}>
                            <Button
                                variant="secondary"
                                className="w-full h-14 text-lg font-semibold cursor-pointer"
                            >
                                Explore
                            </Button>
                        </Link>
                    </div>
                </div>
            </div>
            <div className="min-w-4/11 h-[90%] hidden lg:flex p-4">
                <div className="relative w-full h-full rounded-3xl overflow-hidden">
                    <Image
                        alt="Banner"
                        src="/banner.png"
                        fill
                        className="object-cover"
                        priority
                    />
                </div>
            </div>
        </main>
    );
}