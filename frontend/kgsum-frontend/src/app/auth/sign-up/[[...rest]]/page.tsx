import {ReactNode} from "react";
import {SignUp} from "@clerk/nextjs";

export default function Registrati(): ReactNode {
    return (
        <div className="w-full h-auto mt-10 flex justify-center items-center">
            <SignUp/>
        </div>
    )
}