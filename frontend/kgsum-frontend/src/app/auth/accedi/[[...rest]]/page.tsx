import {ReactNode} from "react";
import {SignIn} from "@clerk/nextjs";

export default function Login(): ReactNode {
    return (
        <div className="w-full h-auto mt-10 flex justify-center items-center">
            <SignIn/>
        </div>
    )
}