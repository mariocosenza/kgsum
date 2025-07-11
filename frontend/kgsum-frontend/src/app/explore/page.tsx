"use client"
import {ReactNode} from "react";

export default function QueryBuilder(): ReactNode {

    return (
        <iframe
            src="http://localhost:7200/sparql"
            className="grow"
        />
    );
}
