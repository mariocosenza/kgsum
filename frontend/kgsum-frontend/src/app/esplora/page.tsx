"use client"
import { useRef, ReactNode } from "react";

export default function QueryBuilder(): ReactNode {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const handleIframeLoad = () => {
    try {
      const iframeDoc = iframeRef.current?.contentWindow?.document;
      if (!iframeDoc) return;

      const modals = iframeDoc.querySelectorAll<HTMLElement>('.cookie-consent-modal');
      modals.forEach(modal => {
        modal.style.display = 'none';
      });
    } catch (e) {
      console.error('Impossibile accedere al contenuto dell’iframe:', e);
    }
  };

  return (
    <iframe
      ref={iframeRef}
      src="http://localhost:7200/sparql"
      className="grow"
      onLoad={handleIframeLoad}
    />
  );
}
