"use client";

import { useEffect } from "react";
import { AnimatePresence } from "framer-motion";

// ClientBody.tsx modifications

export default function ClientBody({
  children,
}: {
  children: React.ReactNode;
}) {
  // Remove any extension-added classes during hydration
  useEffect(() => {
    // This runs only on the client after hydration
    document.body.className = "antialiased dark";
  }, []);

  return (
    <body
      className="antialiased bg-zinc-950 text-zinc-50 dark"
      suppressHydrationWarning
    >
      <AnimatePresence mode="wait">{children}</AnimatePresence>
    </body>
  );
}
