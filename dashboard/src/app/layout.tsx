import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Forge Dashboard",
  description: "Real-time experiment tracking for Forge LLM",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-background text-foreground flex flex-col`}>
        <nav className="glass border-b border-white/10 px-6 py-4 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-6">
              <Link href="/" className="font-bold text-xl tracking-tight text-white flex items-center gap-2">
                <span className="text-blue-500">Forge</span>
                <span className="opacity-50 font-normal">|</span>
                <span className="text-sm font-medium opacity-80">Dashboard</span>
              </Link>
              <div className="flex items-center gap-4 text-sm font-medium opacity-80">
                <Link href="/experiments" className="hover:text-blue-400 transition-colors">Experiments</Link>
                <Link href="/compare" className="hover:text-blue-400 transition-colors">Compare</Link>
                <Link href="/registry" className="hover:text-blue-400 transition-colors">Registry</Link>
              </div>
            </div>
          </div>
        </nav>
        <main className="flex-1 max-w-7xl w-full mx-auto p-6">
          {children}
        </main>
      </body>
    </html>
  );
}
