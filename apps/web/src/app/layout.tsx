import type { Metadata } from "next";
import Navbar from "./components/Navbar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Statistical Research Assistant",
  description: "Chat with a statistical consultant and browse research methods",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body className="antialiased bg-zinc-50 text-zinc-900 font-sans selection:bg-indigo-100 selection:text-indigo-900">
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
