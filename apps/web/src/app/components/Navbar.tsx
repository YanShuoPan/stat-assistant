"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { BarChart3 } from "lucide-react";
import { getUser } from "../lib/api";

export default function Navbar() {
  const router = useRouter();
  const [user, setUser] = useState<ReturnType<typeof getUser>>(null);

  const loadUser = () => setUser(getUser());

  useEffect(() => {
    loadUser();
    window.addEventListener("auth-change", loadUser);
    return () => window.removeEventListener("auth-change", loadUser);
  }, []);

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    setUser(null);
    window.dispatchEvent(new Event("auth-change"));
    router.push("/login");
  };

  const canUpload = user && (user.role === "admin" || user.role === "researcher");

  return (
    <nav className="border-b border-zinc-200 bg-white/80 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white">
            <BarChart3 className="w-5 h-5" />
          </div>
          <span className="font-bold text-lg tracking-tight">StatAssist</span>
        </Link>
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-6 text-sm font-medium text-zinc-500">
            <Link href="/chat" className="hover:text-zinc-900 transition-colors">Chat</Link>
            {canUpload && (
              <Link href="/upload" className="hover:text-zinc-900 transition-colors">Upload</Link>
            )}
          </div>
          <div className="flex items-center gap-3 text-sm">
            {user ? (
              <>
                <span className="text-zinc-600">
                  {user.username}
                  <span className="ml-1.5 rounded-full bg-indigo-50 px-2 py-0.5 text-xs font-medium text-indigo-700">
                    {user.role}
                  </span>
                </span>
                <button
                  onClick={logout}
                  className="text-zinc-400 hover:text-zinc-700 transition-colors"
                >
                  Logout
                </button>
              </>
            ) : (
              <Link
                href="/login"
                className="rounded-lg bg-indigo-600 px-4 py-2 text-white text-sm font-medium hover:bg-indigo-700 transition-colors"
              >
                Log In
              </Link>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}
