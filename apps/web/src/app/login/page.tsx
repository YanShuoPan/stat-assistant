"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { API } from "../lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const res = await fetch(`${API}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });

      if (!res.ok) {
        const data = await res.json();
        throw new Error(data.detail || "Login failed");
      }

      const { access_token } = await res.json();
      localStorage.setItem("token", access_token);

      // Fetch user info
      const meRes = await fetch(`${API}/auth/me`, {
        headers: { Authorization: `Bearer ${access_token}` },
      });
      if (meRes.ok) {
        const user = await meRes.json();
        localStorage.setItem("user", JSON.stringify(user));
      }

      window.dispatchEvent(new Event("auth-change"));
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
      <div className="w-full max-w-sm">
        <h1 className="text-2xl font-bold text-center mb-6">Log In</h1>

        <form onSubmit={handleLogin} className="space-y-4 rounded-xl bg-white p-6 shadow-sm border border-gray-100">
          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full rounded-lg border border-gray-300 px-3 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40"
          >
            {loading ? "Logging in..." : "Log In"}
          </button>
          <p className="text-center text-sm text-gray-500">
            Don&apos;t have an account?{" "}
            <a href="/register" className="text-indigo-600 hover:underline">
              Register
            </a>
          </p>
        </form>
      </div>
    </div>
  );
}
