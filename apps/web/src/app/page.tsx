"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { motion } from "motion/react";
import {
  MessageSquare,
  BookOpen,
  Database,
  ArrowRight,
  Search,
} from "lucide-react";
import { API } from "./lib/api";

export default function Home() {
  const [health, setHealth] = useState<string>("checking...");

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then((d) => setHealth(d.status))
      .catch(() => setHealth("unreachable"));
  }, []);

  const features = [
    {
      icon: <MessageSquare className="w-5 h-5" />,
      title: "AI Chat Assistant",
      description:
        "Get instant answers to complex statistical questions and methodology queries.",
    },
    {
      icon: <BookOpen className="w-5 h-5" />,
      title: "Methodology Library",
      description:
        "Browse a comprehensive database of research methods and statistical tests.",
    },
    {
      icon: <Database className="w-5 h-5" />,
      title: "Data Guidance",
      description:
        "Step-by-step instructions on how to prepare and analyze your research data.",
    },
  ];

  return (
    <div className="min-h-[calc(100vh-4rem)] overflow-x-hidden">
      {/* Background Grid Pattern */}
      <div
        className="fixed inset-0 z-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: "radial-gradient(#000 1px, transparent 1px)",
          backgroundSize: "24px 24px",
        }}
      />

      <main className="relative z-10">
        {/* Hero Section */}
        <section className="pt-24 pb-16 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-50 border border-indigo-100 text-indigo-700 text-xs font-semibold mb-6">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500"></span>
              </span>
              v2.0 Now Live with Advanced Modeling
            </div>

            <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-zinc-900 mb-6 max-w-4xl mx-auto leading-[1.1]">
              Your AI Partner for{" "}
              <span className="text-indigo-600">Statistical Research</span>
            </h1>

            <p className="text-lg md:text-xl text-zinc-500 max-w-2xl mx-auto mb-10 leading-relaxed">
              Accelerate your research with expert statistical guidance. Ask
              questions, browse methods, and master your data analysis workflow.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16">
              <Link
                href="/chat"
                className="w-full sm:w-auto bg-indigo-600 text-white px-8 py-4 rounded-xl font-semibold hover:bg-indigo-700 transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-200 group"
              >
                Start Research Chat
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link
                href="/methods"
                className="w-full sm:w-auto bg-white border border-zinc-200 text-zinc-900 px-8 py-4 rounded-xl font-semibold hover:bg-zinc-50 transition-all flex items-center justify-center gap-2"
              >
                Browse Methods
                <Search className="w-4 h-4 text-zinc-400" />
              </Link>
            </div>
          </motion.div>
        </section>

        {/* Features Section */}
        <section className="pb-24 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
          <div className="grid gap-6 sm:grid-cols-3">
            {features.map((feature, i) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 * (i + 1) }}
                className="rounded-2xl bg-white border border-zinc-200 p-6 hover:shadow-md hover:border-indigo-200 transition-all"
              >
                <div className="w-10 h-10 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 mb-4">
                  {feature.icon}
                </div>
                <h3 className="font-semibold text-zinc-900 mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-zinc-500 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </section>
      </main>

      {/* Status Badge */}
      <div className="fixed bottom-6 left-1/2 -translate-x-1/2 z-20">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-zinc-200 shadow-sm text-xs text-zinc-500">
          <div
            className={`w-1.5 h-1.5 rounded-full ${
              health === "ok" ? "bg-emerald-500" : "bg-red-400"
            }`}
          />
          <span className="font-mono uppercase tracking-wider">
            API Status: {health}
          </span>
        </div>
      </div>
    </div>
  );
}
