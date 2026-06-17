"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import { API, authHeaders } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

interface Msg {
  role: "user" | "assistant";
  content: string;
  debug?: string | null;
}

interface SessionSummary {
  session_id: string;
  title: string;
  last_active: string;
  message_count: number;
}

function getSessionId(): string {
  let id = localStorage.getItem("chat_session_id");
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem("chat_session_id", id);
  }
  return id;
}

function setSessionId(id: string) {
  localStorage.setItem("chat_session_id", id);
}

export default function ChatPage() {
  const { user, checked } = useRequireAuth();
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionIdState] = useState<string>("");
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [expandedDebug, setExpandedDebug] = useState<Set<number>>(new Set());
  const [suggestions, setSuggestions] = useState<{question: string; method?: string}[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  const loadSessions = useCallback(async () => {
    try {
      const res = await fetch(`${API}/sessions`, { headers: authHeaders() });
      if (res.ok) setSessions(await res.json());
    } catch { /* ignore */ }
  }, []);

  const loadMessages = useCallback(async (sid: string) => {
    try {
      const res = await fetch(`${API}/sessions/${sid}/messages`, { headers: authHeaders() });
      if (res.ok) {
        const data = await res.json();
        setMessages(data.map((m: { role: string; content: string }) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })));
      }
    } catch { /* ignore */ }
  }, []);

  const loadSuggestions = useCallback(async () => {
    try {
      const res = await fetch(`${API}/suggested-questions`, { headers: authHeaders() });
      if (res.ok) {
        const data = await res.json();
        setSuggestions(data.questions.map((q: string | {question: string; method?: string}) =>
          typeof q === "string" ? { question: q } : q
        ));
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (!checked || !user) return;
    const sid = getSessionId();
    setSessionIdState(sid);
    loadSessions();
    loadMessages(sid);
    loadSuggestions();
  }, [checked, user, loadSessions, loadMessages, loadSuggestions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (!checked || !user) {
    return (
      <div className="flex items-center justify-center min-h-[60vh] text-zinc-400">
        Checking authentication...
      </div>
    );
  }

  const switchSession = (sid: string) => {
    setSessionId(sid);
    setSessionIdState(sid);
    setExpandedDebug(new Set());
    loadMessages(sid);
  };

  const newConversation = () => {
    const id = crypto.randomUUID();
    setSessionId(id);
    setSessionIdState(id);
    setMessages([]);
    setExpandedDebug(new Set());
  };

  const deleteSession = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await fetch(`${API}/sessions/${sid}`, { method: "DELETE", headers: authHeaders() });
      setSessions((prev) => prev.filter((s) => s.session_id !== sid));
      if (sid === sessionId) newConversation();
    } catch { /* ignore */ }
  };

  const toggleDebug = (index: number) => {
    setExpandedDebug((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  const sendMessage = async (text: string) => {
    if (!text.trim() || loading) return;
    const question = text.trim();

    setMessages((prev) => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    // Add placeholder assistant message for streaming
    setMessages((prev) => [...prev, { role: "assistant", content: "", debug: null }]);

    try {
      const res = await fetch(`${API}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-session-id": sessionId, ...authHeaders() },
        body: JSON.stringify({ message: question }),
      });

      if (!res.ok || !res.body) {
        // Fallback to non-streaming endpoint
        const fallbackRes = await fetch(`${API}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-session-id": sessionId, ...authHeaders() },
          body: JSON.stringify({ message: question }),
        });
        const data = await fallbackRes.json();
        if (data.session_id && data.session_id !== sessionId) {
          setSessionId(data.session_id);
          setSessionIdState(data.session_id);
        }
        setMessages((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = { role: "assistant", content: data.response, debug: data.debug };
          return updated;
        });
        loadSessions();
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let streamedContent = "";
      let debugText: string | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const jsonStr = line.slice(6);
            try {
              const parsed = JSON.parse(jsonStr);
              if (parsed.text !== undefined) {
                streamedContent += parsed.text;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: "assistant", content: streamedContent, debug: debugText };
                  return updated;
                });
              } else if (parsed.debug !== undefined) {
                debugText = parsed.debug;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: "assistant", content: streamedContent, debug: debugText };
                  return updated;
                });
              } else if (parsed.session_id !== undefined) {
                if (parsed.session_id !== sessionId) {
                  setSessionId(parsed.session_id);
                  setSessionIdState(parsed.session_id);
                }
                loadSessions();
              } else if (parsed.error !== undefined) {
                streamedContent += "\n\nError: " + parsed.error;
                setMessages((prev) => {
                  const updated = [...prev];
                  updated[updated.length - 1] = { role: "assistant", content: streamedContent, debug: debugText };
                  return updated;
                });
              }
            } catch { /* skip malformed JSON */ }
          }
        }
      }
    } catch {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = { role: "assistant", content: "Error: could not reach the API." };
        return updated;
      });
    } finally {
      setLoading(false);
    }
  };

  const send = () => sendMessage(input);
  const sendSuggestion = (question: string) => sendMessage(question);

  return (
    <div className="flex h-[calc(100vh-4rem)]">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? "w-72" : "w-0"} transition-all duration-200 overflow-hidden border-r border-zinc-200 bg-white flex flex-col`}>
        <div className="p-3 border-b border-zinc-100">
          <button onClick={newConversation} className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 transition-colors">
            + New Conversation
          </button>
        </div>
        <div className="flex-1 overflow-y-auto">
          {sessions.map((s) => (
            <div
              key={s.session_id}
              onClick={() => switchSession(s.session_id)}
              className={`group flex items-center gap-2 px-3 py-2.5 cursor-pointer text-sm border-b border-zinc-50 hover:bg-zinc-50 transition-colors ${
                s.session_id === sessionId ? "bg-indigo-50 text-indigo-700" : "text-zinc-600"
              }`}
            >
              <div className="flex-1 min-w-0">
                <p className="truncate font-medium">{s.title}</p>
                <p className="text-xs text-zinc-400 mt-0.5">{s.message_count} messages</p>
              </div>
              <button
                onClick={(e) => deleteSession(s.session_id, e)}
                className="opacity-0 group-hover:opacity-100 text-zinc-400 hover:text-red-500 transition-all text-xs px-1"
                title="Delete"
              >
                {"✕"}
              </button>
            </div>
          ))}
          {sessions.length === 0 && (
            <p className="text-center text-zinc-400 text-sm py-8">No conversations yet</p>
          )}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        <div className="flex items-center gap-2 px-4 py-2 border-b border-zinc-100 bg-white">
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="text-zinc-400 hover:text-zinc-600 transition-colors text-sm"
          >
            {sidebarOpen ? "\u25C0" : "\u25B6"} History
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="text-center pt-24">
                <p className="text-lg font-medium text-zinc-500">Ask a statistical question to get started</p>
                <p className="text-sm text-zinc-400 mt-2">or try one of these:</p>
                <div className="mt-6 flex flex-wrap justify-center gap-3 max-w-2xl mx-auto">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => sendSuggestion(s.question)}
                      className="text-left rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-600 hover:border-indigo-300 hover:bg-indigo-50 hover:text-indigo-700 transition-colors shadow-sm"
                    >
                      {s.method && <span className="text-xs font-medium text-indigo-500 block mb-1">{s.method}</span>}
                      {s.question}
                    </button>
                  ))}
                  {suggestions.length === 0 && (
                    <p className="text-sm text-zinc-400">e.g. &quot;I have high-dimensional data and want to select important variables&quot;</p>
                  )}
                </div>
              </div>
            )}
            {messages.filter((msg) => msg.role === "user" || msg.content).map((msg, i) => (
              msg.role === "user" ? (
                <div key={i} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-indigo-600 text-white">
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  </div>
                </div>
              ) : (
                <div key={i} className="w-full text-sm leading-relaxed text-zinc-800">
                  <div className="prose prose-sm max-w-none prose-headings:mb-2 prose-headings:mt-4 prose-p:my-1 prose-li:my-0.5 prose-ul:my-1">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex, rehypeRaw]}>{msg.content}</ReactMarkdown>
                  </div>
                  {msg.debug && (
                    <div className="mt-3 border-t border-zinc-100 pt-2">
                      <button onClick={() => toggleDebug(i)} className="text-xs text-zinc-400 hover:text-zinc-600 transition-colors">
                        {expandedDebug.has(i) ? "▼ Hide debug info" : "▶ Show debug info"}
                      </button>
                      {expandedDebug.has(i) && (
                        <pre className="mt-2 text-xs text-zinc-500 bg-zinc-50 rounded-lg p-3 overflow-x-auto whitespace-pre-wrap">{msg.debug}</pre>
                      )}
                    </div>
                  )}
                </div>
              )
            ))}
            {loading && messages.length > 0 && messages[messages.length - 1].role === "assistant" && !messages[messages.length - 1].content && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-white px-4 py-3 shadow-sm border border-zinc-200">
                  <span className="text-zinc-400 text-sm animate-pulse">Thinking...</span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-zinc-200 bg-white">
          <div className="mx-auto max-w-3xl px-4 py-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
                placeholder="Ask a statistical question..."
                className="flex-1 rounded-xl border border-zinc-300 px-4 py-2.5 text-sm focus:border-indigo-500 focus:outline-none"
              />
              <button
                onClick={send}
                disabled={loading || !input.trim()}
                className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-40 transition-colors"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
