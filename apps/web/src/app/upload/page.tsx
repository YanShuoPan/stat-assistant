"use client";

import { useEffect, useRef, useState } from "react";
import { API, authHeaders } from "../lib/api";
import { useRequireAuth } from "../lib/auth";

const KNOWLEDGE_TYPES = [
  "definition",
  "assumption",
  "algorithm",
  "parameter",
  "result",
  "limitation",
  "implementation",
  "failure_mode",
  "comparison",
] as const;

const SOURCE_TYPES = ["paper", "code", "docstring", "note"] as const;
const CONFIDENCE_LEVELS = ["high", "medium", "low"] as const;

interface KnowledgeUnit {
  source_type: string;
  title: string;
  section: string | null;
  knowledge_type: string;
  topic_tags: string[];
  question_intent_tags: string[];
  content: string;
  evidence_span: string | null;
  dependencies: string[];
  limitations: string | null;
  confidence: string;
  reusable_for_questions: string[];
}

interface RecentUnit {
  id: number;
  title: string;
  knowledge_type: string;
  created_at: string;
}

export default function UploadPage() {
  const { user, checked } = useRequireAuth();
  const [step, setStep] = useState<"upload" | "review">("upload");

  const fileRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

  const [units, setUnits] = useState<KnowledgeUnit[]>([]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [recentUnits, setRecentUnits] = useState<RecentUnit[]>([]);

  if (!checked || !user) {
    return (
      <div className="flex items-center justify-center min-h-[60vh] text-zinc-400">
        Checking authentication...
      </div>
    );
  }

  const fetchRecent = async () => {
    try {
      const res = await fetch(`${API}/knowledge`, {
        headers: authHeaders(),
      });
      const data = await res.json();
      setRecentUnits(data.slice(0, 8));
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    fetchRecent();
  }, []);

  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const newFiles = Array.from(files);
    setSelectedFiles((prev) => [...prev, ...newFiles]);
  };

  const removeFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleFileUpload = async () => {
    if (selectedFiles.length === 0) return;

    setParsing(true);
    setParseError(null);
    setSuccess(null);

    const formData = new FormData();
    for (const file of selectedFiles) {
      formData.append("files", file);
    }

    try {
      const res = await fetch(`${API}/knowledge/parse`, {
        method: "POST",
        headers: authHeaders(),
        body: formData,
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${res.status}`);
      }
      const parsed = await res.json();
      setUnits(parsed.units || []);
      setExpandedIdx(null);
      setStep("review");
      if (fileRef.current) fileRef.current.value = "";
    } catch (err) {
      setParseError(err instanceof Error ? err.message : "Parse failed");
    } finally {
      setParsing(false);
    }
  };

  const updateUnit = (idx: number, patch: Partial<KnowledgeUnit>) => {
    setUnits((prev) =>
      prev.map((u, i) => (i === idx ? { ...u, ...patch } : u))
    );
  };

  const removeUnit = (idx: number) => {
    setUnits((prev) => prev.filter((_, i) => i !== idx));
    if (expandedIdx === idx) setExpandedIdx(null);
    else if (expandedIdx !== null && expandedIdx > idx)
      setExpandedIdx(expandedIdx - 1);
  };

  const handleConfirm = async () => {
    if (units.length === 0) return;
    setSaving(true);
    setSaveError(null);

    try {
      const res = await fetch(`${API}/knowledge/upload`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({ units }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${res.status}`);
      }
      setSuccess(`${units.length} knowledge units saved successfully!`);
      setStep("upload");
      setSelectedFiles([]);
      setUnits([]);
      if (fileRef.current) fileRef.current.value = "";
      fetchRecent();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 space-y-8">
      <h1 className="text-2xl font-bold">Upload Knowledge</h1>

      {success && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-3 text-sm text-green-800">
          {success}
        </div>
      )}

      {/* ===== Step 1: File Upload ===== */}
      {step === "upload" && (
        <div className="rounded-lg bg-white p-6 shadow-sm space-y-4">
          <p className="text-sm text-gray-600">
            Upload one or more documents (PDF, TXT, MD, PY, R, etc.).
            The AI will extract structured knowledge units for review.
          </p>

          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.py,.r,.rmd,.sas,.do,.jl"
            onChange={handleFilesSelected}
            className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
          />

          {selectedFiles.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-sm font-medium text-gray-700">
                Selected files ({selectedFiles.length}):
              </p>
              {selectedFiles.map((file, i) => (
                <div
                  key={`${file.name}-${i}`}
                  className="flex items-center justify-between rounded border border-gray-200 bg-gray-50 px-3 py-2"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-sm text-gray-800 truncate">
                      {file.name}
                    </span>
                    <span className="text-xs text-gray-400 shrink-0">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeFile(i)}
                    className="text-red-400 hover:text-red-600 text-sm ml-2 shrink-0"
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleFileUpload}
            disabled={parsing || selectedFiles.length === 0}
            className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
          >
            {parsing ? "AI is extracting knowledge..." : "Upload & Extract"}
          </button>

          {parseError && (
            <div className="rounded bg-red-50 p-3 text-sm text-red-700">
              {parseError}
            </div>
          )}
        </div>
      )}

      {/* ===== Step 2: Review Knowledge Units ===== */}
      {step === "review" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-800">
              Review Extracted Knowledge ({units.length} units)
            </h2>
            <button
              type="button"
              onClick={() => setStep("upload")}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              &larr; Back
            </button>
          </div>
          <p className="text-sm text-gray-500">
            AI extracted the following knowledge units. Click to expand, edit, or
            remove units before saving.
          </p>

          {units.map((unit, idx) => (
            <div
              key={idx}
              className="rounded-lg bg-white shadow-sm border border-gray-100 overflow-hidden"
            >
              {/* Collapsed header */}
              <button
                type="button"
                onClick={() =>
                  setExpandedIdx(expandedIdx === idx ? null : idx)
                }
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50"
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="rounded bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">
                    {unit.knowledge_type}
                  </span>
                  <span className="text-sm font-medium text-gray-900 truncate">
                    {unit.title}
                  </span>
                  {unit.section && (
                    <span className="text-xs text-gray-400 truncate">
                      / {unit.section}
                    </span>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs ${
                      unit.confidence === "high"
                        ? "bg-green-50 text-green-700"
                        : unit.confidence === "low"
                          ? "bg-red-50 text-red-600"
                          : "bg-yellow-50 text-yellow-700"
                    }`}
                  >
                    {unit.confidence}
                  </span>
                  <span className="text-gray-400 text-sm">
                    {expandedIdx === idx ? "\u25B2" : "\u25BC"}
                  </span>
                </div>
              </button>

              {/* Expanded editor */}
              {expandedIdx === idx && (
                <div className="border-t border-gray-100 px-4 py-4 space-y-3">
                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        Source Type
                      </label>
                      <select
                        value={unit.source_type}
                        onChange={(e) =>
                          updateUnit(idx, { source_type: e.target.value })
                        }
                        className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm bg-white"
                      >
                        {SOURCE_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        Knowledge Type
                      </label>
                      <select
                        value={unit.knowledge_type}
                        onChange={(e) =>
                          updateUnit(idx, { knowledge_type: e.target.value })
                        }
                        className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm bg-white"
                      >
                        {KNOWLEDGE_TYPES.map((t) => (
                          <option key={t} value={t}>
                            {t}
                          </option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs font-medium text-gray-500 mb-1">
                        Confidence
                      </label>
                      <select
                        value={unit.confidence}
                        onChange={(e) =>
                          updateUnit(idx, { confidence: e.target.value })
                        }
                        className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm bg-white"
                      >
                        {CONFIDENCE_LEVELS.map((c) => (
                          <option key={c} value={c}>
                            {c}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Title
                    </label>
                    <input
                      type="text"
                      value={unit.title}
                      onChange={(e) =>
                        updateUnit(idx, { title: e.target.value })
                      }
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Content
                    </label>
                    <textarea
                      value={unit.content}
                      onChange={(e) =>
                        updateUnit(idx, { content: e.target.value })
                      }
                      rows={4}
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Evidence Span
                    </label>
                    <textarea
                      value={unit.evidence_span || ""}
                      onChange={(e) =>
                        updateUnit(idx, {
                          evidence_span: e.target.value || null,
                        })
                      }
                      rows={2}
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Limitations
                    </label>
                    <input
                      type="text"
                      value={unit.limitations || ""}
                      onChange={(e) =>
                        updateUnit(idx, {
                          limitations: e.target.value || null,
                        })
                      }
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Topic Tags (comma-separated)
                    </label>
                    <input
                      type="text"
                      value={unit.topic_tags.join(", ")}
                      onChange={(e) =>
                        updateUnit(idx, {
                          topic_tags: e.target.value
                            .split(",")
                            .map((s) => s.trim())
                            .filter(Boolean),
                        })
                      }
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-500 mb-1">
                      Reusable For Questions (one per line)
                    </label>
                    <textarea
                      value={unit.reusable_for_questions.join("\n")}
                      onChange={(e) =>
                        updateUnit(idx, {
                          reusable_for_questions: e.target.value
                            .split("\n")
                            .filter(Boolean),
                        })
                      }
                      rows={3}
                      className="w-full rounded border border-gray-300 px-2 py-1.5 text-sm"
                    />
                  </div>

                  <div className="flex justify-end">
                    <button
                      type="button"
                      onClick={() => removeUnit(idx)}
                      className="text-sm text-red-500 hover:text-red-700"
                    >
                      Remove this unit
                    </button>
                  </div>
                </div>
              )}
            </div>
          ))}

          {units.length === 0 && (
            <p className="text-sm text-gray-400 text-center py-8">
              No knowledge units extracted. Try uploading different files.
            </p>
          )}

          {saveError && (
            <div className="rounded bg-red-50 border border-red-200 p-3 text-sm text-red-700">
              Save failed: {saveError}
            </div>
          )}

          {units.length > 0 && (
            <button
              onClick={handleConfirm}
              disabled={saving}
              className="rounded-lg bg-green-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-green-700 disabled:opacity-40"
            >
              {saving
                ? "Saving..."
                : `Confirm & Save ${units.length} Units`}
            </button>
          )}
        </div>
      )}

      {/* Recent uploads */}
      {recentUnits.length > 0 && (
        <div>
          <h2 className="text-lg font-semibold mb-3">Recently Uploaded</h2>
          <div className="space-y-2">
            {recentUnits.map((u) => (
              <div
                key={u.id}
                className="flex items-center justify-between rounded-lg bg-white p-3 shadow-sm border border-gray-100"
              >
                <div className="flex items-center gap-2">
                  <span className="rounded bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">
                    {u.knowledge_type}
                  </span>
                  <span className="font-medium text-sm">{u.title}</span>
                </div>
                <span className="text-xs text-gray-400">
                  {new Date(u.created_at).toLocaleDateString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
