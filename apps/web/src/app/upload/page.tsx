"use client";

import { useEffect, useRef, useState, useCallback } from "react";
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

const AVAILABLE_DOMAINS = [
  "bayesian",
  "causal_inference",
  "clinical_trials",
  "functional_data",
  "high_dimensional",
  "inference_testing",
  "longitudinal",
  "machine_learning",
  "missing_data",
  "multiple_testing",
  "network_graph",
  "nonparametric",
  "optimal_transport",
  "probability_theory",
  "robust_distributed",
  "spatial",
  "survey_sampling",
  "survival_analysis",
  "time_series",
] as const;

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

interface PaperSection {
  section_type: string;
  section_index: number;
  summary: string;
  content: string;
  char_count: number;
}

interface RecentUnit {
  id: number;
  title: string;
  knowledge_type: string;
  created_at: string;
}

interface PaperRecord {
  id: number;
  title: string;
  authors: string | null;
  year: number | null;
  domain: string[];
  filename: string;
  file_size: number | null;
  created_at: string;
  ku_count: number;
}

export default function UploadPage() {
  const { user, checked } = useRequireAuth();
  const [activeTab, setActiveTab] = useState<"upload" | "papers">("upload");
  const [step, setStep] = useState<"upload" | "review">("upload");

  const fileRef = useRef<HTMLInputElement>(null);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [selectedDomains, setSelectedDomains] = useState<string[]>([]);
  const [parsing, setParsing] = useState(false);
  const [parseError, setParseError] = useState<string | null>(null);

  const [units, setUnits] = useState<KnowledgeUnit[]>([]);
  const [sections, setSections] = useState<PaperSection[]>([]);
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [recentUnits, setRecentUnits] = useState<RecentUnit[]>([]);

  const [papers, setPapers] = useState<PaperRecord[]>([]);
  const [papersLoading, setPapersLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);

  const fetchRecent = useCallback(async () => {
    try {
      const res = await fetch(`${API}/knowledge`, {
        headers: authHeaders(),
      });
      const data = await res.json();
      setRecentUnits(data.slice(0, 8));
    } catch {
      /* ignore */
    }
  }, []);

  const fetchPapers = useCallback(async () => {
    setPapersLoading(true);
    try {
      const res = await fetch(`${API}/knowledge/papers`, { headers: authHeaders() });
      if (res.ok) setPapers(await res.json());
    } catch { /* ignore */ }
    finally { setPapersLoading(false); }
  }, []);

  useEffect(() => {
    if (!checked || !user) return;
    fetchRecent();
  }, [checked, user, fetchRecent]);

  useEffect(() => {
    if (!checked || !user) return;
    if (activeTab === "papers") fetchPapers();
  }, [checked, user, activeTab, fetchPapers]);

  if (!checked || !user) {
    return (
      <div className="flex items-center justify-center min-h-[60vh] text-zinc-400">
        Checking authentication...
      </div>
    );
  }

  const handleFilesSelected = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    setSelectedFiles([files[0]]);
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
      const domainParam = selectedDomains.length > 0 ? `?domain=${selectedDomains[0]}` : "";
      const res = await fetch(`${API}/knowledge/parse${domainParam}`, {
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
      setSections(parsed.sections || []);
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
        body: JSON.stringify({
          units,
          sections,
          paper: {
            title: selectedFiles[0]?.name || "Untitled",
            domain: selectedDomains.length > 0 ? selectedDomains : ["statistics"],
            filename: selectedFiles[0]?.name || "unknown",
          },
        }),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${res.status}`);
      }

      const savedUnits = await res.json();
      // Get paper_id from saved units
      const paperId = savedUnits.find((u: any) => u.paper_id)?.paper_id;

      // Upload raw file(s) if paper was created
      if (paperId && selectedFiles.length > 0) {
        for (const file of selectedFiles) {
          const fileForm = new FormData();
          fileForm.append("file", file);
          try {
            await fetch(`${API}/knowledge/papers/${paperId}/file`, {
              method: "POST",
              headers: authHeaders(),
              body: fileForm,
            });
          } catch {
            // File upload failure is non-critical
          }
        }
      }

      setSuccess(`${units.length} knowledge units saved successfully!`);
      setStep("upload");
      setSelectedFiles([]);
      setSelectedDomains([]);
      setUnits([]);
      setSections([]);
      if (fileRef.current) fileRef.current.value = "";
      fetchRecent();
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  const deletePaper = async (paperId: number) => {
    if (!confirm("Delete this paper and all its knowledge units? This cannot be undone.")) return;
    setDeletingId(paperId);
    setDeleteError(null);
    try {
      const res = await fetch(`${API}/knowledge/papers/${paperId}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Server error: ${res.status}`);
      }
      setPapers((prev) => prev.filter((p) => p.id !== paperId));
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 space-y-8">
      <h1 className="text-2xl font-bold">Upload Knowledge</h1>

      {/* Tab bar */}
      <div className="flex border-b border-zinc-200">
        <button
          onClick={() => setActiveTab("upload")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "upload"
              ? "border-indigo-600 text-indigo-700"
              : "border-transparent text-zinc-500 hover:text-zinc-700"
          }`}
        >
          Upload
        </button>
        <button
          onClick={() => setActiveTab("papers")}
          className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
            activeTab === "papers"
              ? "border-indigo-600 text-indigo-700"
              : "border-transparent text-zinc-500 hover:text-zinc-700"
          }`}
        >
          Manage Papers
        </button>
      </div>

      {success && (
        <div className="rounded-lg bg-green-50 border border-green-200 p-3 text-sm text-green-800">
          {success}
        </div>
      )}

      {/* ===== Tab: Upload ===== */}
      {activeTab === "upload" && (
        <>
      {/* ===== Step 1: File Upload ===== */}
      {step === "upload" && (
        <div className="rounded-lg bg-white p-6 shadow-sm space-y-4">
          <p className="text-sm text-gray-600">
            Upload a document (PDF, TXT, MD, PY, R, etc.).
            The AI will extract structured knowledge units for review.
          </p>

          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt,.md,.py,.r,.rmd,.sas,.do,.jl"
            onChange={handleFilesSelected}
            className="text-sm file:mr-3 file:rounded-lg file:border-0 file:bg-blue-50 file:px-4 file:py-2 file:text-sm file:font-medium file:text-blue-700 hover:file:bg-blue-100"
          />

          {selectedFiles.length > 0 && (
            <div className="space-y-1.5">
              <p className="text-sm font-medium text-gray-700">
                Selected file:
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

          {/* Domain selection */}
          <div>
            <label className="block text-xs font-medium text-gray-500 mb-2">
              Domain (select one or more)
            </label>
            <div className="flex flex-wrap gap-1.5">
              {AVAILABLE_DOMAINS.map((d) => (
                <button
                  key={d}
                  type="button"
                  onClick={() =>
                    setSelectedDomains((prev) =>
                      prev.includes(d) ? prev.filter((x) => x !== d) : [...prev, d]
                    )
                  }
                  className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                    selectedDomains.includes(d)
                      ? "bg-indigo-100 text-indigo-700 ring-1 ring-indigo-300"
                      : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"
                  }`}
                >
                  {d.replace(/_/g, " ")}
                </button>
              ))}
            </div>
          </div>

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

          {/* Sections preview */}
          {sections.length > 0 && (
            <details className="rounded-lg bg-white shadow-sm border border-gray-100 overflow-hidden">
              <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700 hover:bg-gray-50">
                Detected Paper Sections ({sections.length})
              </summary>
              <div className="border-t border-gray-100 px-4 py-3 space-y-2">
                {sections.map((sec, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    <span className="rounded bg-purple-50 px-2 py-0.5 text-xs font-medium text-purple-700 shrink-0">
                      {sec.section_type}
                    </span>
                    <span className="text-gray-600">{sec.summary}</span>
                  </div>
                ))}
              </div>
            </details>
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
        </>
      )}

      {/* ===== Tab: Manage Papers ===== */}
      {activeTab === "papers" && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-zinc-500">Papers uploaded and linked to knowledge units.</p>
            <button onClick={fetchPapers} className="text-xs text-zinc-400 hover:text-zinc-600">Refresh</button>
          </div>
          {deleteError && (
            <div className="rounded bg-red-50 border border-red-200 p-3 text-sm text-red-700">{deleteError}</div>
          )}
          {papersLoading ? (
            <p className="text-sm text-zinc-400 text-center py-8">Loading...</p>
          ) : papers.length === 0 ? (
            <p className="text-sm text-zinc-400 text-center py-8">No papers found.</p>
          ) : (
            <div className="space-y-2">
              {papers.map((paper) => (
                <div key={paper.id} className="rounded-lg bg-white border border-zinc-100 shadow-sm px-4 py-3 flex items-center justify-between gap-4">
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-zinc-900 truncate">{paper.title}</p>
                    <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                      {paper.authors && <span className="text-xs text-zinc-500 truncate">{paper.authors}</span>}
                      {paper.year && <span className="text-xs text-zinc-400">{paper.year}</span>}
                      <span className="text-xs text-indigo-600 font-medium">{paper.ku_count} KU{paper.ku_count !== 1 ? "s" : ""}</span>
                      {Array.isArray(paper.domain) && paper.domain.map((d) => (
                        <span key={d} className="rounded-full bg-indigo-50 px-2 py-0.5 text-xs text-indigo-600">
                          {d.replace(/_/g, " ")}
                        </span>
                      ))}
                      <span className="text-xs text-zinc-300">{new Date(paper.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => deletePaper(paper.id)}
                    disabled={deletingId === paper.id}
                    className="shrink-0 rounded px-3 py-1.5 text-xs font-medium text-red-600 border border-red-200 hover:bg-red-50 disabled:opacity-40 transition-colors"
                  >
                    {deletingId === paper.id ? "Deleting..." : "Delete"}
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
