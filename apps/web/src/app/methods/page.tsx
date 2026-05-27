"use client";

import { useEffect, useState } from "react";
import { API, authHeaders, getUser } from "../lib/api";

interface KnowledgeUnit {
  id: number;
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
  uploaded_by: number | null;
  created_at: string;
}

type GroupedUnits = Record<string, KnowledgeUnit[]>;

export default function KnowledgeLibraryPage() {
  const [units, setUnits] = useState<KnowledgeUnit[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedTitle, setExpandedTitle] = useState<string | null>(null);
  const [expandedUnit, setExpandedUnit] = useState<number | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<KnowledgeUnit | null>(
    null
  );

  const fetchUnits = () => {
    fetch(`${API}/api/knowledge`, { headers: authHeaders() })
      .then((r) => {
        if (!r.ok) throw new Error(r.statusText);
        return r.json();
      })
      .then((data) => setUnits(Array.isArray(data) ? data : []))
      .catch(() => setUnits([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchUnits();
  }, []);

  const handleDelete = async (unit: KnowledgeUnit) => {
    try {
      const res = await fetch(`${API}/api/knowledge/${unit.id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!res.ok) throw new Error("Delete failed");
      setConfirmDelete(null);
      setExpandedUnit(null);
      fetchUnits();
    } catch {
      /* ignore */
    }
  };

  // Group units by title
  const grouped: GroupedUnits = {};
  for (const u of units) {
    if (!grouped[u.title]) grouped[u.title] = [];
    grouped[u.title].push(u);
  }
  const titles = Object.keys(grouped);

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center text-gray-400">
        Loading...
      </div>
    );
  }

  const user = getUser();

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="text-2xl font-bold mb-6">Knowledge Library</h1>

      {titles.length === 0 ? (
        <p className="text-gray-500 text-center py-16">
          No knowledge uploaded yet. Go to{" "}
          <a href="/upload" className="text-blue-600 hover:underline">
            Upload
          </a>{" "}
          to add some.
        </p>
      ) : (
        <div className="space-y-3">
          {titles.map((title) => {
            const group = grouped[title];
            const isExpanded = expandedTitle === title;
            const typeCount: Record<string, number> = {};
            for (const u of group) {
              typeCount[u.knowledge_type] =
                (typeCount[u.knowledge_type] || 0) + 1;
            }

            return (
              <div
                key={title}
                className="rounded-xl bg-white shadow-sm border border-gray-100 overflow-hidden"
              >
                {/* Title header */}
                <button
                  onClick={() =>
                    setExpandedTitle(isExpanded ? null : title)
                  }
                  className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-gray-50 transition-colors"
                >
                  <div>
                    <h3 className="font-semibold text-gray-900">{title}</h3>
                    <div className="flex flex-wrap gap-1.5 mt-1">
                      {Object.entries(typeCount).map(([type, count]) => (
                        <span
                          key={type}
                          className="rounded-full bg-blue-50 px-2 py-0.5 text-xs text-blue-700"
                        >
                          {type}
                          {count > 1 ? ` (${count})` : ""}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0">
                    <span className="text-sm text-gray-400">
                      {group.length} units
                    </span>
                    <span className="text-gray-400">
                      {isExpanded ? "\u25B2" : "\u25BC"}
                    </span>
                  </div>
                </button>

                {/* Expanded: list of units */}
                {isExpanded && (
                  <div className="border-t border-gray-100">
                    {group.map((unit) => {
                      const isUnitExpanded = expandedUnit === unit.id;
                      const canDelete =
                        user?.role === "admin" ||
                        (user?.role === "researcher" &&
                          unit.uploaded_by === user?.id);

                      return (
                        <div
                          key={unit.id}
                          className="border-b border-gray-50 last:border-b-0"
                        >
                          <button
                            onClick={() =>
                              setExpandedUnit(
                                isUnitExpanded ? null : unit.id
                              )
                            }
                            className="w-full flex items-center justify-between px-5 py-3 text-left hover:bg-gray-50"
                          >
                            <div className="flex items-center gap-2 min-w-0">
                              <span className="rounded bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-600">
                                {unit.knowledge_type}
                              </span>
                              <span className="text-sm text-gray-700 truncate">
                                {unit.content.length > 100
                                  ? unit.content.slice(0, 100) + "..."
                                  : unit.content}
                              </span>
                            </div>
                            <span
                              className={`shrink-0 ml-2 rounded-full px-2 py-0.5 text-xs ${
                                unit.confidence === "high"
                                  ? "bg-green-50 text-green-700"
                                  : unit.confidence === "low"
                                    ? "bg-red-50 text-red-600"
                                    : "bg-yellow-50 text-yellow-700"
                              }`}
                            >
                              {unit.confidence}
                            </span>
                          </button>

                          {isUnitExpanded && (
                            <div className="px-5 pb-4 space-y-3 text-sm">
                              <div>
                                <h4 className="font-semibold text-gray-800 mb-1">
                                  Content
                                </h4>
                                <p className="text-gray-700 whitespace-pre-wrap">
                                  {unit.content}
                                </p>
                              </div>

                              {unit.evidence_span && (
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-1">
                                    Evidence
                                  </h4>
                                  <p className="text-gray-600 text-xs bg-gray-50 rounded p-2 whitespace-pre-wrap">
                                    {unit.evidence_span}
                                  </p>
                                </div>
                              )}

                              {unit.limitations && (
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-1">
                                    Limitations
                                  </h4>
                                  <p className="text-gray-600">
                                    {unit.limitations}
                                  </p>
                                </div>
                              )}

                              {unit.topic_tags.length > 0 && (
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-1">
                                    Topic Tags
                                  </h4>
                                  <div className="flex flex-wrap gap-1.5">
                                    {unit.topic_tags.map((tag, i) => (
                                      <span
                                        key={i}
                                        className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs text-blue-700"
                                      >
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}

                              {unit.reusable_for_questions.length > 0 && (
                                <div>
                                  <h4 className="font-semibold text-gray-800 mb-1">
                                    Reusable For Questions
                                  </h4>
                                  <ul className="space-y-0.5 text-gray-600">
                                    {unit.reusable_for_questions.map(
                                      (q, i) => (
                                        <li key={i} className="flex gap-2">
                                          <span className="text-gray-400">
                                            ?
                                          </span>
                                          {q}
                                        </li>
                                      )
                                    )}
                                  </ul>
                                </div>
                              )}

                              {unit.dependencies &&
                                unit.dependencies.length > 0 && (
                                  <div>
                                    <h4 className="font-semibold text-gray-800 mb-1">
                                      Dependencies
                                    </h4>
                                    <p className="text-gray-600">
                                      {unit.dependencies.join(", ")}
                                    </p>
                                  </div>
                                )}

                              <div className="flex items-center justify-between pt-2">
                                <span className="text-xs text-gray-400">
                                  {unit.source_type} &middot;{" "}
                                  {new Date(
                                    unit.created_at
                                  ).toLocaleDateString()}
                                </span>
                                {canDelete && (
                                  <button
                                    onClick={() => setConfirmDelete(unit)}
                                    className="text-sm text-red-500 hover:text-red-700"
                                  >
                                    Delete
                                  </button>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Confirm delete dialog */}
      {confirmDelete && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow-xl">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Confirm Delete
            </h3>
            <p className="text-sm text-gray-600 mb-5">
              Delete this knowledge unit from{" "}
              <strong>{confirmDelete.title}</strong>? This cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setConfirmDelete(null)}
                className="rounded-lg border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(confirmDelete)}
                className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
