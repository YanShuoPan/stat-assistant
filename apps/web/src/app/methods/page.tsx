"use client";

import { useEffect, useState, useCallback } from "react";
import { API, authHeaders } from "../lib/api";

/* ---------- Types ---------- */

interface TaxonomyNode {
  id: number;
  name: string;
  node_type: "problem_category" | "method_family" | "method" | "variant";
  description: string | null;
  auto_generated: boolean;
  children_count: number;
  unit_count: number;
  children: TaxonomyNode[];
}

interface NodeDetail {
  id: number;
  name: string;
  node_type: string;
  description: string | null;
  aliases: string[];
  auto_generated: boolean;
  parent: TaxonomyNode | null;
  children: TaxonomyNode[];
  siblings: TaxonomyNode[];
  units_by_type: Record<string, number>;
  units: KnowledgeUnit[];
  created_at: string;
  updated_at: string;
}

interface KnowledgeUnit {
  id: number;
  source_type: string;
  title: string;
  section: string | null;
  knowledge_type: string;
  content: string;
  evidence_span: string | null;
  limitations: string | null;
  confidence: string;
  method_name: string | null;
  field: string | null;
}

/* ---------- Helpers ---------- */

const NODE_TYPE_LABELS: Record<string, string> = {
  problem_category: "Problem Category",
  method_family: "Method Family",
  method: "Method",
  variant: "Variant",
};

const NODE_TYPE_COLORS: Record<string, string> = {
  problem_category: "bg-indigo-100 text-indigo-700",
  method_family: "bg-blue-100 text-blue-700",
  method: "bg-emerald-100 text-emerald-700",
  variant: "bg-amber-100 text-amber-700",
};

const CONFIDENCE_COLORS: Record<string, string> = {
  high: "bg-green-50 text-green-700",
  medium: "bg-yellow-50 text-yellow-700",
  low: "bg-red-50 text-red-600",
};

/** Recursively check if a node or any descendant matches the search query. */
function nodeMatchesSearch(node: TaxonomyNode, query: string): boolean {
  const q = query.toLowerCase();
  if (node.name.toLowerCase().includes(q)) return true;
  if (node.children.some((c) => nodeMatchesSearch(c, q))) return true;
  return false;
}

/* ---------- TreeNode Component ---------- */

function TreeNode({
  node,
  selectedId,
  onSelect,
  depth = 0,
  searchQuery = "",
}: {
  node: TaxonomyNode;
  selectedId: number | null;
  onSelect: (id: number) => void;
  depth?: number;
  searchQuery?: string;
}) {
  const [expanded, setExpanded] = useState(depth < 1);
  const hasChildren = node.children_count > 0 || node.children.length > 0;
  const isSelected = selectedId === node.id;

  // Auto-expand when search is active and this subtree matches
  useEffect(() => {
    if (searchQuery && hasChildren && node.children.some((c) => nodeMatchesSearch(c, searchQuery))) {
      setExpanded(true);
    }
  }, [searchQuery, hasChildren, node.children]);

  // If search is active and this node doesn't match, hide it
  if (searchQuery && !nodeMatchesSearch(node, searchQuery)) {
    return null;
  }

  // Styling varies by node_type
  let textClass = "text-sm text-zinc-700";
  if (node.node_type === "problem_category") {
    textClass = "text-sm font-bold text-zinc-900";
  } else if (node.node_type === "method_family") {
    textClass = "text-sm font-semibold text-zinc-800";
  } else if (node.node_type === "variant") {
    textClass = "text-xs text-zinc-600";
  }

  return (
    <div>
      <div
        className={`flex items-center gap-1 py-1 pr-2 rounded-md cursor-pointer transition-colors group ${
          isSelected ? "bg-indigo-50 text-indigo-700" : "hover:bg-zinc-100"
        }`}
        style={{ paddingLeft: `${depth * 16 + 8}px` }}
      >
        {/* Expand / collapse toggle */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            if (hasChildren) setExpanded(!expanded);
          }}
          className={`w-5 h-5 flex items-center justify-center shrink-0 text-zinc-400 hover:text-zinc-600 transition-colors ${
            !hasChildren ? "invisible" : ""
          }`}
          aria-label={expanded ? "Collapse" : "Expand"}
        >
          {expanded ? "\u25BC" : "\u25B6"}
        </button>

        {/* Node name (click to select) */}
        <button
          onClick={() => onSelect(node.id)}
          className={`flex-1 text-left truncate ${textClass} ${
            isSelected ? "!text-indigo-700 !font-semibold" : ""
          }`}
          title={node.name}
        >
          {node.name}
        </button>

        {/* Unit count badge */}
        {node.unit_count > 0 && (
          <span className="shrink-0 rounded-full bg-zinc-200 px-1.5 py-0.5 text-[10px] font-medium text-zinc-600 leading-none">
            {node.unit_count}
          </span>
        )}
      </div>

      {/* Children */}
      {expanded && hasChildren && (
        <div>
          {node.children.map((child) => (
            <TreeNode
              key={child.id}
              node={child}
              selectedId={selectedId}
              onSelect={onSelect}
              depth={depth + 1}
              searchQuery={searchQuery}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/* ---------- Detail Panel ---------- */

function DetailPanel({
  detail,
  loading,
  error,
  onSelect,
}: {
  detail: NodeDetail | null;
  loading: boolean;
  error: string | null;
  onSelect: (id: number) => void;
}) {
  const [expandedUnits, setExpandedUnits] = useState<Set<number>>(new Set());

  // Reset expanded units when detail changes
  useEffect(() => {
    setExpandedUnits(new Set());
  }, [detail?.id]);

  const toggleUnit = (id: number) => {
    setExpandedUnits((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-400">
        <div className="text-center">
          <div className="inline-block w-6 h-6 border-2 border-zinc-300 border-t-indigo-500 rounded-full animate-spin mb-3" />
          <p className="text-sm">Loading details...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-500">
        <p className="text-sm">{error}</p>
      </div>
    );
  }

  if (!detail) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-400">
        <div className="text-center">
          <p className="text-lg font-medium mb-1">Select a method</p>
          <p className="text-sm">Choose a node from the tree to view its details</p>
        </div>
      </div>
    );
  }

  // Build breadcrumb from parent chain
  const breadcrumb: { id: number; name: string }[] = [];
  if (detail.parent) {
    // The API gives us the immediate parent; we show what we have
    breadcrumb.push({ id: detail.parent.id, name: detail.parent.name });
  }
  breadcrumb.push({ id: detail.id, name: detail.name });

  // Group units by knowledge_type
  const unitsByType: Record<string, KnowledgeUnit[]> = {};
  for (const u of detail.units) {
    if (!unitsByType[u.knowledge_type]) unitsByType[u.knowledge_type] = [];
    unitsByType[u.knowledge_type].push(u);
  }
  const sortedTypes = Object.keys(unitsByType).sort();

  return (
    <div className="overflow-y-auto h-full p-5 space-y-6">
      {/* Header */}
      <div>
        <div className="flex flex-wrap items-center gap-2 mb-1">
          <h2 className="text-xl font-bold text-zinc-900">{detail.name}</h2>
          <span
            className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${
              NODE_TYPE_COLORS[detail.node_type] || "bg-zinc-100 text-zinc-600"
            }`}
          >
            {NODE_TYPE_LABELS[detail.node_type] || detail.node_type}
          </span>
          {detail.auto_generated && (
            <span className="rounded-full bg-zinc-100 px-2 py-0.5 text-[10px] font-medium text-zinc-500">
              auto-generated
            </span>
          )}
        </div>

        {/* Breadcrumb */}
        {breadcrumb.length > 1 && (
          <nav className="flex items-center gap-1 text-sm text-zinc-500 mt-2">
            {breadcrumb.map((crumb, i) => (
              <span key={crumb.id} className="flex items-center gap-1">
                {i > 0 && <span className="text-zinc-300">/</span>}
                {i < breadcrumb.length - 1 ? (
                  <button
                    onClick={() => onSelect(crumb.id)}
                    className="text-indigo-600 hover:text-indigo-800 hover:underline transition-colors"
                  >
                    {crumb.name}
                  </button>
                ) : (
                  <span className="text-zinc-700 font-medium">{crumb.name}</span>
                )}
              </span>
            ))}
          </nav>
        )}
      </div>

      {/* Description */}
      {detail.description && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Description
          </h3>
          <p className="text-sm text-zinc-700 leading-relaxed">{detail.description}</p>
        </div>
      )}

      {/* Aliases */}
      {detail.aliases.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-1">
            Aliases
          </h3>
          <p className="text-sm text-zinc-600">{detail.aliases.join(", ")}</p>
        </div>
      )}

      {/* Children */}
      {detail.children.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
            Children ({detail.children.length})
          </h3>
          <div className="space-y-1">
            {detail.children.map((child) => (
              <button
                key={child.id}
                onClick={() => onSelect(child.id)}
                className="block w-full text-left rounded-lg border border-zinc-100 bg-white px-3 py-2 text-sm text-zinc-700 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-700 transition-colors"
              >
                <span className="font-medium">{child.name}</span>
                {child.unit_count > 0 && (
                  <span className="ml-2 text-xs text-zinc-400">
                    {child.unit_count} units
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Siblings */}
      {detail.siblings.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
            Siblings
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {detail.siblings.map((sib) => (
              <button
                key={sib.id}
                onClick={() => onSelect(sib.id)}
                className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                  sib.id === detail.id
                    ? "bg-indigo-100 text-indigo-700"
                    : "bg-zinc-100 text-zinc-600 hover:bg-indigo-50 hover:text-indigo-600"
                }`}
              >
                {sib.name}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Knowledge Units by Type (summary cards) */}
      {Object.keys(detail.units_by_type).length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
            Knowledge by Type
          </h3>
          <div className="flex flex-wrap gap-2">
            {Object.entries(detail.units_by_type)
              .sort(([, a], [, b]) => b - a)
              .map(([type, count]) => (
                <div
                  key={type}
                  className="rounded-lg border border-zinc-100 bg-white px-3 py-2 text-center min-w-[80px]"
                >
                  <p className="text-lg font-bold text-zinc-800">{count}</p>
                  <p className="text-[10px] font-medium text-zinc-500 uppercase tracking-wide">
                    {type}
                  </p>
                </div>
              ))}
          </div>
        </div>
      )}

      {/* Knowledge Units List */}
      {detail.units.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-400 mb-2">
            Knowledge Units ({detail.units.length})
          </h3>
          <div className="space-y-2">
            {sortedTypes.map((type) => (
              <div key={type}>
                <p className="text-xs font-semibold text-zinc-500 mb-1 mt-3 first:mt-0">
                  {type}
                </p>
                {unitsByType[type].map((unit) => {
                  const isExpanded = expandedUnits.has(unit.id);
                  return (
                    <div
                      key={unit.id}
                      className="rounded-lg border border-zinc-100 bg-white overflow-hidden mb-1.5"
                    >
                      <button
                        onClick={() => toggleUnit(unit.id)}
                        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-zinc-50 transition-colors"
                      >
                        <span className="text-zinc-400 text-xs shrink-0">
                          {isExpanded ? "\u25BC" : "\u25B6"}
                        </span>
                        <span
                          className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] font-medium ${
                            NODE_TYPE_COLORS[unit.knowledge_type] ||
                            "bg-zinc-100 text-zinc-600"
                          }`}
                        >
                          {unit.knowledge_type}
                        </span>
                        <span className="text-sm text-zinc-700 truncate flex-1">
                          {unit.content.length > 120
                            ? unit.content.slice(0, 120) + "..."
                            : unit.content}
                        </span>
                        <span
                          className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium ${
                            CONFIDENCE_COLORS[unit.confidence] ||
                            "bg-zinc-100 text-zinc-600"
                          }`}
                        >
                          {unit.confidence}
                        </span>
                      </button>

                      {isExpanded && (
                        <div className="px-4 pb-3 pt-1 space-y-3 text-sm border-t border-zinc-50">
                          <div>
                            <h4 className="font-semibold text-zinc-800 mb-1 text-xs uppercase tracking-wide">
                              Content
                            </h4>
                            <p className="text-zinc-700 whitespace-pre-wrap text-sm leading-relaxed">
                              {unit.content}
                            </p>
                          </div>

                          {unit.evidence_span && (
                            <div>
                              <h4 className="font-semibold text-zinc-800 mb-1 text-xs uppercase tracking-wide">
                                Evidence
                              </h4>
                              <p className="text-zinc-600 text-xs bg-zinc-50 rounded-lg p-2.5 whitespace-pre-wrap">
                                {unit.evidence_span}
                              </p>
                            </div>
                          )}

                          {unit.limitations && (
                            <div>
                              <h4 className="font-semibold text-zinc-800 mb-1 text-xs uppercase tracking-wide">
                                Limitations
                              </h4>
                              <p className="text-zinc-600 text-sm">
                                {unit.limitations}
                              </p>
                            </div>
                          )}

                          <div className="flex items-center gap-3 pt-1 text-xs text-zinc-400">
                            <span>{unit.source_type}</span>
                            {unit.title && (
                              <>
                                <span>&middot;</span>
                                <span>{unit.title}</span>
                              </>
                            )}
                            {unit.method_name && (
                              <>
                                <span>&middot;</span>
                                <span>{unit.method_name}</span>
                              </>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Ask about this method button */}
      <div className="pt-2">
        <a
          href={`/chat?method=${encodeURIComponent(detail.name)}`}
          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-indigo-700 transition-colors"
        >
          Ask about this method
          <span className="text-indigo-200">&rarr;</span>
        </a>
      </div>
    </div>
  );
}

/* ---------- Main Page ---------- */

export default function TaxonomyBrowserPage() {
  const [tree, setTree] = useState<TaxonomyNode[]>([]);
  const [treeLoading, setTreeLoading] = useState(true);
  const [treeError, setTreeError] = useState<string | null>(null);

  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<NodeDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const [searchQuery, setSearchQuery] = useState("");

  // Fetch taxonomy tree on mount
  useEffect(() => {
    setTreeLoading(true);
    setTreeError(null);
    fetch(`${API}/taxonomy`, { headers: authHeaders() })
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load taxonomy (${r.status})`);
        return r.json();
      })
      .then((data) => {
        setTree(Array.isArray(data.nodes) ? data.nodes : []);
      })
      .catch((err) => {
        setTreeError(err.message || "Failed to load taxonomy");
        setTree([]);
      })
      .finally(() => setTreeLoading(false));
  }, []);

  // Fetch detail when a node is selected
  const selectNode = useCallback((id: number) => {
    setSelectedId(id);
    setDetailLoading(true);
    setDetailError(null);
    fetch(`${API}/taxonomy/${id}`, { headers: authHeaders() })
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load node (${r.status})`);
        return r.json();
      })
      .then((data) => setDetail(data))
      .catch((err) => {
        setDetailError(err.message || "Failed to load node details");
        setDetail(null);
      })
      .finally(() => setDetailLoading(false));
  }, []);

  // Tree loading state
  if (treeLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-zinc-400">
        <div className="text-center">
          <div className="inline-block w-6 h-6 border-2 border-zinc-300 border-t-indigo-500 rounded-full animate-spin mb-3" />
          <p className="text-sm">Loading taxonomy...</p>
        </div>
      </div>
    );
  }

  // Tree error state
  if (treeError) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 text-sm mb-2">{treeError}</p>
          <button
            onClick={() => window.location.reload()}
            className="text-sm text-indigo-600 hover:text-indigo-800 hover:underline"
          >
            Try again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6">
      {/* Page header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <h1 className="text-2xl font-bold text-zinc-900">Method Taxonomy</h1>
        <div className="relative w-full sm:w-72">
          <input
            type="text"
            placeholder="Search methods..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full rounded-lg border border-zinc-300 bg-white pl-9 pr-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-colors"
          />
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600 text-sm"
              aria-label="Clear search"
            >
              &times;
            </button>
          )}
        </div>
      </div>

      {/* Empty state */}
      {tree.length === 0 ? (
        <div className="rounded-xl bg-white shadow-sm border border-zinc-100 p-16 text-center">
          <p className="text-zinc-500">
            No methods classified yet. Upload papers to automatically build the
            method taxonomy.
          </p>
        </div>
      ) : (
        /* Two-column layout */
        <div className="flex flex-col md:flex-row gap-5 min-h-[70vh]">
          {/* Tree navigation (left panel) */}
          <div className="w-full md:w-72 shrink-0 rounded-xl bg-white shadow-sm border border-zinc-100 overflow-hidden flex flex-col">
            <div className="flex-1 overflow-y-auto py-2">
              {tree.map((node) => (
                <TreeNode
                  key={node.id}
                  node={node}
                  selectedId={selectedId}
                  onSelect={selectNode}
                  depth={0}
                  searchQuery={searchQuery}
                />
              ))}
              {/* If search yields nothing */}
              {searchQuery &&
                tree.every((n) => !nodeMatchesSearch(n, searchQuery)) && (
                  <p className="text-center text-zinc-400 text-sm py-8 px-4">
                    No methods matching &ldquo;{searchQuery}&rdquo;
                  </p>
                )}
            </div>
          </div>

          {/* Detail panel (right) */}
          <div className="flex-1 rounded-xl bg-white shadow-sm border border-zinc-100 overflow-hidden min-h-[50vh]">
            <DetailPanel
              detail={detail}
              loading={detailLoading}
              error={detailError}
              onSelect={selectNode}
            />
          </div>
        </div>
      )}
    </div>
  );
}
