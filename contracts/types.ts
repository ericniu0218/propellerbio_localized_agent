// Shared TypeScript types for the Biotech Agent Pack frontend contract.
//
// This file is a hand-maintained 1:1 mirror of the Pydantic models in
// app/schemas/*.py, which are the source of truth. If you change a Pydantic
// model, update this file in the same commit (or regenerate via
// scripts/gen_ts_types.py). Do not hand-edit shapes that diverge from the
// Python models.

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

export type AgentRole = "director" | "specialist" | "runner";

export type AgentStatus =
  | "idle"
  | "thinking"
  | "running_tool"
  | "waiting_for_review"
  | "done"
  | "error";

export type TaskComplexity = "simple" | "normal" | "hard";

export type TaskDomain =
  | "literature"
  | "internal_docs"
  | "bioinformatics"
  | "chemistry"
  | "data_analysis"
  | "regulatory"
  | "clinical"
  | "lab_data"
  | "meetings"
  | "general";

export type TaskStatus =
  | "pending"
  | "planned"
  | "in_progress"
  | "waiting_for_review"
  | "waiting_for_approval"
  | "completed"
  | "failed"
  | "blocked_on_human"
  | "cancelled";

export type ApprovalDecision = "approve" | "reject";

export type EventType =
  | "session.started"
  | "task.planned"
  | "task.assigned"
  | "agent.started"
  | "agent.status"
  | "agent.token"
  | "tool.called"
  | "tool.result"
  | "review.approved"
  | "review.rejected"
  | "approval.required"
  | "approval.resolved"
  | "task.completed"
  | "task.failed"
  | "usage.updated";

// ---------------------------------------------------------------------------
// Core models
// ---------------------------------------------------------------------------

/** Everything the right-hand "running agents" panel needs for one card. */
export interface AgentCard {
  agent_id: string;
  name: string;
  role: AgentRole;
  model_name: string;
  status: AgentStatus;
  current_task_id: string | null;
  current_task_title: string | null;
  tokens_used: number;
  elapsed_ms: number;
}

export interface TaskNode {
  task_id: string;
  title: string;
  description: string;
  assigned_to: AgentRole;
  depends_on: string[];
  status: TaskStatus;
  complexity: TaskComplexity;
  domain: TaskDomain;
  is_high_stakes: boolean;
  retry_count: number;
  max_retries: number;
  result_ref: string | null;
  rejection_reason: string | null;
  /** Set by a Director rejection; guides the retry's drafting without
   * changing the query text handed to search tools. */
  fix_instruction: string | null;
  citations: string[];
}

export interface TaskPlan {
  session_id: string;
  root_request: string;
  nodes: TaskNode[];
  created_by: AgentRole;
}

// ---------------------------------------------------------------------------
// Event payloads (one per EventType; see Event envelope below)
// ---------------------------------------------------------------------------

export interface SessionStartedPayload {
  session_id: string;
  user_id: string;
}

export interface TaskPlannedPayload {
  plan: TaskPlan;
}

export interface TaskAssignedPayload {
  task_id: string;
  assigned_to: AgentRole;
  title: string;
}

export interface AgentStartedPayload {
  agent: AgentCard;
}

export interface AgentStatusPayload {
  agent_id: string;
  role: AgentRole;
  status: AgentStatus;
  task_id: string | null;
  task_title: string | null;
  tokens_used: number;
  elapsed_ms: number;
}

/** One streamed text chunk from an agent's model call. */
export interface AgentTokenPayload {
  agent_id: string;
  role: AgentRole;
  task_id: string | null;
  text: string;
  done: boolean;
}

export interface ToolCalledPayload {
  agent_id: string;
  role: AgentRole;
  task_id: string | null;
  tool_name: string;
  call_id: string;
  arguments: Record<string, unknown>;
}

export interface ToolResultPayload {
  agent_id: string;
  role: AgentRole;
  task_id: string | null;
  tool_name: string;
  call_id: string;
  summary: string;
  result_ref: string;
  citations: string[];
}

export interface ReviewDecisionPayload {
  task_id: string;
  reviewed_by: AgentRole;
  approved: boolean;
  reason: string | null;
  fix_instruction: string | null;
  retry_count: number;
}

export interface ApprovalResolvedPayload {
  task_id: string;
  decision: string;
  /** The person who signed off — recorded in the audit log. */
  approved_by: string;
  comment?: string | null;
  recorded_at: string;
}

export interface ApprovalRequiredPayload {
  task_id: string;
  title: string;
  reason: string;
  domain: TaskDomain;
  preview: string;
}

export interface TaskCompletedPayload {
  task_id: string;
  title: string;
  result_ref: string;
  citations: string[];
}

export interface TaskFailedPayload {
  task_id: string;
  title: string;
  error: string;
  blocked_on_human: boolean;
}

export interface UsageUpdatedPayload {
  session_id: string;
  agent_id: string;
  role: AgentRole;
  task_id: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens_for_task: number;
  budget_for_task: number;
}

/** Maps an EventType to its payload shape — use to type-narrow `Event.payload`. */
export interface EventPayloadMap {
  "session.started": SessionStartedPayload;
  "task.planned": TaskPlannedPayload;
  "task.assigned": TaskAssignedPayload;
  "agent.started": AgentStartedPayload;
  "agent.status": AgentStatusPayload;
  "agent.token": AgentTokenPayload;
  "tool.called": ToolCalledPayload;
  "tool.result": ToolResultPayload;
  "review.approved": ReviewDecisionPayload;
  "review.rejected": ReviewDecisionPayload;
  "approval.required": ApprovalRequiredPayload;
  "approval.resolved": ApprovalResolvedPayload;
  "task.completed": TaskCompletedPayload;
  "task.failed": TaskFailedPayload;
  "usage.updated": UsageUpdatedPayload;
}

/** Envelope for every event on the SSE stream (`GET /sessions/{id}/events`). */
export interface Event<T extends EventType = EventType> {
  event_id: string;
  type: T;
  session_id: string;
  timestamp: string; // ISO 8601
  payload: EventPayloadMap[T];
}

// ---------------------------------------------------------------------------
// REST request/response shapes
// ---------------------------------------------------------------------------

export interface StartSessionRequest {
  user_id: string;
}

export interface SessionResponse {
  session_id: string;
  user_id: string;
  created_at: string;
}

export interface SendMessageRequest {
  session_id: string;
  message: string;
  /** "runner" | "specialist" | "director" | null — null lets the Runner's
   * classifier decide (default). Never bypasses the high-stakes safety net. */
  force_tier: string | null;
}

/** One entry in GET /agent-tiers — for a tier picker / "Create Agent" UI. */
export interface AgentTierInfo {
  key: string; // "runner" | "specialist" | "director"
  name: string; // friendly tier name, e.g. "Deep Analysis"
  model_name: string;
  description: string;
}

export interface AgentTiersResponse {
  tiers: AgentTierInfo[];
}

export interface SendMessageResponse {
  session_id: string;
  task_id: string;
  accepted: boolean;
}

export interface CancelTaskRequest {
  task_id: string;
}

export interface ApprovalRequest {
  task_id: string;
  decision: ApprovalDecision;
  approved_by: string;
  comment: string | null;
}

export interface HistoryTurn {
  turn_id: string;
  role: string; // "user" | AgentRole
  content: string;
  created_at: string;
}

export interface HistoryResponse {
  session_id: string;
  turns: HistoryTurn[];
}

export interface AuditLogEntry {
  entry_id: string;
  session_id: string;
  task_id: string | null;
  user_id: string;
  agent: AgentRole | null;
  model_name: string | null;
  action: string;
  detail: Record<string, unknown>;
  created_at: string;
}

export interface AuditLogResponse {
  session_id: string;
  entries: AuditLogEntry[];
}

export interface AgentPanelResponse {
  session_id: string;
  agents: AgentCard[];
}

/** GET /results/{result_ref} — resolves a reference (from e.g. a
 * TaskCompletedPayload.result_ref) into the actual text to display. */
export interface ResultResponse {
  result_ref: string;
  content: string;
  content_type: string;
}
