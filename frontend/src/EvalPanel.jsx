import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch, detectionImageUrl } from './api.js'

// Agreement moves only when uploads arrive, so this polls slower than the
// detection grid.
const EVAL_POLL_INTERVAL_MS = 60_000
const DISAGREEMENT_LIMIT = 24

const FOMO_COLOR = '#f2a65a'
const YOLO_COLOR = '#6fd08c'

// Every model source gets a stable display name and color; unknown future
// teachers fall back to their slug and a neutral tint.
const SOURCE_STYLES = {
  fomo: { name: 'FOMO', color: FOMO_COLOR },
  yolo: { name: 'YOLO', color: YOLO_COLOR },
  yolo26x: { name: 'YOLO26x', color: '#7aa2f7' },
  'rtdetr-x': { name: 'RT-DETR-X', color: '#c678dd' },
}

function sourceStyle(source) {
  return SOURCE_STYLES[source] ?? { name: source, color: '#9a9a9a' }
}

function pct(value) {
  if (value === null || value === undefined) return '—'
  return `${(value * 100).toFixed(1)}%`
}

function modelName(version, hash) {
  return version ?? hash ?? 'unstamped'
}

function formatTimestamp(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

// FOMO scored against the Pi's YOLO detections. Wording matters here:
// these are agreement rates with a reference model, not accuracy against
// ground truth, and the UI never claims otherwise.
export default function EvalView({ onSelectImage }) {
  const queryClient = useQueryClient()
  const [backfilling, setBackfilling] = useState(false)
  const [backfillError, setBackfillError] = useState(null)
  const summaryQuery = useQuery({
    queryKey: ['eval-summary'],
    queryFn: () => apiFetch('/eval/summary'),
    refetchInterval: EVAL_POLL_INTERVAL_MS,
  })
  const disagreementsQuery = useQuery({
    queryKey: ['eval-disagreements'],
    queryFn: () =>
      apiFetch('/eval/disagreements', {
        params: { limit: DISAGREEMENT_LIMIT },
      }),
    refetchInterval: EVAL_POLL_INTERVAL_MS,
  })
  const runsQuery = useQuery({
    queryKey: ['teacher-runs'],
    queryFn: () => apiFetch('/eval/teacher/runs', { params: { limit: 1 } }),
    refetchInterval: EVAL_POLL_INTERVAL_MS,
  })

  const summary = summaryQuery.data
  const pairs = summary?.pairs ?? []
  const unscored = summary?.unscored_images ?? 0
  const disagreements = disagreementsQuery.data?.disagreements ?? []
  // The live pair (FOMO vs the Pi's YOLO) is scored as uploads arrive;
  // everything else comes from the offline teacher batch. Different
  // cadences, different meaning — shown as separate groups.
  const livePairs = pairs.filter((pair) => pair.teacher_source === 'yolo')
  const offlinePairs = pairs.filter((pair) => pair.teacher_source !== 'yolo')

  async function runBackfill() {
    setBackfilling(true)
    setBackfillError(null)
    try {
      await apiFetch('/eval/backfill', { method: 'POST' })
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['eval-summary'] }),
        queryClient.invalidateQueries({ queryKey: ['eval-disagreements'] }),
      ])
    } catch (backfillFailure) {
      setBackfillError(backfillFailure.message)
    } finally {
      setBackfilling(false)
    }
  }

  return (
    <div className="eval-view">
      <section className="eval-panel">
        <div className="eval-panel-header">
          <h2>Live agreement</h2>
          {summary && (
            <span className="eval-panel-note">
              scored as uploads arrive · labels:{' '}
              {(summary.labels ?? []).join(', ')} · teacher ≥{' '}
              {summary.teacher_min_confidence}
            </span>
          )}
          {unscored > 0 && (
            <button
              type="button"
              className="ghost"
              onClick={runBackfill}
              disabled={backfilling}
            >
              {backfilling
                ? 'Scoring…'
                : `Score ${unscored} unscored ${unscored === 1 ? 'image' : 'images'}`}
            </button>
          )}
        </div>
        {summaryQuery.isPending && (
          <p className="dashboard-status">Loading agreement summary…</p>
        )}
        {summaryQuery.error && summaryQuery.error.status !== 401 && (
          <p className="dashboard-status error">
            Failed to load eval summary: {summaryQuery.error.message}
          </p>
        )}
        {backfillError && (
          <p className="dashboard-status error">
            Backfill failed: {backfillError}
          </p>
        )}
        {summary && livePairs.length === 0 && (
          <p className="eval-panel-note">
            No images scored yet. Uploads are scored as they arrive.
          </p>
        )}
        {livePairs.map((pair) => (
          <PairRow key={pairKey(pair)} pair={pair} />
        ))}
      </section>

      <section className="eval-panel offline">
        <div className="eval-panel-header">
          <h2>Offline teachers</h2>
          <span className="eval-panel-note">
            bigger models, annotated by the nightly batch runner
          </span>
        </div>
        <TeacherRunStatus run={runsQuery.data?.runs?.[0]} />
        {summary && offlinePairs.length === 0 && (
          <p className="eval-panel-note">
            No teacher annotations yet — the nightly runner (or a manual
            backfill run) populates this.
          </p>
        )}
        {offlinePairs.map((pair) => (
          <PairRow key={pairKey(pair)} pair={pair} />
        ))}
      </section>

      <section>
        <div className="eval-section-header">
          <h2>Disagreements</h2>
          <span className="eval-panel-note">
            Frames where one model saw something the other did not — the
            first place to look when a number above moves.
          </span>
        </div>
        {disagreementsQuery.isPending && (
          <p className="dashboard-status">Loading disagreements…</p>
        )}
        {disagreementsQuery.error &&
          disagreementsQuery.error.status !== 401 && (
            <p className="dashboard-status error">
              Failed to load disagreements: {disagreementsQuery.error.message}
            </p>
          )}
        {disagreementsQuery.data && disagreements.length === 0 && (
          <p className="dashboard-status">
            No disagreements — every scored frame matched.
          </p>
        )}
        <div className="detection-grid">
          {disagreements.map((entry) => (
            <DisagreementCard
              key={entry.image_id}
              entry={entry}
              onClick={() => onSelectImage?.(entry.image_id)}
            />
          ))}
        </div>
      </section>
    </div>
  )
}

function pairKey(pair) {
  return `${pair.student_source}-${pair.student_hash}-${pair.teacher_source}-${pair.teacher_hash}`
}

function PairRow({ pair }) {
  return (
    <div className="eval-pair">
      <span className="eval-models">
        <span
          className="eval-model"
          style={{ '--chip-color': sourceStyle(pair.student_source).color }}
        >
          {sourceStyle(pair.student_source).name}{' '}
          {modelName(pair.student_version, pair.student_hash)}
        </span>
        {' vs '}
        <span
          className="eval-model"
          style={{ '--chip-color': sourceStyle(pair.teacher_source).color }}
        >
          {sourceStyle(pair.teacher_source).name}{' '}
          {modelName(pair.teacher_version, pair.teacher_hash)}
        </span>
      </span>
      <span className="eval-metric">
        <strong>{pct(pair.agreement_precision)}</strong> precision
      </span>
      <span className="eval-metric">
        <strong>{pct(pair.agreement_recall)}</strong> recall
      </span>
      <span className="eval-counts">
        {pair.images} {pair.images === 1 ? 'image' : 'images'}
        {pair.disagreement_images > 0 &&
          ` · ${pair.disagreement_images} disagree`}
        {pair.empty_images > 0 && ` · ${pair.empty_images} empty`}
      </span>
    </div>
  )
}

function TeacherRunStatus({ run }) {
  if (!run) return null
  const counts = Object.entries(run.detail ?? {})
    .filter(([, value]) => value && typeof value === 'object')
    .map(
      ([source, stats]) =>
        `${sourceStyle(source).name}: ${stats.images ?? 0} images` +
        (stats.errors > 0 ? ` (${stats.errors} errors)` : ''),
    )
  const when = formatTimestamp(run.finished_at ?? run.started_at)
  return (
    <p className="eval-panel-note">
      Teacher run {run.status === 'running' ? 'started' : run.status} {when}
      {run.runner ? ` on ${run.runner}` : ''}
      {counts.length > 0 ? ` — ${counts.join(' · ')}` : ''}
      {run.detail?.duration_s != null
        ? ` — ${Math.round(run.detail.duration_s / 60)} min`
        : ''}
    </p>
  )
}

function DisagreementCard({ entry, onClick }) {
  const [broken, setBroken] = useState(false)
  const missed = entry.teacher_only?.length ?? 0
  const extra = entry.student_only?.length ?? 0
  const labels = [
    ...new Set(
      [...(entry.teacher_only ?? []), ...(entry.student_only ?? [])]
        .map((det) => det.label)
        .filter(Boolean),
    ),
  ]

  return (
    <button type="button" className="detection-card" onClick={onClick}>
      {broken ? (
        <div className="detection-placeholder">image unavailable</div>
      ) : (
        <div className="thumb-wrap">
          <img
            src={detectionImageUrl(entry.image_id)}
            alt={`Disagreement on ${entry.device_id ?? 'unknown device'}`}
            loading="lazy"
            onError={() => setBroken(true)}
          />
        </div>
      )}
      <div className="detection-meta">
        <span className="detection-device">
          {entry.device_id ?? 'unknown device'}
        </span>
        <span className="detection-time">
          {formatTimestamp(entry.captured_at ?? entry.created_at)}
        </span>
        <div className="detection-badges">
          {missed > 0 && (
            <span className="badge" style={{ '--chip-color': YOLO_COLOR }}>
              FOMO missed {missed}
            </span>
          )}
          {extra > 0 && (
            <span className="badge" style={{ '--chip-color': FOMO_COLOR }}>
              FOMO extra {extra}
            </span>
          )}
          {labels.length > 0 && (
            <span className="detection-labels">{labels.join(', ')}</span>
          )}
        </div>
      </div>
    </button>
  )
}
