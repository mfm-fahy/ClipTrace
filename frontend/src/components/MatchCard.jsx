import ConfidenceBar from './ConfidenceBar'
import { Clock, User, Layers } from 'lucide-react'

export default function MatchCard({ match, rank }) {
  const { title, owner, confidence, matched_segments, matched_timestamps, is_mixed } = match

  return (
    <div className="card space-y-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-brand-400 bg-brand-900/40 px-2 py-0.5 rounded-full">
              #{rank}
            </span>
            {is_mixed && (
              <span className="badge bg-yellow-900/50 text-yellow-400">
                <Layers size={11} /> Mixed Source
              </span>
            )}
          </div>
          <h3 className="text-base font-semibold mt-1">{title}</h3>
          <p className="text-sm text-gray-400 flex items-center gap-1 mt-0.5">
            <User size={13} /> {owner}
          </p>
        </div>
        <div className="text-right text-xs text-gray-500">
          {matched_segments} segments matched
        </div>
      </div>

      <ConfidenceBar value={confidence} />

      {matched_timestamps?.length > 0 && (
        <div className="space-y-1">
          <p className="text-xs text-gray-500 font-medium uppercase tracking-wide">
            Matched Timestamps
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-36 overflow-y-auto pr-1">
            {matched_timestamps.map((t, i) => (
              <div key={i} className="bg-gray-800 rounded-lg px-3 py-2 text-xs">
                <div className="flex items-center gap-1 text-gray-400">
                  <Clock size={11} />
                  <span>Query {t.query_time}s</span>
                </div>
                <div className="text-gray-200 font-medium mt-0.5">
                  {t.source_start.toFixed(1)}s – {t.source_end.toFixed(1)}s
                </div>
                <div className="text-brand-400 mt-0.5">{Math.round(t.score * 100)}% sim</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
