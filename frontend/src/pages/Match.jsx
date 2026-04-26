import { useState } from 'react'
import { api } from '../api/client'
import DropZone from '../components/DropZone'
import MatchCard from '../components/MatchCard'
import { Loader, SearchX } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function Match() {
  const [file, setFile]     = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError]   = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('file', file)

    try {
      const { data } = await api.matchClip(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Matching failed')
    } finally {
      setLoading(false)
    }
  }

  const chartData = result?.sources?.map((s) => ({
    name: s.title.length > 18 ? s.title.slice(0, 18) + '…' : s.title,
    confidence: Math.round(s.confidence * 100),
  })) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Match Clip</h1>
        <p className="text-gray-400 text-sm mt-1">
          Upload any clip — edited, cropped, or mixed — to identify its source video(s).
        </p>
      </div>

      <form onSubmit={submit} className="card space-y-4 max-w-xl">
        <DropZone onFile={setFile} label="Drop clip to match or click to browse" />
        <button
          type="submit"
          disabled={!file || loading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? <><Loader size={16} className="animate-spin" /> Matching…</> : 'Find Source'}
        </button>
      </form>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <span>{result.query_segments} query segments</span>
            <span>·</span>
            <span>{result.matches_found} source{result.matches_found !== 1 ? 's' : ''} found</span>
          </div>

          {result.matches_found === 0 ? (
            <div className="card text-center py-12 text-gray-500">
              <SearchX size={36} className="mx-auto mb-3 opacity-30" />
              <p>No matching source found in the database.</p>
            </div>
          ) : (
            <>
              {/* Confidence chart */}
              {chartData.length > 0 && (
                <div className="card">
                  <p className="text-sm font-medium text-gray-400 mb-4">Confidence by Source</p>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
                      <XAxis dataKey="name" tick={{ fill: '#9ca3af', fontSize: 11 }} />
                      <YAxis domain={[0, 100]} tick={{ fill: '#9ca3af', fontSize: 11 }} unit="%" />
                      <Tooltip
                        contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
                        labelStyle={{ color: '#f3f4f6' }}
                        formatter={(v) => [`${v}%`, 'Confidence']}
                      />
                      <Bar dataKey="confidence" radius={[6, 6, 0, 0]}>
                        {chartData.map((_, i) => (
                          <Cell key={i} fill={i === 0 ? '#6366f1' : '#374151'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Match cards */}
              <div className="space-y-4">
                {result.sources.map((m, i) => (
                  <MatchCard key={m.video_id} match={m} rank={i + 1} />
                ))}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}
