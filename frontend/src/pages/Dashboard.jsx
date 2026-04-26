import { useEffect, useState } from 'react'
import { api } from '../api/client'
import { Film, ShieldCheck, ShieldX, Trash2, RefreshCw } from 'lucide-react'

export default function Dashboard() {
  const [videos, setVideos] = useState([])
  const [chains, setChains] = useState({})
  const [loading, setLoading] = useState(true)

  const load = async () => {
    setLoading(true)
    try {
      const { data } = await api.listVideos()
      setVideos(data)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const checkChain = async (id) => {
    try {
      const { data } = await api.verifyChain(id)
      setChains((c) => ({ ...c, [id]: data }))
    } catch {
      setChains((c) => ({ ...c, [id]: { chain_intact: false } }))
    }
  }

  const remove = async (id) => {
    await api.deleteVideo(id)
    setVideos((v) => v.filter((x) => x.video_id !== id))
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-gray-400 text-sm mt-1">
            {videos.length} video{videos.length !== 1 ? 's' : ''} registered
          </p>
        </div>
        <button onClick={load} className="btn-secondary flex items-center gap-2">
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Registered Videos', value: videos.length },
          { label: 'Chains Verified',   value: Object.values(chains).filter(c => c.chain_intact).length },
          { label: 'Owners',            value: new Set(videos.map(v => v.owner)).size },
          { label: 'Total Duration',    value: `${videos.reduce((s, v) => s + (v.duration || 0), 0).toFixed(0)}s` },
        ].map(({ label, value }) => (
          <div key={label} className="card text-center">
            <p className="text-2xl font-bold text-brand-400">{value}</p>
            <p className="text-xs text-gray-500 mt-1">{label}</p>
          </div>
        ))}
      </div>

      {/* Video list */}
      {loading ? (
        <p className="text-gray-500 text-center py-12">Loading…</p>
      ) : videos.length === 0 ? (
        <div className="card text-center py-16 text-gray-500">
          <Film size={40} className="mx-auto mb-3 opacity-30" />
          <p>No videos registered yet. Go to <strong>Register</strong> to add one.</p>
        </div>
      ) : (
        <div className="space-y-3">
          {videos.map((v) => {
            const chain = chains[v.video_id]
            return (
              <div key={v.video_id} className="card flex items-center gap-4">
                <Film size={20} className="text-brand-400 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-semibold truncate">{v.title}</p>
                  <p className="text-xs text-gray-400">
                    Owner: {v.owner} · {v.duration?.toFixed(1)}s ·{' '}
                    {new Date(v.registered_at * 1000).toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-600 font-mono truncate mt-0.5">
                    {v.chain_root_hash}
                  </p>
                </div>

                {/* Chain status */}
                {chain ? (
                  chain.chain_intact
                    ? <ShieldCheck size={18} className="text-green-400 shrink-0" title="Chain intact" />
                    : <ShieldX    size={18} className="text-red-400 shrink-0"   title="Chain tampered" />
                ) : (
                  <button
                    onClick={() => checkChain(v.video_id)}
                    className="text-xs text-gray-500 hover:text-brand-400 transition-colors shrink-0"
                  >
                    Verify chain
                  </button>
                )}

                <button
                  onClick={() => remove(v.video_id)}
                  className="text-gray-600 hover:text-red-400 transition-colors shrink-0"
                >
                  <Trash2 size={16} />
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
