import { useState } from 'react'
import { api } from '../api/client'
import DropZone from '../components/DropZone'
import { CheckCircle, Loader } from 'lucide-react'

export default function Register() {
  const [file, setFile]     = useState(null)
  const [title, setTitle]   = useState('')
  const [owner, setOwner]   = useState('')
  const [result, setResult] = useState(null)
  const [error, setError]   = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    if (!file || !title || !owner) return
    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('file', file)
    fd.append('title', title)
    fd.append('owner', owner)

    try {
      const { data } = await api.registerVideo(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Register Video</h1>
        <p className="text-gray-400 text-sm mt-1">
          Register an original video to generate its fingerprint identity and tamper-proof chain.
        </p>
      </div>

      <form onSubmit={submit} className="card space-y-5">
        <DropZone onFile={setFile} label="Drop original video here or click to browse" />

        <div className="space-y-3">
          <div>
            <label className="text-sm text-gray-400 block mb-1">Video Title</label>
            <input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Champions League Final 2024"
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm
                         focus:outline-none focus:border-brand-500"
            />
          </div>
          <div>
            <label className="text-sm text-gray-400 block mb-1">Owner / Rights Holder</label>
            <input
              value={owner}
              onChange={(e) => setOwner(e.target.value)}
              placeholder="e.g. UEFA Media"
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm
                         focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={!file || !title || !owner || loading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? <><Loader size={16} className="animate-spin" /> Processing…</> : 'Register Video'}
        </button>
      </form>

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {result && (
        <div className="card space-y-3">
          <div className="flex items-center gap-2 text-green-400 font-semibold">
            <CheckCircle size={18} /> Video Registered Successfully
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            {[
              ['Video ID',  result.video_id],
              ['Title',     result.title],
              ['Owner',     result.owner],
              ['Duration',  `${result.duration?.toFixed(1)}s`],
              ['Segments',  result.segments_registered],
            ].map(([k, v]) => (
              <div key={k} className="bg-gray-800 rounded-lg p-3">
                <p className="text-gray-500 text-xs">{k}</p>
                <p className="font-medium truncate">{v}</p>
              </div>
            ))}
          </div>
          <div className="bg-gray-800 rounded-lg p-3 text-xs">
            <p className="text-gray-500 mb-1">Chain Root Hash (Capture-Time Proof)</p>
            <p className="font-mono text-brand-400 break-all">{result.chain_root_hash}</p>
          </div>
        </div>
      )}
    </div>
  )
}
