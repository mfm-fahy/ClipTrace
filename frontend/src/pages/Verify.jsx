import { useState } from 'react'
import { api } from '../api/client'
import DropZone from '../components/DropZone'
import MatchCard from '../components/MatchCard'
import { Loader, ShieldCheck, ShieldX, ShieldAlert, Link } from 'lucide-react'

export default function Verify() {
  const [file, setFile]       = useState(null)
  const [chainId, setChainId] = useState('')
  const [result, setResult]   = useState(null)
  const [chainResult, setChainResult] = useState(null)
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [chainLoading, setChainLoading] = useState(false)

  const submitClip = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('file', file)

    try {
      const { data } = await api.verifyClip(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Verification failed')
    } finally {
      setLoading(false)
    }
  }

  const submitChain = async (e) => {
    e.preventDefault()
    if (!chainId.trim()) return
    setChainLoading(true); setChainResult(null)
    try {
      const { data } = await api.verifyChain(chainId.trim())
      setChainResult(data)
    } catch (err) {
      setChainResult({ error: err.response?.data?.detail ?? 'Not found' })
    } finally {
      setChainLoading(false)
    }
  }

  const StatusIcon = result
    ? result.is_original
      ? ShieldCheck
      : result.status === 'edited'
        ? ShieldAlert
        : ShieldX
    : null

  const statusColor = result
    ? result.is_original ? 'text-green-400' : result.status === 'edited' ? 'text-yellow-400' : 'text-red-400'
    : ''

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Verify Authenticity</h1>
        <p className="text-gray-400 text-sm mt-1">
          "Shazam for Sports Video" — open API to check any clip's origin and integrity.
        </p>
      </div>

      {/* Clip verification */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-gray-300">Clip Verification</h2>
        <form onSubmit={submitClip} className="card space-y-4 max-w-xl">
          <DropZone onFile={setFile} label="Drop clip to verify or click to browse" />
          <button
            type="submit"
            disabled={!file || loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading ? <><Loader size={16} className="animate-spin" /> Verifying…</> : 'Verify Clip'}
          </button>
        </form>

        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-4 max-w-xl">
            <div className="card space-y-3">
              <div className={`flex items-center gap-2 font-semibold ${statusColor}`}>
                {StatusIcon && <StatusIcon size={20} />}
                {result.is_original ? 'Original / Unregistered Content' : `Content ${result.status === 'edited' ? 'Edited' : 'Reused'}`}
              </div>
              <p className="text-sm text-gray-300">{result.message}</p>
              {result.is_mixed_content && (
                <div className="badge bg-yellow-900/40 text-yellow-400 text-xs">
                  Mixed-content video detected
                </div>
              )}
            </div>

            {result.sources?.map((m, i) => (
              <MatchCard key={m.video_id} match={m} rank={i + 1} />
            ))}
          </div>
        )}
      </div>

      {/* Chain integrity check */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-gray-300 flex items-center gap-2">
          <Link size={16} /> Hash Chain Integrity
        </h2>
        <form onSubmit={submitChain} className="card flex gap-3 max-w-xl">
          <input
            value={chainId}
            onChange={(e) => setChainId(e.target.value)}
            placeholder="Paste Video ID to verify chain…"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm
                       focus:outline-none focus:border-brand-500"
          />
          <button
            type="submit"
            disabled={!chainId.trim() || chainLoading}
            className="btn-primary flex items-center gap-2"
          >
            {chainLoading ? <Loader size={15} className="animate-spin" /> : 'Check'}
          </button>
        </form>

        {chainResult && (
          <div className="card max-w-xl space-y-2">
            {chainResult.error ? (
              <p className="text-red-400 text-sm">{chainResult.error}</p>
            ) : (
              <>
                <div className={`flex items-center gap-2 font-semibold ${chainResult.chain_intact ? 'text-green-400' : 'text-red-400'}`}>
                  {chainResult.chain_intact
                    ? <><ShieldCheck size={18} /> Chain Intact — Content Unmodified</>
                    : <><ShieldX size={18} /> Chain Broken — Possible Tampering</>}
                </div>
                <p className="text-sm text-gray-400">{chainResult.title}</p>
                <p className="text-xs text-gray-500">{chainResult.total_segments} segments verified</p>
                <p className="font-mono text-xs text-brand-400 break-all">{chainResult.chain_root_hash}</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
