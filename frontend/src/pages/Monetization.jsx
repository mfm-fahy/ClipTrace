import { useState, useEffect } from 'react'
import { api } from '../api/client'
import DropZone from '../components/DropZone'
import { Loader, DollarSign, Settings } from 'lucide-react'
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e']

export default function Monetization() {
  const [file, setFile]       = useState(null)
  const [revenue, setRevenue] = useState('100')
  const [result, setResult]   = useState(null)
  const [rules, setRules]     = useState([])
  const [error, setError]     = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    api.listRules().then(({ data }) => setRules(data)).catch(() => {})
  }, [])

  const submit = async (e) => {
    e.preventDefault()
    if (!file) return
    setLoading(true); setError(null); setResult(null)

    const fd = new FormData()
    fd.append('file', file)
    fd.append('total_revenue', parseFloat(revenue) || 0)

    try {
      const { data } = await api.routeRevenue(fd)
      setResult(data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Revenue routing failed')
    } finally {
      setLoading(false)
    }
  }

  const updateRule = async (videoId, action) => {
    await api.updateRule(videoId, action, 1.0)
    setRules((r) => r.map((x) => x.video_id === videoId ? { ...x, action } : x))
  }

  const pieData = result?.allocations?.map((a) => ({
    name: a.owner,
    value: a.revenue_share_pct,
  })) ?? []

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold">Smart Monetization</h1>
        <p className="text-gray-400 text-sm mt-1">
          Automatically route revenue to rightful owners instead of blocking content.
        </p>
      </div>

      {/* Revenue routing */}
      <div className="space-y-4">
        <h2 className="text-base font-semibold text-gray-300">Route Revenue for a Clip</h2>
        <form onSubmit={submit} className="card space-y-4 max-w-xl">
          <DropZone onFile={setFile} label="Drop clip to analyse revenue routing" />
          <div>
            <label className="text-sm text-gray-400 block mb-1">Total Revenue ($)</label>
            <input
              type="number"
              min="0"
              step="0.01"
              value={revenue}
              onChange={(e) => setRevenue(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm
                         focus:outline-none focus:border-brand-500"
            />
          </div>
          <button
            type="submit"
            disabled={!file || loading}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            {loading
              ? <><Loader size={16} className="animate-spin" /> Analysing…</>
              : <><DollarSign size={16} /> Calculate Revenue Split</>}
          </button>
        </form>

        {error && (
          <div className="bg-red-900/30 border border-red-800 rounded-xl p-4 text-red-300 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-4">
            {result.mixed_content_detected && (
              <div className="badge bg-yellow-900/40 text-yellow-400 text-sm px-3 py-1">
                Mixed-content detected — revenue split across multiple owners
              </div>
            )}

            <div className="grid md:grid-cols-2 gap-6">
              {/* Pie chart */}
              {pieData.length > 0 && (
                <div className="card">
                  <p className="text-sm font-medium text-gray-400 mb-2">Revenue Distribution</p>
                  <ResponsiveContainer width="100%" height={220}>
                    <PieChart>
                      <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label={({ name, value }) => `${name} ${value}%`}>
                        {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                      </Pie>
                      <Tooltip formatter={(v) => [`${v}%`, 'Share']} contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}

              {/* Allocation table */}
              <div className="card space-y-3">
                <p className="text-sm font-medium text-gray-400">Allocation Breakdown</p>
                {result.allocations.map((a, i) => (
                  <div key={a.video_id} className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full shrink-0" style={{ background: COLORS[i % COLORS.length] }} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{a.title}</p>
                      <p className="text-xs text-gray-500">{a.owner}</p>
                    </div>
                    <div className="text-right shrink-0">
                      <p className="text-sm font-semibold text-green-400">${a.allocated_revenue}</p>
                      <p className="text-xs text-gray-500">{a.revenue_share_pct}%</p>
                    </div>
                  </div>
                ))}
                <div className="border-t border-gray-800 pt-2 flex justify-between text-sm font-semibold">
                  <span>Total</span>
                  <span className="text-green-400">${result.total_revenue}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Rules management */}
      {rules.length > 0 && (
        <div className="space-y-3">
          <h2 className="text-base font-semibold text-gray-300 flex items-center gap-2">
            <Settings size={16} /> Monetization Rules
          </h2>
          <div className="space-y-2">
            {rules.map((r) => (
              <div key={r.video_id} className="card flex items-center gap-4">
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{r.video_id}</p>
                  <p className="text-xs text-gray-500">{r.owner}</p>
                </div>
                <select
                  value={r.action}
                  onChange={(e) => updateRule(r.video_id, e.target.value)}
                  className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm
                             focus:outline-none focus:border-brand-500"
                >
                  <option value="monetize">Monetize</option>
                  <option value="block">Block</option>
                  <option value="allow">Allow Free</option>
                </select>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
