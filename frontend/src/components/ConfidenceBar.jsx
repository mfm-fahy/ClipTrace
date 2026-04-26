export default function ConfidenceBar({ value }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 85 ? 'bg-green-500' :
    pct >= 60 ? 'bg-yellow-500' :
                'bg-red-500'

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
        <div
          className={`h-2 rounded-full transition-all ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-sm font-semibold w-12 text-right">{pct}%</span>
    </div>
  )
}
