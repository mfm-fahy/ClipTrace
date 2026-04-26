import { useRef, useState } from 'react'
import { UploadCloud } from 'lucide-react'

export default function DropZone({ onFile, accept = 'video/*', label = 'Drop a video file' }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)
  const [fileName, setFileName] = useState(null)

  const handle = (file) => {
    if (!file) return
    setFileName(file.name)
    onFile(file)
  }

  return (
    <div
      onClick={() => inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files[0]) }}
      className={`cursor-pointer border-2 border-dashed rounded-2xl p-10 flex flex-col items-center gap-3
                  transition-colors select-none
                  ${dragging ? 'border-brand-500 bg-brand-900/20' : 'border-gray-700 hover:border-gray-500'}`}
    >
      <UploadCloud size={36} className="text-gray-500" />
      <p className="text-gray-400 text-sm">{fileName ?? label}</p>
      {fileName && <p className="text-brand-400 text-xs font-medium">{fileName}</p>}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handle(e.target.files[0])}
      />
    </div>
  )
}
