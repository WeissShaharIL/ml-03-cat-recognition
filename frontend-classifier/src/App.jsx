import { useState, useRef } from 'react'

const API = '/api'

export default function App() {
  const [image, setImage]         = useState(null)   // base64 preview
  const [dragging, setDragging]   = useState(false)
  const [loading, setLoading]     = useState(false)
  const [result, setResult]       = useState(null)   // { prediction, confidence, probabilities }
  const [error, setError]         = useState(null)
  const inputRef = useRef(null)

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => setImage(e.target.result)
    reader.readAsDataURL(file)
    setResult(null)
    setError(null)
    predict(file)
  }

  const predict = async (file) => {
    setLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res  = await fetch(`${API}/predict`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail)
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault(); setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div style={{ fontFamily: 'system-ui', minHeight: '100vh', background: '#0a0a0b', color: '#f0f0f0', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: 40 }}>
      <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 8 }}>🐱 Cat Recognizer</h1>
      <p style={{ color: '#666', marginBottom: 32 }}>Drop an image to find out if it's a cat</p>

      {/* Drop zone */}
      <div
        onClick={() => inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        style={{
          width: 340, height: 280, borderRadius: 14,
          border: `2px dashed ${dragging ? '#3b82f6' : '#333'}`,
          background: dragging ? '#1a1a2e' : '#111113',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', transition: 'all 0.2s', overflow: 'hidden', position: 'relative',
        }}
      >
        {image ? (
          <img src={image} alt="preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
        ) : (
          <>
            <span style={{ fontSize: 40, marginBottom: 12 }}>📁</span>
            <span style={{ color: '#666', fontSize: 14 }}>Drop image here or click to upload</span>
          </>
        )}
        <input ref={inputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={e => handleFile(e.target.files[0])} />
      </div>

      {/* Result */}
      {loading && <p style={{ marginTop: 24, color: '#666' }}>Analyzing...</p>}
      {error && <p style={{ marginTop: 24, color: '#ef4444' }}>✗ {error}</p>}
      {result && !loading && (
        <div style={{ marginTop: 24, width: 340, background: '#111113', border: '1px solid #222228', borderRadius: 12, padding: '20px 24px' }}>
          <p style={{ fontSize: 13, color: '#666', marginBottom: 14, textAlign: 'center' }}>
            Prediction: <strong style={{ color: result.prediction === 'cat' ? '#22c55e' : '#f0f0f0' }}>
              {result.prediction === 'cat' ? '🐱 Cat' : '✗ Not a cat'}
            </strong> &nbsp;({(result.confidence * 100).toFixed(1)}% confident)
          </p>
          {[
            { label: '🐱 Cat',      key: 'cat',     color: '#22c55e' },
            { label: '✗ Not a cat', key: 'not_cat', color: '#ef4444' },
          ].map(({ label, key, color }) => {
            const pct = ((result.probabilities[key] || 0) * 100).toFixed(1)
            return (
              <div key={key} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
                  <span>{label}</span>
                  <span style={{ color, fontWeight: 600 }}>{pct}%</span>
                </div>
                <div style={{ height: 6, background: '#222228', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}