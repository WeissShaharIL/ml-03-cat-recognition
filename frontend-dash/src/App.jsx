import { useEffect, useState, useRef } from 'react'

const API = '/api'

const themes = {
  dark:  { bg: '#0a0a0b', surface: '#111113', surface2: '#18181c', border: '#222228', text: '#f0f0f0', muted: '#666', up: '#22c55e', down: '#ef4444', inputBg: '#0e0e11', logBg: '#08080a', accent: '#3b82f6' },
  light: { bg: '#f0f0f2', surface: '#ffffff',  surface2: '#f7f7f9', border: '#e0e0e6', text: '#111116', muted: '#888', up: '#16a34a', down: '#dc2626', inputBg: '#ffffff', logBg: '#eeeef2', accent: '#2563eb' },
}

const SectionHeader = ({ title, t }) => (
  <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: t.muted, marginBottom: 12, paddingBottom: 8, borderBottom: `1px solid ${t.border}` }}>
    {title}
  </div>
)

const Card = ({ children, t, style = {} }) => (
  <div style={{ background: t.surface, border: `1px solid ${t.border}`, borderRadius: 10, padding: '14px 16px', marginBottom: 12, ...style }}>
    {children}
  </div>
)

const StatusDot = ({ ok, t }) => (
  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: ok ? t.up : t.down, marginRight: 8, flexShrink: 0 }} />
)

const AccBar = ({ value, t }) => {
  const pct   = (value * 100).toFixed(1)
  const color = value >= 0.9 ? t.up : value >= 0.7 ? '#facc15' : t.down
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 5, background: t.border, borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3, transition: 'width 0.4s ease' }} />
      </div>
      <span style={{ fontSize: 11, fontWeight: 600, color, minWidth: 36, textAlign: 'right' }}>{pct}%</span>
    </div>
  )
}

export default function App() {
  const [mode, setMode]         = useState('dark')
  const [health, setHealth]     = useState(null)
  const [models, setModels]     = useState([])
  const [runs, setRuns]         = useState([])
  const [expandedRun, setExpandedRun] = useState(null)
  const [message, setMessage]   = useState(null)
  const [deploying, setDeploying] = useState(null)  // filename being deployed

  const [resetConfirm, setResetConfirm] = useState(false)
  const [resetting, setResetting]       = useState(false)

  // Train state
  const [trainEpochs,    setTrainEpochs]    = useState(5)
  const [trainBatchSize, setTrainBatchSize] = useState(32)
  const [trainLr,        setTrainLr]        = useState(0.001)
  const [training,       setTraining]       = useState(false)
  const [trainLog,       setTrainLog]       = useState([])
  const [currentEpoch,   setCurrentEpoch]   = useState(null)

  const trainLogRef = useRef(null)
  const t = themes[mode]

  const fetchHealth = async () => {
    try { setHealth(await (await fetch(`${API}/health`)).json()) } catch { setHealth(null) }
  }
  const fetchModels = async () => {
    try { setModels(await (await fetch(`${API}/models`)).json()) } catch { setModels([]) }
  }
  const fetchRuns = async () => {
    try { setRuns((await (await fetch(`${API}/train/runs`)).json()).reverse()) } catch { setRuns([]) }
  }

  useEffect(() => {
    fetchHealth(); fetchModels(); fetchRuns()
    const iv = setInterval(() => { fetchHealth() }, 10000)
    return () => clearInterval(iv)
  }, [])

  useEffect(() => {
    if (trainLogRef.current) trainLogRef.current.scrollTop = trainLogRef.current.scrollHeight
  }, [trainLog])

  // ── Actions ───────────────────────────────────────────────────────────────────
  const deployModel = async (filename) => {
    setDeploying(filename)
    try {
      const res = await fetch(`${API}/models/${encodeURIComponent(filename)}/deploy`, { method: 'POST' })
      if (!res.ok) throw new Error((await res.json()).detail)
      setMessage({ type: 'success', text: `✓ ${filename} deployed — classifier is now using this model` })
      fetchHealth(); fetchModels()
    } catch (e) { setMessage({ type: 'error', text: `✗ ${e.message}` }) }
    finally { setDeploying(null) }
  }

  const resetAll = async () => {
    setResetting(true)
    try {
      await fetch(`${API}/reset`, { method: 'POST' })
      setModels([]); setRuns([]); setTrainLog([]); setCurrentEpoch(null)
      setMessage({ type: 'success', text: '✓ Model and history wiped. Images kept.' })
      fetchHealth()
    } catch (e) { setMessage({ type: 'error', text: `✗ ${e.message}` }) }
    finally { setResetting(false); setResetConfirm(false) }
  }

  // ── Train ─────────────────────────────────────────────────────────────────────
  const runTrain = async () => {
    setTraining(true); setTrainLog([]); setCurrentEpoch(null)
    try {
      const res    = await fetch(`${API}/train`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ epochs: parseInt(trainEpochs), batch_size: parseInt(trainBatchSize), lr: parseFloat(trainLr) }) })
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = ''
      while (true) {
        const { done, value } = await reader.read(); if (done) break
        buf += dec.decode(value, { stream: true })
        const lines = buf.split('\n'); buf = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const p = JSON.parse(line.slice(6))
          if (p.type === 'status')    setTrainLog(prev => [...prev, { text: p.text, color: t.muted }])
          if (p.type === 'batch')     setTrainLog(prev => {
            const last = prev[prev.length - 1]
            const txt  = `Epoch ${p.epoch}/${p.epochs} — batch ${p.batch}/${p.batches}`
            if (last && last.isBatch) return [...prev.slice(0, -1), { text: txt, color: t.muted, isBatch: true }]
            return [...prev, { text: txt, color: t.muted, isBatch: true }]
          })
          if (p.type === 'epoch') {
            setCurrentEpoch(p)
            setTrainLog(prev => [...prev.filter(l => !l.isBatch), {
              text: `Epoch ${p.epoch}/${p.epochs} — train_acc: ${(p.train_acc*100).toFixed(1)}%  val_acc: ${(p.val_acc*100).toFixed(1)}%  val_loss: ${p.val_loss}`,
              color: p.val_acc >= 0.9 ? t.up : t.muted,
            }])
          }
          if (p.type === 'error')     setTrainLog(prev => [...prev, { text: `✗ ${p.text}`, color: t.down }])
          if (p.type === 'cancelled') setTrainLog(prev => [...prev, { text: `⏹ Training cancelled.`, color: '#facc15' }])
          if (p.type === 'done') {
            setTrainLog(prev => [...prev, { text: `✓ Done — best val acc: ${(p.best_val_acc*100).toFixed(1)}%  |  ${p.model_file}`, color: t.up }])
            fetchHealth(); fetchModels(); fetchRuns()
          }
        }
      }
    } catch (e) { setTrainLog(prev => [...prev, { text: `✗ ${e.message}`, color: t.down }]) }
    finally { setTraining(false); setCurrentEpoch(null) }
  }

  // ── Styles ────────────────────────────────────────────────────────────────────
  const btn = (disabled, color = t.accent) => ({ padding: '8px 16px', background: color, color: '#fff', border: 'none', borderRadius: 7, fontSize: 13, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? 0.5 : 1, whiteSpace: 'nowrap' })
  const numInput = (width = 60) => ({ width, padding: '4px 8px', borderRadius: 6, border: `1px solid ${t.border}`, background: t.inputBg, color: t.text, fontSize: 13, textAlign: 'center', outline: 'none' })

  return (
    <div style={{ fontFamily: 'system-ui', minHeight: '100vh', background: t.bg, color: t.text, display: 'flex', flexDirection: 'column' }}>

      {/* ── Top bar ─────────────────────────────────────────────────────────── */}
      <div style={{ borderBottom: `1px solid ${t.border}`, background: t.surface, padding: '0 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', height: 52, flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 15, fontWeight: 700, letterSpacing: '-0.02em' }}>🐱 Cat Recognition</span>
          <span style={{ fontSize: 11, color: t.muted, padding: '2px 8px', border: `1px solid ${t.border}`, borderRadius: 20 }}>DevOps Dashboard</span>
          {health?.deployed_model ? (
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: '#16a34a22', color: t.up, border: '1px solid #16a34a44' }}>
              Serving: <span style={{ fontFamily: 'monospace', fontWeight: 700 }}>{health.deployed_model}</span>
            </span>
          ) : (
            <span style={{ fontSize: 11, padding: '2px 8px', borderRadius: 20, background: '#88888822', color: t.muted, border: `1px solid ${t.border}` }}>
              No model deployed
            </span>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {message && <span style={{ fontSize: 12, color: message.type === 'success' ? t.up : t.down, maxWidth: 400, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{message.text}</span>}
          {resetConfirm ? (
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: t.down }}>Wipe model and history?</span>
              <button onClick={resetAll} disabled={resetting} style={{ padding: '5px 12px', borderRadius: 7, border: 'none', background: '#ef4444', color: '#fff', cursor: 'pointer', fontSize: 12, opacity: resetting ? 0.5 : 1 }}>{resetting ? '...' : 'Yes, wipe'}</button>
              <button onClick={() => setResetConfirm(false)} style={{ padding: '5px 12px', borderRadius: 7, border: `1px solid ${t.border}`, background: 'transparent', color: t.text, cursor: 'pointer', fontSize: 12 }}>Cancel</button>
            </div>
          ) : (
            <button onClick={() => setResetConfirm(true)} style={{ padding: '5px 12px', borderRadius: 7, border: `1px solid ${t.down}`, background: 'transparent', color: t.down, cursor: 'pointer', fontSize: 12 }}>🗑 Reset</button>
          )}
          <button onClick={() => setMode(m => m === 'dark' ? 'light' : 'dark')} style={{ padding: '5px 12px', borderRadius: 7, border: `1px solid ${t.border}`, background: 'transparent', color: t.text, cursor: 'pointer', fontSize: 12 }}>
            {mode === 'dark' ? '☀ Light' : '🌙 Dark'}
          </button>
        </div>
      </div>

      {/* ── 3-column body ───────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: '220px 1fr 300px', minHeight: 0 }}>

        {/* ── LEFT: Status ─────────────────────────────────────────────────── */}
        <div style={{ borderRight: `1px solid ${t.border}`, padding: '20px 14px', overflowY: 'auto' }}>
          <SectionHeader title="Services" t={t} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 24 }}>
            {!health && <p style={{ color: t.muted, fontSize: 12 }}>Checking...</p>}
            {health && [
              { label: 'Backend', ok: health.status === 'ok' },
              { label: 'Model',   ok: health.model_ready },
            ].map(({ label, ok }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', fontSize: 13, padding: '7px 10px', borderRadius: 7, background: t.surface2, border: `1px solid ${t.border}` }}>
                <StatusDot ok={ok} t={t} />
                <span style={{ flex: 1 }}>{label}</span>
                <span style={{ fontSize: 11, color: ok ? t.up : t.down, fontWeight: 600 }}>{ok ? 'ready' : 'not ready'}</span>
              </div>
            ))}
          </div>

          <SectionHeader title="Dataset" t={t} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 24 }}>
            {[
              { label: '🐱 Cat',    count: health?.dataset?.cat     ?? 0 },
              { label: '✗ Not cat', count: health?.dataset?.not_cat ?? 0 },
            ].map(({ label, count }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', fontSize: 13, padding: '7px 10px', borderRadius: 7, background: t.surface2, border: `1px solid ${t.border}` }}>
                <span style={{ flex: 1 }}>{label}</span>
                <span style={{ fontWeight: 600, color: count > 0 ? t.text : t.muted }}>{count} imgs</span>
              </div>
            ))}
          </div>

          <SectionHeader title="Links" t={t} />
          <a href="http://localhost:3000" target="_blank" rel="noreferrer"
            style={{ display: 'block', fontSize: 12, color: t.accent, padding: '6px 0' }}>
            → Open Classifier UI
          </a>
        </div>

        {/* ── CENTER: Train ────────────────────────────────────────────────── */}
        <div style={{ padding: '20px 24px', overflowY: 'auto', borderRight: `1px solid ${t.border}` }}>

          <SectionHeader title="Train Model" t={t} />
          <Card t={t}>
            <div style={{ display: 'flex', gap: 12, alignItems: 'stretch', marginBottom: 12 }}>
              <div style={{ border: `1px solid ${t.border}`, borderRadius: 8, padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                  <span style={{ color: t.muted, width: 70 }}>Epochs</span>
                  <input type="text" inputMode="numeric" value={trainEpochs} onChange={e => setTrainEpochs(e.target.value)} style={numInput()} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                  <span style={{ color: t.muted, width: 70 }}>Batch size</span>
                  <input type="text" inputMode="numeric" value={trainBatchSize} onChange={e => setTrainBatchSize(e.target.value)} style={numInput()} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
                  <span style={{ color: t.muted, width: 70 }}>Learn rate</span>
                  <input type="text" value={trainLr} onChange={e => setTrainLr(e.target.value)} style={numInput(70)} />
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                <button onClick={runTrain} disabled={training} style={{ ...btn(training, '#059669'), flex: 1 }}>
                  {training ? '⟳ Training...' : '▶ Train Model'}
                </button>
                {training && (
                  <button onClick={() => fetch(`${API}/train/cancel`, { method: 'POST' })}
                    style={{ padding: '8px 16px', background: 'transparent', color: t.down, border: `1px solid ${t.down}`, borderRadius: 7, fontSize: 13, cursor: 'pointer' }}>
                    ✕ Cancel
                  </button>
                )}
              </div>
            </div>

            {/* Live epoch metrics */}
            {currentEpoch && (
              <div style={{ marginBottom: 10, padding: '10px 12px', background: t.surface2, borderRadius: 8, border: `1px solid ${t.border}` }}>
                <div style={{ fontSize: 12, color: t.muted, marginBottom: 6 }}>
                  Epoch {currentEpoch.epoch}/{currentEpoch.epochs} — best val acc so far
                </div>
                <AccBar value={currentEpoch.best_val_acc} t={t} />
                <div style={{ display: 'flex', gap: 16, fontSize: 11, color: t.muted, marginTop: 6 }}>
                  <span>train acc: {(currentEpoch.train_acc*100).toFixed(1)}%</span>
                  <span>val acc: {(currentEpoch.val_acc*100).toFixed(1)}%</span>
                  <span>val loss: {currentEpoch.val_loss}</span>
                </div>
              </div>
            )}

            {/* Train log */}
            {trainLog.length > 0 && (
              <div ref={trainLogRef} style={{ maxHeight: 200, overflowY: 'auto', background: t.logBg, borderRadius: 7, border: `1px solid ${t.border}`, padding: '8px 12px', fontFamily: 'monospace', fontSize: 12, lineHeight: 1.7 }}>
                {trainLog.map((l, i) => <div key={i} style={{ color: l.color }}>{l.text}</div>)}
              </div>
            )}
          </Card>

          {/* Past runs */}
          {runs.length > 0 && (
            <>
              <SectionHeader title="Training History" t={t} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {runs.map((run, i) => (
                  <div key={run.id}>
                    <div onClick={() => setExpandedRun(expandedRun === run.id ? null : run.id)}
                      style={{ padding: '10px 12px', border: `1px solid ${t.border}`, borderRadius: expandedRun === run.id ? '8px 8px 0 0' : 8, cursor: 'pointer', fontSize: 12, background: t.surface }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                        {i === 0 && <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#16a34a22', color: t.up, border: '1px solid #16a34a44' }}>latest</span>}
                        <span style={{ flex: 1, color: t.muted, fontSize: 10 }}>{run.created_at}</span>
                        <span style={{ color: t.muted, fontSize: 10 }}>{expandedRun === run.id ? '▲' : '▼'}</span>
                      </div>
                      <AccBar value={run.best_val_acc} t={t} />
                      <div style={{ display: 'flex', gap: 10, fontSize: 10, color: t.muted, marginTop: 5 }}>
                        <span>{run.epochs} epochs</span>
                        <span>batch {run.batch_size}</span>
                        <span>lr {run.lr}</span>
                        <span>{run.duration_seconds >= 60 ? `${Math.floor(run.duration_seconds/60)}m${run.duration_seconds%60}s` : `${run.duration_seconds}s`}</span>
                      </div>
                    </div>
                    {expandedRun === run.id && (
                      <div style={{ border: `1px solid ${t.border}`, borderTop: 'none', borderRadius: '0 0 8px 8px', background: t.surface2 }}>
                        <div style={{ padding: '8px 12px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', gap: 4, fontSize: 10, color: t.muted, borderBottom: `1px solid ${t.border}` }}>
                          <span>epoch</span><span>train loss</span><span>train acc</span><span>val loss</span><span>val acc</span>
                        </div>
                        {run.history?.map(h => (
                          <div key={h.epoch} style={{ padding: '6px 12px', display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', gap: 4, fontSize: 11, borderBottom: `1px solid ${t.border}` }}>
                            <span style={{ color: t.muted }}>{h.epoch}/{h.epochs}</span>
                            <span>{h.train_loss}</span>
                            <span style={{ color: h.train_acc >= 0.9 ? t.up : t.text }}>{(h.train_acc*100).toFixed(1)}%</span>
                            <span>{h.val_loss}</span>
                            <span style={{ color: h.val_acc >= 0.9 ? t.up : t.text }}>{(h.val_acc*100).toFixed(1)}%</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ── RIGHT: Models ────────────────────────────────────────────────── */}
        <div style={{ padding: '20px 16px', overflowY: 'auto' }}>
          <SectionHeader title="Trained Models" t={t} />
          {models.length === 0 && <p style={{ color: t.muted, fontSize: 12 }}>No models yet. Train one first.</p>}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {models.map((m, i) => {
              const isDeployed  = m.deployed
              const isDeploying = deploying === m.filename
              return (
                <div key={m.filename} style={{
                  padding: '12px 12px', border: `1px solid ${isDeployed ? '#16a34a44' : t.border}`,
                  borderRadius: 8, background: isDeployed ? '#16a34a0d' : t.surface, fontSize: 12
                }}>
                  {/* Header row */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
                    {isDeployed && (
                      <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#16a34a22', color: t.up, border: '1px solid #16a34a44', fontWeight: 600 }}>● live</span>
                    )}
                    {i === 0 && !isDeployed && (
                      <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#3b82f622', color: t.accent, border: '1px solid #3b82f644' }}>latest</span>
                    )}
                    <span style={{ flex: 1, color: t.muted, fontSize: 10 }}>{m.created_at}</span>
                  </div>

                  {/* Accuracy bar */}
                  {m.best_val_acc != null && (
                    <div style={{ marginBottom: 6 }}>
                      <AccBar value={m.best_val_acc} t={t} />
                    </div>
                  )}

                  {/* Meta */}
                  <div style={{ display: 'flex', gap: 8, fontSize: 10, color: t.muted, marginBottom: 8, flexWrap: 'wrap' }}>
                    {m.epochs && <span>{m.epochs} epochs</span>}
                    {m.batch_size && <span>batch {m.batch_size}</span>}
                    {m.dataset?.cat && <span>🐱 {m.dataset.cat} / ✗ {m.dataset.not_cat}</span>}
                    {m.duration_seconds && <span>{m.duration_seconds >= 60 ? `${Math.floor(m.duration_seconds/60)}m${m.duration_seconds%60}s` : `${m.duration_seconds}s`}</span>}
                  </div>

                  {/* Deploy button */}
                  {!isDeployed && (
                    <button onClick={() => deployModel(m.filename)} disabled={isDeploying}
                      style={{ width: '100%', padding: '6px 0', fontSize: 11, borderRadius: 6, border: 'none', background: '#059669', color: '#fff', cursor: isDeploying ? 'not-allowed' : 'pointer', opacity: isDeploying ? 0.5 : 1 }}>
                      {isDeploying ? '⟳ Deploying...' : '🚀 Deploy'}
                    </button>
                  )}
                  {isDeployed && (
                    <div style={{ fontSize: 10, color: t.up, textAlign: 'center', padding: '4px 0' }}>
                      ✓ Currently serving the classifier
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}