import { useEffect, useState } from 'react'

const API = '/api'

export default function App() {
  const [health, setHealth] = useState(null)

  const fetchHealth = async () => {
    try { setHealth(await (await fetch(`${API}/health`)).json()) } catch { setHealth(null) }
  }

  useEffect(() => {
    fetchHealth()
    const iv = setInterval(fetchHealth, 10000)
    return () => clearInterval(iv)
  }, [])

  return (
    <div style={{ fontFamily: 'system-ui', background: '#0a0a0b', color: '#f0f0f0', minHeight: '100vh', padding: 40 }}>
      <h1>Cat Recognition — Dashboard</h1>
      <pre style={{ color: '#666', fontSize: 12 }}>{JSON.stringify(health, null, 2)}</pre>
      <p style={{ color: '#666' }}>Dashboard coming in Phase 1</p>
    </div>
  )
}