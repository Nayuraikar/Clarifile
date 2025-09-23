import React, { useEffect, useMemo, useState } from 'react'

const BASE = 'http://127.0.0.1:4000'

async function call(path: string, opts?: RequestInit) {
  const res = await fetch(BASE + path, opts)
  if (res.status === 204) return { ok: true }
  const text = await res.text()
  try { return JSON.parse(text) } catch { return { error: 'Invalid JSON', raw: text } }
}

type Proposal = { id: number; file: string; proposed: string; final?: string }
type DriveProposal = { id: string; name: string; proposed_category: string }

function Section({ children }: { children: React.ReactNode }) {
  return <div className="max-w-7xl mx-auto px-6 py-8">{children}</div>
}

function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: 'primary' | 'secondary' | 'accent' | 'success' }) {
  const tone = props.tone || 'secondary'
  const cls = useMemo(() => ({
    primary: 'bg-gradient-to-r from-[var(--sand-200)] to-[var(--pearl-400)] text-[var(--charcoal-900)]',
    secondary: 'bg-white/10 text-[var(--sand-50)] border border-[color:var(--pearl-400)]/40',
    accent: 'bg-gradient-to-r from-[var(--pearl-400)] to-[var(--sand-200)] text-[var(--charcoal-900)]',
    success: 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-white'
  })[tone], [tone])
  const { className, children, ...rest } = props
  return <button className={`px-4 py-2 rounded-xl shadow transition hover:scale-[1.02] ${cls} ${className||''}`} {...rest}>{children}</button>
}

export default function App() {
  const [tab, setTab] = useState<'dashboard'|'drive'|'dups'|'cats'|'ai'>('dashboard')
  const [isScanning, setIsScanning] = useState(false)
  const [proposals, setProposals] = useState<Proposal[]>([])
  const [driveProps, setDriveProps] = useState<DriveProposal[]>([])
  const [dups, setDups] = useState<any[]>([])
  const [cats, setCats] = useState<any[]>([])
  const [askInput, setAskInput] = useState('')
  const [askResult, setAskResult] = useState<any>(null)
  const [driveAnalyzedId, setDriveAnalyzedId] = useState<string | null>(null)
  const [messages, setMessages] = useState<{ role: 'user'|'assistant', content: string }[]>([])
  const inputRef = React.useRef<HTMLInputElement>(null)

  async function refreshDrive() {
    const list: DriveProposal[] = await call('/drive/proposals')
    setDriveProps(Array.isArray(list) ? list : [])
    await refreshCats()
    await refreshDups()
  }

  async function refreshDups() {
    const d = await call('/duplicates')
    const groups = (d.duplicates||[]).map((g:any)=>({
      ...g,
      files: Array.from(new Map((g.files||[]).map((f:any)=>[String(f.id), f])).values())
    })).filter((g:any)=> (g.files||[]).length >= 2)
    setDups(groups)
  }

  async function refreshCats() {
    let c = await call('/drive/categories')
    if (!Array.isArray(c) || c.length===0) c = await call('/categories')
    setCats(Array.isArray(c) ? c : [])
  }

  // New: Scan Drive via the extension pipeline (organize with move:false) to populate proposals
  async function doScan() {
    setIsScanning(true)
    try {
      // Ask gateway for existing proposals; if empty, tell user to use the extension to list files
      await refreshDrive()
      if (driveProps.length === 0) {
        alert('Use the Chrome extension popup: Authorize → Organize Drive Files, then click Refresh Drive Proposals here.')
      }
    } finally { setIsScanning(false) }
  }

  useEffect(()=>{ /* initial load */ (async()=>{ await refreshDrive() })() }, [])

  async function sendMsg(){
    const q = askInput.trim(); if(!q) return;
    setMessages(prev => [...prev, { role:'user', content: q }])
    setAskInput('')
    // typing indicator
    setMessages(prev => [...prev, { role:'assistant', content: '::typing::' }])
    const updateLast = (text:string)=> setMessages(prev => { const arr=[...prev]; arr[arr.length-1] = { role:'assistant', content: text }; return arr })
    try {
      if (driveAnalyzedId) {
        const r = await call('/drive/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ file:{ id:driveAnalyzedId, name:'', mimeType:'', parents:[] }, q }) })
        if ((r as any)?.error) { updateLast(`Error: ${(r as any).error}`); return }
        const ans = (r as any)?.qa?.answer || (r as any)?.summary || '(no answer)'
        updateLast(ans)
        setAskResult((r as any)?.qa || { answer: ans })
        return
      }
      const ids = (proposals||[]).slice(0,1).map(p=>p.id)
      if (ids.length===0) { updateLast('No document selected. Use Drive tab to analyze a file first.'); return }
      const r = await fetch(`${BASE}/ask?file_id=${ids[0]}&q=${encodeURIComponent(q)}`)
      const text = await r.text()
      let j:any; try { j = JSON.parse(text) } catch { j = { error: text } }
      const ans = j.answer || j.error || '(no answer)'
      updateLast(ans)
      setAskResult(j)
    } catch (e:any) {
      updateLast(`Error: ${e?.message || String(e)}`)
    }
  }

  return (
    <div className="min-h-screen text-[var(--sand-50)]" style={{ background: 'linear-gradient(135deg, var(--charcoal-900) 0%, var(--mink-500) 50%, var(--charcoal-900) 100%)' }}>
      <header className="sticky top-0 z-10 bg-[var(--charcoal-900)]/70 backdrop-blur border-b border-[var(--pearl-400)]/20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-[var(--sand-200)] to-[var(--pearl-400)] flex items-center justify-center text-[var(--charcoal-900)] font-bold">CF</div>
            <div>
              <div className="text-2xl font-extrabold">ClarifFile</div>
              <div className="text-xs text-[var(--pearl-400)]">DOCUMENT INTELLIGENCE</div>
            </div>
          </div>
          <nav className="flex gap-2 text-[var(--pearl-400)]">
            {[
              ['dashboard','Dashboard'],
              ['drive','Files'],
              ['cats','Categories'],
              ['dups','Duplicates'],
              ['ai','AI Assistant']
            ].map(([id,label])=> (
              <button key={id} onClick={()=>setTab(id as any)} className={`px-4 py-2 rounded-xl transition ${tab===id?'bg-gradient-to-r from-[var(--sand-200)] to-[var(--pearl-400)] text-[var(--charcoal-900)]':'hover:bg-white/10'}`}>{label}</button>
            ))}
          </nav>
        </div>
      </header>

      {tab==='dashboard' && (
        <Section>
          <div className="text-center py-10">
            <h2 className="text-5xl font-extrabold mb-4">Organize Your Digital Life</h2>
            <p className="text-[var(--pearl-400)] mb-8">AI-powered file organization and intelligent document analysis</p>
            <Button tone='primary' onClick={doScan} disabled={isScanning} className="px-8 py-4 text-lg">{isScanning?'Scanning...':'Start Smart Scan'}</Button>
          </div>
          <h3 className="text-2xl font-bold mb-4">Document Proposals</h3>
          <div className="grid gap-4 md:grid-cols-2">
            {proposals.map(p=> (
              <div key={p.id} className="glass p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{p.file}</div>
                    <div className="text-sm text-[var(--pearl-400)]">Proposed: {p.proposed}</div>
                  </div>
                  <div className="text-sm">{p.final? 'Approved' : 'Proposed'}</div>
                </div>
                <div className="mt-3 flex gap-2 flex-wrap">
                  <Button tone='success' onClick={async()=>{ await call('/approve',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_id:p.id, final_label:p.proposed})}); }}>Approve</Button>
                  <Button tone='secondary' onClick={async()=>{
                    const r = await fetch(`${BASE}/file_summary?file_id=${p.id}`); if(r.status===200){ const j = await r.json(); alert(j.summary||'(no summary)') } else alert('No summary available')
                  }}>View Summary</Button>
                  <Button tone='accent' onClick={async()=>{
                    const r = await fetch(`${BASE}/file_entities?file_id=${p.id}`); const j = await r.json(); alert((j||[]).map((e:any)=>`${e.name} (${e.type} x${e.count})`).join(' • ')||'(no tags)')
                  }}>Show Tags</Button>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {tab==='drive' && (
        <Section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-2xl font-bold">Google Drive Proposals</h3>
            <Button onClick={refreshDrive}>Refresh Drive Proposals</Button>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            {driveProps.map(f=> (
              <div key={f.id} className="glass p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{f.name}</div>
                    <div className="text-sm text-[var(--pearl-400)]">Proposed: {f.proposed_category||'Other'}</div>
                  </div>
                  <div className="text-xs">Proposed</div>
                </div>
                <div className="mt-3 flex gap-2 flex-wrap">
                  <Button tone='success' onClick={async()=>{
                    const r = await call('/drive/approve', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ file: { id:f.id, name:f.name, mimeType:'', parents:[] }, category:f.proposed_category }) })
                    if ((r as any)?.ok) await refreshDrive(); else alert((r as any)?.error||'Move failed')
                  }}>Approve & Move</Button>
                  <Button tone='secondary' onClick={async()=>{
                    const r = await call('/drive/analyze', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ file:{ id:f.id, name:f.name, mimeType:'', parents:[] } }) })
                    setDriveAnalyzedId(f.id)
                    const summary = (r as any)?.summary || '(no summary)'
                    setMessages(prev => [...prev, { role: 'assistant', content: summary }])
                    setTab('ai')
                    setTimeout(()=> inputRef.current?.focus(), 0)
                  }}>View Summary</Button>
                  <Button tone='accent' onClick={async()=>{
                    // Route to AI; let user type the question there
                    setDriveAnalyzedId(f.id)
                    setTab('ai')
                    setTimeout(()=> inputRef.current?.focus(), 0)
                  }}>Ask</Button>
                </div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {tab==='dups' && (
        <Section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-2xl font-bold">Duplicate Files</h3>
            <Button onClick={refreshDups}>Refresh</Button>
          </div>
          <div className="space-y-4">
            {dups.map((g:any, idx:number)=> (
              <div key={idx} className="glass p-4">
                <div className="font-semibold mb-2">Group {g.group_id||''} ({g.files.length} files)</div>
                <ul className="text-[var(--pearl-400)] list-disc ml-5">
                  {(g.files||[]).map((f:any)=> <li key={f.id}>{f.name}</li>)}
                </ul>
                {g.files.length>=2 && (
                  <div className="mt-3 flex gap-2 flex-wrap">
                    <Button tone='success' onClick={async()=>{
                      const a=g.files[0], b=g.files[1]; await call('/resolve_duplicate',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_a:a.id, file_b:b.id, action:'keep_a'})}); await refreshDups()
                    }}>Keep {g.files[0].name}</Button>
                    <Button tone='success' onClick={async()=>{
                      const a=g.files[0], b=g.files[1]; await call('/resolve_duplicate',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_a:a.id, file_b:b.id, action:'keep_b'})}); await refreshDups()
                    }}>Keep {g.files[1].name}</Button>
                    <Button tone='accent' onClick={async()=>{
                      const a=g.files[0], b=g.files[1]; await call('/resolve_duplicate',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_a:a.id, file_b:b.id, action:'keep_both'})}); await refreshDups()
                    }}>Keep Both</Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {tab==='cats' && (
        <Section>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-2xl font-bold">Smart Categories</h3>
            <Button onClick={refreshCats}>Refresh</Button>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {cats.map((c:any)=> (
              <div key={c.name} className="glass p-6 text-center">
                <div className="text-xl font-bold mb-1">{c.name}</div>
                <div className="text-4xl font-extrabold text-[var(--sand-200)]">{c.file_count}</div>
                <div className="text-sm text-[var(--pearl-400)]">files</div>
              </div>
            ))}
          </div>
        </Section>
      )}

      {tab==='ai' && (
        <Section>
          <h3 className="text-2xl font-bold mb-4">AI Document Assistant</h3>
          <div className="glass p-0 overflow-hidden">
            <div className="h-[420px] overflow-y-auto px-4 py-6 space-y-4 bg-gradient-to-b from-white/5 to-white/0">
              {messages.map((m, i)=> (
                <div key={i} className={`max-w-3xl ${m.role==='user'?'ml-auto text-right':'mr-auto text-left'}`}>
                  <div className={`${m.role==='user'?'bg-gradient-to-r from-[var(--sand-200)] to-[var(--pearl-400)] text-[var(--charcoal-900)]':'bg-white/8 text-[var(--sand-50)]'} inline-block px-4 py-3 rounded-2xl shadow`}>
                    {m.content === '::typing::' ? (
                      <span className="typing"><span className="dot"></span><span className="dot"></span><span className="dot"></span></span>
                    ) : m.content}
                  </div>
                </div>
              ))}
              {askResult && (
                <div className="max-w-3xl mr-auto">
                  <div className="bg-white/8 text-[var(--sand-50)] inline-block px-4 py-3 rounded-2xl shadow">
                    {askResult.answer || askResult.error || '(no answer)'}
                  </div>
                </div>
              )}
            </div>
            <div className="border-t border-[var(--pearl-400)]/20 p-3 flex gap-2 bg-[var(--charcoal-900)]/50">
              <input ref={inputRef} value={askInput} onChange={e=>setAskInput(e.target.value)} onKeyDown={(e)=>{ if(e.key==='Enter') sendMsg() }} placeholder="Ask anything about this document..." className="flex-1 px-4 py-3 rounded-xl bg-white/5 border border-[var(--pearl-400)]/30 outline-none" />
              <Button tone='primary' onClick={sendMsg}>Send</Button>
            </div>
          </div>
        </Section>
      )}
    </div>
  )
}


