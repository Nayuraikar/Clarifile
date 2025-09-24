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
  return <div className="max-w-7xl mx-auto px-6 py-12">{children}</div>
}

function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: 'primary' | 'secondary' | 'accent' | 'success', loading?: boolean }) {
  const tone = props.tone || 'secondary'
  const cls = useMemo(() => ({
    primary: 'professional-btn-primary',
    secondary: 'professional-btn-secondary',
    accent: 'professional-btn-secondary',
    success: 'professional-btn-primary'
  })[tone], [tone])
  const { className, children, loading, ...rest } = props
  return <button className={`professional-btn ${cls} ${className||''}`} {...rest}>
    {loading ? (
      <div className="flex items-center gap-2">
        <div className="professional-spinner"></div>
        Loading...
      </div>
    ) : children}
  </button>
}

export default function App() {
  const [tab, setTab] = useState<'dashboard'|'drive'|'dups'|'cats'|'ai'>('dashboard')
  const [isScanning, setIsScanning] = useState(false)
  const [scanProgress, setScanProgress] = useState(0)
  const [scanStatus, setScanStatus] = useState('')
  const [proposals, setProposals] = useState<Proposal[]>([])
  const [driveProps, setDriveProps] = useState<DriveProposal[]>([])
  const [dups, setDups] = useState<any[]>([])
  const [cats, setCats] = useState<any[]>([])
  const [askInput, setAskInput] = useState('')
  const [askResult, setAskResult] = useState<any>(null)
  const [driveAnalyzedId, setDriveAnalyzedId] = useState<string | null>(null)
  const [analyzingFile, setAnalyzingFile] = useState<string | null>(null)
  const [notification, setNotification] = useState<string | null>(null)
  const [messages, setMessages] = useState<{ role: 'user'|'assistant', content: string }[]>([])
  const [loadingStates, setLoadingStates] = useState<{[key: string]: boolean}>({})
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [showDocumentSelector, setShowDocumentSelector] = useState(false)
  const [currentQuickAction, setCurrentQuickAction] = useState<string | null>(null)
  const [quickActionLoading, setQuickActionLoading] = useState(false)
  const [duplicateResolution, setDuplicateResolution] = useState<{ [key: string]: boolean }>({})
  const [duplicateResolutionLoading, setDuplicateResolutionLoading] = useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)

  // Helper function to set loading state for specific operations
  const setLoading = (operation: string, loading: boolean) => {
    setLoadingStates(prev => ({ ...prev, [operation]: loading }))
  }

  // Helper function to check if an operation is loading
  const isLoading = (operation: string) => loadingStates[operation] || false

  async function refreshDrive() {
    setLoading('refreshDrive', true)
    try {
      const list: DriveProposal[] = await call('/drive/proposals')
      setDriveProps(Array.isArray(list) ? list : [])
      await refreshCats()
      await refreshDups()
    } finally {
      setLoading('refreshDrive', false)
    }
  }

  async function refreshDups() {
    setLoading('refreshDups', true)
    try {
      const d = await call('/duplicates')
      const groups = (d.duplicates||[]).map((g:any)=>({
        ...g,
        files: Array.from(new Map((g.files||[]).map((f:any)=>[String(f.id), f])).values())
      })).filter((g:any)=> (g.files||[]).length >= 2)
      setDups(groups)
    } finally {
      setLoading('refreshDups', false)
    }
  }

  async function refreshCats() {
    setLoading('refreshCats', true)
    try {
      let c = await call('/drive/categories')
      if (!Array.isArray(c) || c.length===0) c = await call('/categories')
      const categoriesWithFiles = c.map(category => ({
        ...category,
        count: driveProps.filter(file => file.proposed_category === category.name).length
      }))
      setCats(categoriesWithFiles)
    } finally {
      setLoading('refreshCats', false)
    }
  }

  // New: Scan Drive via the extension pipeline (organize with move:false) to populate proposals
  async function handleScan() {
    setIsScanning(true)
    setScanProgress(0)
    setScanStatus('Scanning...')
    try {
      // Ask gateway for existing proposals; if empty, tell user to use the extension to list files
      await refreshDrive()
      if (driveProps.length === 0) {
        alert('Use the Chrome extension popup: Authorize → Organize Drive Files, then click Refresh Drive Proposals here.')
      }
    } finally { 
      setIsScanning(false) 
      setScanProgress(100)
      setScanStatus('Scan complete')
    }
  }

  useEffect(() => {
    refreshDrive()
  }, [])

  async function sendMsg(){
    const q = askInput.trim(); 
    if(!q) return;
    
    // Add user message
    setMessages(prev => [...prev, { role:'user', content: q }])
    setAskInput('')
    
    // Add typing indicator
    setMessages(prev => [...prev, { role:'assistant', content: '::typing::' }])
    
    // Helper function to update the last message
    const updateLast = (text:string)=> setMessages(prev => { 
      const arr=[...prev]; 
      arr[arr.length-1] = { role:'assistant', content: text }; 
      return arr 
    })
    
    try {
      let response: any;
      
      // Check if we have a specific drive file to analyze
      if (driveAnalyzedId) {
        response = await call('/drive/analyze', { 
          method:'POST', 
          headers:{'Content-Type':'application/json'}, 
          body: JSON.stringify({ 
            file:{ id:driveAnalyzedId, name:'', mimeType:'', parents:[] }, 
            q 
          }) 
        })
        
        if (response?.error) { 
          updateLast(`Error: ${response.error}`)
          return 
        }
        
        const ans = response?.qa?.answer || response?.summary || 'No answer available'
        updateLast(ans)
        setAskResult(response?.qa || { answer: ans })
        
      } else {
        // Check if we have any proposals to work with
        const ids = (proposals||[]).slice(0,1).map(p=>p.id)
        
        if (ids.length===0) { 
          updateLast('No document selected. Please go to the Drive tab and select a file to analyze first.')
          return 
        }
        
        // Use the consistent call helper function
        response = await call(`/ask?file_id=${ids[0]}&q=${encodeURIComponent(q)}`)
        
        if (response?.error) { 
          updateLast(`Error: ${response.error}`)
          return 
        }
        
        const ans = response?.answer || response?.error || 'No answer available'
        updateLast(ans)
        setAskResult(response)
      }
      
    } catch (e:any) {
      console.error('Chatbot error:', e)
      updateLast(`Error: ${e?.message || 'Failed to get response from AI assistant. Please try again.'}`)
    }
  }

  const handleQuickAction = (action: string) => {
    console.log('Quick Action clicked:', action)
    
    // Check if there are documents available
    if (driveProps.length === 0) {
      setNotification('No documents available. Please refresh files first.')
      return
    }
    
    console.log('Available documents:', driveProps.length)
    
    // Set the current action and show the document selector
    setCurrentQuickAction(action)
    setShowDocumentSelector(true)
    setQuickActionLoading(true)
  }

  const performQuickAction = async (action: string, file: DriveProposal) => {
    console.log('🚀 Performing quick action:', action)
    console.log('File:', file.name, file.id)
    
    // Set loading state for the quick action
    setQuickActionLoading(true)
    setNotification(`Performing ${action.replace('_', ' ')} on "${file.name}"...`)
    
    try {
      let response: any;
      
      // Add timeout to prevent infinite loading
      const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Request timeout - backend endpoint may not exist')), 10000);
      });
      
      switch (action) {
        case 'summarize':
          response = await Promise.race([
            call('/drive/summarize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                file: { 
                  id: file.id, 
                  name: file.name, 
                  mimeType: '', 
                  parents: [] 
                } 
              })
            }),
            timeoutPromise
          ]);
          break;
        case 'find_similar':
          response = await Promise.race([
            call('/drive/find_similar', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                file: { 
                  id: file.id, 
                  name: file.name, 
                  mimeType: '', 
                  parents: [] 
                } 
              })
            }),
            timeoutPromise
          ]);
          break;
        case 'extract_insights':
          response = await Promise.race([
            call('/drive/extract_insights', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                file: { 
                  id: file.id, 
                  name: file.name, 
                  mimeType: '', 
                  parents: [] 
                } 
              })
            }),
            timeoutPromise
          ]);
          break;
        case 'organize':
          response = await Promise.race([
            call('/drive/organize', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ 
                file: { 
                  id: file.id, 
                  name: file.name, 
                  mimeType: '', 
                  parents: [] 
                } 
              })
            }),
            timeoutPromise
          ]);
          break;
        default:
          console.error('Unknown quick action:', action)
          setQuickActionLoading(false)
          return;
      }
      
      if (response?.error) {
        console.error('Error performing quick action:', response.error)
        setNotification(`Error: ${response.error}`)
        setQuickActionLoading(false)
        return;
      }
      
      console.log('Quick action response:', response)
      setNotification(`${action.replace('_', ' ')} completed successfully!`)
      
      // Add the result to the AI Assistant chat
      if (response) {
        let messageContent = ''
        
        switch (action) {
          case 'summarize':
            messageContent = `I've summarized the document "${file.name}". Here's the summary:\n\n📄 **Summary:** ${response.summary || 'No summary available'}\n🏷️ **Category:** ${response.category || 'General'}\n🏷️ **Tags:** ${response.tags?.join(', ') || 'None'}`
            break;
          case 'find_similar':
            messageContent = `I've found similar files to "${file.name}". Here are the results:\n\n📄 **Similar Files:** ${response.similar_files?.map((f: any) => f.name).join(', ') || 'No similar files found'}\n🔍 **Match Score:** ${response.match_score || 'N/A'}`
            break;
          case 'extract_insights':
            messageContent = `I've extracted insights from "${file.name}". Here's what I found:\n\n💡 **Key Insights:** ${response.insights?.join('\n') || 'No insights extracted'}\n📊 **Analysis:** ${response.analysis || 'No analysis available'}`
            break;
          case 'organize':
            messageContent = `I've organized the file "${file.name}". Here's what happened:\n\n🗂️ **Target Folder:** ${response.target_folder || 'Default'}\n📂 **Category:** ${response.category || 'General'}\n✅ **Status:** ${response.status || 'Organized successfully'}`
            break;
        }
        
        const resultMessage = {
          role: 'assistant' as const,
          content: messageContent
        };
        setMessages(prev => [...prev, resultMessage]);
      }
      
    } catch (e: any) {
      console.error('Error performing quick action:', e)
      
      // Check if it's a network error or backend not available
      if (e.message.includes('timeout') || e.message.includes('Network') || e.message.includes('Failed to fetch')) {
        setNotification(`Backend endpoint not available yet. The ${action.replace('_', ' ')} feature will be implemented soon!`)
      } else {
        setNotification(`Error: ${e?.message || 'Failed to perform quick action. Please try again.'}`)
      }
    } finally {
      setQuickActionLoading(false)
    }
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Professional Background Effects */}
      <div className="professional-shapes">
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
      </div>
      
      {/* Professional Particle System */}
      <div className="professional-particles" id="professional-particles"></div>
      
      {/* Professional Wave Effects */}
      <div className="professional-waves">
        <div className="professional-wave"></div>
        <div className="professional-wave"></div>
        <div className="professional-wave"></div>
      </div>
      
      {/* Professional Mouse Trail */}
      <div className="professional-mouse-trail" id="professional-mouse-trail"></div>
      
      {/* Main content with higher z-index to appear above background */}
      <div className="relative z-20">
        <header className="sticky top-0 z-30 professional-glass border-b border-white/20">
          <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-accent-primary to-accent-secondary flex items-center justify-center text-white font-bold text-xl shadow-lg professional-interactive">
                CF
              </div>
              <div>
                <div className="text-3xl font-extrabold text-gradient">Clarifile</div>
                <div className="text-sm text-text-muted font-medium tracking-wide">DOCUMENT INTELLIGENCE PLATFORM</div>
              </div>
            </div>
            <nav className="flex gap-2">
              {[
                ['dashboard','Dashboard'],
                ['drive','Files'],
                ['cats','Categories'],
                ['dups','Duplicates'],
                ['ai','AI Assistant']
              ].map(([id,label])=> (
                <button 
                  key={id} 
                  onClick={()=>setTab(id as any)} 
                  className={`professional-nav-item ${tab===id?'active':''}`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        <div className="page-transition">
          {tab==='dashboard' && (
            <Section>
              {/* Enhanced Hero Section */}
              <div className="text-center py-20 fade-in relative">
                <div className="absolute inset-0 bg-gradient-to-r from-accent-primary/10 to-accent-secondary/10 rounded-3xl blur-3xl"></div>
                <div className="relative z-10">
                  <h1 className="hero-heading mb-8 text-pop">Welcome to Clarifile</h1>
                  <p className="body-text text-lg mb-16 max-w-4xl mx-auto text-gradient">
                    Transform your digital chaos into organized perfection with AI-powered document management and intelligent file organization
                  </p>
                  
                  {/* Enhanced Action Cards */}
                  <div className="grid gap-8 md:grid-cols-3 max-w-5xl mx-auto mb-16">
                    <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.1s'}}>
                      <div className="text-5xl mb-6">📁</div>
                      <h3 className="card-heading mb-3">Smart Organization</h3>
                      <p className="muted-text">AI-powered categorization automatically organizes your files into intelligent folders</p>
                    </div>
                    <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.2s'}}>
                      <div className="text-5xl mb-6">🔍</div>
                      <h3 className="card-heading mb-3">Intelligent Search</h3>
                      <p className="muted-text">Find any document instantly with advanced semantic search and content analysis</p>
                    </div>
                    <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.3s'}}>
                      <div className="text-5xl mb-6">⚡</div>
                      <h3 className="card-heading mb-3">Lightning Fast</h3>
                      <p className="muted-text">Process thousands of files in seconds with our optimized AI engine</p>
                    </div>
                  </div>
                  
                  {/* Enhanced Main Action Button */}
                  <Button tone='primary' onClick={handleScan} disabled={isScanning} loading={isScanning} className="text-lg px-16 py-4 text-pop">
                    {isScanning ? (
                      <div className="flex items-center gap-3">
                        <div className="professional-spinner"></div>
                        <span>Scanning Your Files...</span>
                      </div>
                    ) : (
                      <span className="flex items-center gap-3">
                        Start Smart Scan
                        <span className="text-2xl">🚀</span>
                      </span>
                    )}
                  </Button>
                  
                  {/* Enhanced Progress Display */}
                  {isScanning && (
                    <div className="mt-12 max-w-2xl mx-auto fade-in">
                      <div className="professional-card">
                        <div className="flex items-center gap-4 mb-6">
                          <div className="w-6 h-6 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary animate-pulse"></div>
                          <div className="card-heading">{scanStatus}</div>
                        </div>
                        <div className="professional-progress mb-4">
                          <div 
                            className="professional-progress-bar"
                            style={{width: `${scanProgress}%`}}
                          ></div>
                        </div>
                        <div className="text-sm text-text-muted text-center">
                          {scanProgress}% Complete
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
              
              {/* Enhanced Stats Section */}
              <div className="grid gap-6 md:grid-cols-4 mb-16">
                <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.4s'}}>
                  <div className="text-4xl font-bold text-gradient mb-3">{proposals.length}</div>
                  <div className="muted-text">Document Proposals</div>
                </div>
                <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.5s'}}>
                  <div className="text-4xl font-bold text-gradient mb-3">{driveProps.length}</div>
                  <div className="muted-text">Drive Files</div>
                </div>
                <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.6s'}}>
                  <div className="text-4xl font-bold text-gradient mb-3">{dups.length}</div>
                  <div className="muted-text">Duplicate Groups</div>
                </div>
                <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.7s'}}>
                  <div className="text-4xl font-bold text-gradient mb-3">{cats.length}</div>
                  <div className="muted-text">Categories</div>
                </div>
              </div>
              
              {/* Enhanced Document Proposals Section */}
              <div className="fade-in" style={{animationDelay: '0.8s'}}>
                <div className="flex items-center justify-between mb-10">
                  <h2 className="section-heading">Recent Document Proposals</h2>
                  <Button tone='secondary'>
                    View All
                  </Button>
                </div>
                
                {proposals.length > 0 ? (
                  <div className="grid gap-6 md:grid-cols-2">
                    {proposals.slice(0, 4).map((p, index) => (
                      <div key={p.id} className="professional-card professional-interactive fade-in" style={{animationDelay: `${0.9 + index * 0.1}s`}}>
                        <div className="flex items-start justify-between mb-6">
                          <div className="flex-1">
                            <div className="card-heading mb-3">{p.file}</div>
                            <div className="muted-text">
                              <span className="inline-flex items-center gap-2">
                                <span>📂</span>
                                Proposed: {p.proposed}
                              </span>
                            </div>
                          </div>
                          <div className="text-sm px-4 py-2 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary text-white text-accent-primary border border-accent-primary/30">
                            {p.final ? (
                              <span className="flex items-center gap-2">
                                <span>✅</span>
                                Approved
                              </span>
                            ) : (
                              <span className="flex items-center gap-2">
                                <span>📋</span>
                                Proposed
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="mt-8 flex gap-4 flex-wrap">
                          <Button tone='success' onClick={async()=>{ 
                            await call('/approve',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_id:p.id, final_label:p.proposed})}); 
                          }}>
                            <span className="flex items-center gap-2">
                              <span>✓</span>
                              Approve
                            </span>
                          </Button>
                          <Button tone='secondary' onClick={async()=>{
                            const r = await fetch(`${BASE}/file_summary?file_id=${p.id}`); 
                            if(r.status===200){ 
                              const j = await r.json(); 
                              alert(`Summary: ${j.summary}`); 
                            }
                          }}>
                            <span className="flex items-center gap-2">
                              <span>📄</span>
                              View Summary
                            </span>
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6">📂</div>
                    <h3 className="card-heading mb-4">No Document Proposals Yet</h3>
                    <p className="muted-text mb-8">Start by scanning your files to get intelligent organization suggestions</p>
                    <Button tone='primary' onClick={handleScan}>
                      Start Your First Scan
                    </Button>
                  </div>
                )}
              </div>
            </Section>
          )}

          {tab==='drive' && (
            <Section>
              <div className="fade-in">
                <div className="flex items-center justify-between mb-10">
                  <h2 className="section-heading">Drive Files</h2>
                  <div className="flex gap-4">
                    <Button tone='secondary' onClick={refreshDrive} disabled={isLoading('refreshDrive')} loading={isLoading('refreshDrive')}>
                      Refresh Files
                    </Button>
                    <Button tone='primary' onClick={handleScan} disabled={isScanning} loading={isScanning}>
                      Scan Drive
                    </Button>
                  </div>
                </div>
                
                {driveProps.length > 0 ? (
                  <div className="grid gap-6">
                    {driveProps.map((file, index) => (
                      <div key={file.id} className={`professional-card professional-interactive fade-in ${driveAnalyzedId === file.id ? 'ring-2 ring-accent-primary bg-accent-primary/10' : ''}`} style={{animationDelay: `${index * 0.1}s`}}>
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <div className="card-heading">{file.name}</div>
                              {driveAnalyzedId === file.id && (
                                <div className="px-3 py-1 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary text-white text-sm font-medium">
                                  ✓ Selected for AI
                                </div>
                              )}
                            </div>
                            <div className="muted-text">
                              <span className="inline-flex items-center gap-2">
                                <span>📁</span>
                                Proposed Category: {file.proposed_category}
                              </span>
                            </div>
                          </div>
                          <div className="flex gap-3">
                            <Button tone='secondary' onClick={() => {
                              console.log('ANALYZE BUTTON CLICKED!');
                              console.log('File:', file.name, file.id);
                              console.log('Current loading state:', isLoading('analyzeFile'));
                              console.log('Button disabled:', isLoading('analyzeFile'));
                              console.log('Button loading:', isLoading('analyzeFile'));
                              console.log('Drive analyzed ID:', driveAnalyzedId);
                              console.log('=== BUTTON CLICK DEBUG START ===');
                              setLoading('analyzeFile', true);
                              console.log('Set loading to true');
                              setNotification('Analyzing file...');
                              console.log('Set notification to Analyzing file...');
                              
                              // Actually call the backend to analyze the file
                              call('/drive/analyze', {
                                method: 'POST',
                                headers: {
                                  'Content-Type': 'application/json',
                                },
                                body: JSON.stringify({
                                  file: { 
                                    id: file.id, 
                                    name: file.name, 
                                    mimeType: '', 
                                    parents: [] 
                                  },
                                  q: 'Analyze this document and provide a summary'
                                })
                              }).then(response => {
                                console.log('Backend response:', response);
                                console.log('Set drive analyzed ID to:', file.id);
                                setDriveAnalyzedId(file.id);
                                setNotification('File analyzed successfully!');
                                
                                // Add the analyzed file to the AI Assistant context
                                if (response.summary) {
                                  const analysisMessage = {
                                    role: 'assistant' as const,
                                    content: `I've analyzed the file "${file.name}". Here's what I found:\n\n📄 **Summary:** ${response.summary}\n🏷️ **Category:** ${response.category || 'General'}\n🏷️ **Tags:** ${response.tags?.join(', ') || 'None'}\n\nYou can now ask me questions about this file!`
                                  };
                                  setMessages(prev => [...prev, analysisMessage]);
                                  console.log('Added analysis to AI Assistant');
                                }
                                
                                setLoading('analyzeFile', false);
                                console.log('Set loading to false');
                                console.log('=== BUTTON CLICK DEBUG END ===');
                              }).catch(error => {
                                console.error('Error analyzing file:', error);
                                setNotification('Error analyzing file. Please try again.');
                                setLoading('analyzeFile', false);
                                console.log('Set loading to false due to error');
                                console.log('=== BUTTON CLICK DEBUG END ===');
                              });
                            }} disabled={isLoading('analyzeFile')} loading={isLoading('analyzeFile')}>
                              {driveAnalyzedId === file.id ? 'Selected' : 'Analyze'}
                            </Button>
                            <Button tone='success' onClick={async () => {
                              await call('/drive/approve', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ 
                                  file: { 
                                    id: file.id, 
                                    name: file.name, 
                                    mimeType: '', 
                                    parents: [] 
                                  }, 
                                  final_label: file.proposed_category 
                                })
                              });
                            }}>
                              Approve
                            </Button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6">📁</div>
                    <h3 className="card-heading mb-4">No Drive Files Found</h3>
                    <p className="muted-text mb-8">Connect your Google Drive and scan your files to get started</p>
                    <Button tone='primary' onClick={handleScan}>
                      Connect & Scan Drive
                    </Button>
                  </div>
                )}
              </div>
            </Section>
          )}

          {tab==='dups' && (
            <Section>
              <div className="fade-in">
                <div className="flex items-center justify-between mb-10">
                  <h2 className="section-heading">Duplicate Files</h2>
                  <Button tone='secondary' onClick={refreshDups} disabled={isLoading('refreshDups')} loading={isLoading('refreshDups')}>
                    Refresh Duplicates
                  </Button>
                </div>
                
                {dups.length > 0 ? (
                  <div className="grid gap-6">
                    {dups.map((group, index) => (
                      <div key={index} className="professional-card professional-interactive fade-in" style={{animationDelay: `${index * 0.1}s`}}>
                        <div className="card-heading mb-4">Duplicate Group {index + 1}</div>
                        <div className="space-y-3">
                          {group.files.map((file: any, fileIndex: number) => (
                            <div key={file.id} className="flex items-center justify-between p-4 bg-bg-secondary rounded-xl">
                              <div className="flex items-center gap-3">
                                <span className="text-2xl">📄</span>
                                <div>
                                  <div className="font-medium text-text-primary">{file.name}</div>
                                  <div className="text-sm text-text-muted">{file.size} bytes</div>
                                </div>
                              </div>
                              <Button tone='secondary' size="sm">
                                Keep
                              </Button>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6">🔄</div>
                    <h3 className="card-heading mb-4">No Duplicates Found</h3>
                    <p className="muted-text">Your files are well organized with no duplicates detected</p>
                  </div>
                )}
              </div>
            </Section>
          )}

          {tab==='cats' && (
            <Section>
              <div className="fade-in">
                <div className="flex items-center justify-between mb-10">
                  <h2 className="section-heading">Categories</h2>
                  <Button tone='secondary' onClick={refreshCats} disabled={isLoading('refreshCats')} loading={isLoading('refreshCats')}>
                    Refresh Categories
                  </Button>
                </div>
                
                {cats.length > 0 ? (
                  <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
                    {cats.map((cat, index) => (
                      <div key={cat.name} className="professional-card professional-interactive fade-in" style={{animationDelay: `${index * 0.1}s`}}>
                        <div className="flex items-center justify-between mb-4">
                          <div className="card-heading">{cat.name}</div>
                          <div className="text-2xl px-4 py-2 rounded-full bg-gradient-to-r from-accent-primary/20 to-accent-secondary/20 text-accent-primary">
                            {cat.count || 0}
                          </div>
                        </div>
                        <div className="muted-text">
                          {cat.description || `Files categorized as ${cat.name}`}
                        </div>
                        <div className="mt-4">
                          <h4 className="text-lg font-medium mb-2">Files in this category:</h4>
                          <ul>
                            {driveProps.filter(file => file.proposed_category === cat.name).map(file => (
                              <li key={file.id} className="text-sm text-text-muted">{file.name}</li>
                            ))}
                          </ul>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6">📂</div>
                    <h3 className="card-heading mb-4">No Categories Yet</h3>
                    <p className="muted-text mb-8">Scan your files to automatically generate smart categories</p>
                    <Button tone='primary' onClick={handleScan}>
                      Create Categories
                    </Button>
                  </div>
                )}
              </div>
            </Section>
          )}

          {tab==='ai' && (
            <Section>
              <div className="fade-in">
                <h2 className="section-heading mb-10">AI Assistant</h2>
                
                {notification && (
                  <div className="professional-card mb-6">
                    <div className="card-heading mb-4">Notification</div>
                    <div className="muted-text">{notification}</div>
                  </div>
                )}
                
                <div className="grid gap-8 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <div className="professional-card mb-6">
                      <div className="card-heading mb-4">Chat History</div>
                      <div className="space-y-4 max-h-96 overflow-y-auto">
                        {messages.length === 0 ? (
                          <div className="text-center py-8 text-text-muted">
                            <div className="text-4xl mb-4">🤖</div>
                            <p>Start a conversation with the AI assistant</p>
                          </div>
                        ) : (
                          <div className="space-y-4">
                            {messages.map((msg, index) => (
                              <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} items-end gap-2`}>
                                {/* Avatar for assistant */}
                                {msg.role === 'assistant' && (
                                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-accent-primary to-accent-secondary flex items-center justify-center text-white text-sm font-semibold">
                                    AI
                                  </div>
                                )}
                                
                                {/* Message bubble */}
                                <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                                  msg.role === 'user' 
                                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-none shadow-lg' 
                                    : 'bg-gray-100 text-gray-800 rounded-bl-none shadow-md'
                                }`}>
                                  {msg.content === '::typing::' ? (
                                    <div className="flex items-center gap-2">
                                      <div className="flex gap-1">
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                      </div>
                                      <span className="text-sm text-gray-500">Typing...</span>
                                    </div>
                                  ) : (
                                    <div className="whitespace-pre-wrap break-words">{msg.content}</div>
                                  )}
                                </div>
                                
                                {/* Avatar for user */}
                                {msg.role === 'user' && (
                                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-green-400 to-green-500 flex items-center justify-center text-white text-sm font-semibold">
                                    U
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="professional-card">
                      <div className="flex gap-4">
                        <input
                          ref={inputRef}
                          type="text"
                          value={askInput}
                          onChange={(e) => setAskInput(e.target.value)}
                          onKeyPress={(e) => e.key === 'Enter' && sendMsg()}
                          placeholder="Ask about your documents..."
                          className="flex-1 px-4 py-3 rounded-xl border border-border focus:outline-none focus:ring-2 focus:ring-accent-primary bg-bg-card text-text-primary"
                        />
                        <Button tone='primary' onClick={sendMsg} disabled={!askInput.trim()}>
                          Send
                        </Button>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="professional-card">
                      <div className="card-heading mb-4">Quick Actions</div>
                      <div className="space-y-3">
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('summarize')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <span className="mr-2">📊</span>
                          Summarize Documents
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('find_similar')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <span className="mr-2">🔍</span>
                          Find Similar Files
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('extract_insights')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <span className="mr-2">📝</span>
                          Extract Insights
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('organize')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <span className="mr-2">🗂️</span>
                          Organize Files
                        </Button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </Section>
          )}
        </div>
      </div>
      
      {showDocumentSelector ? (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h3 className="text-xl font-semibold text-gray-800">
                  Select a Document
                </h3>
                <button 
                  onClick={() => {
                    setShowDocumentSelector(false)
                  }}
                  className="w-8 h-8 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center text-gray-600 transition-colors"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p className="text-gray-600 mt-2">
                Choose a document to {currentQuickAction?.replace('_', ' ')}
              </p>
            </div>
            
            <div className="p-4 overflow-y-auto max-h-[60vh]">
              <div className="space-y-2">
                {driveProps.length === 0 ? (
                  <div className="text-center text-gray-500 py-8">
                    No documents available
                  </div>
                ) : (
                  driveProps.map((file) => (
                    <button
                      key={file.id}
                      onClick={() => {
                        setShowDocumentSelector(false)
                        setNotification(`Selected: ${file.name}`)
                        performQuickAction(currentQuickAction!, file)
                      }}
                      className="w-full p-4 bg-gray-50 hover:bg-gray-100 rounded-xl border border-gray-200 transition-all duration-200 text-left"
                    >
                      <div className="flex items-center gap-3">
                        <div className="flex-shrink-0 w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 font-semibold">
                          {file.name.charAt(0).toUpperCase()}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium text-gray-900 truncate">
                            {file.name}
                          </div>
                          <div className="text-sm text-gray-500">
                            {file.category || 'Document'}
                          </div>
                          <div className="text-xs text-gray-400">
                            ID: {file.id}
                          </div>
                        </div>
                        <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="fixed top-20 right-4 bg-blue-500 text-white p-2 rounded z-40">
          Modal is hidden (showDocumentSelector = false)
        </div>
      )}
    </div>
  )
}
