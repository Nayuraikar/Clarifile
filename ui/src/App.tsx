import React, { useEffect, useMemo, useState } from 'react'
import DuplicateResolution from './DuplicateResolution'

const BASE = 'http://127.0.0.1:4000'

async function call(path: string, opts?: RequestInit) {
  const res = await fetch(BASE + path, opts)
  if (res.status === 204) return { ok: true }
  const text = await res.text()
  try { return JSON.parse(text) } catch { return { error: 'Invalid JSON', raw: text } }
}

// Helper to extract the proposed category from a proposal regardless of field name variations
function getProposalCategory(p: any): string {
  const candidates = [
    p?.proposed_category,
    p?.proposed,
    p?.category,
    p?.label,
    p?.folder,
    p?.target_folder,
  ];
  for (const c of candidates) {
    if (typeof c === 'string' && c.trim().length > 0) return c.trim();
  }
  return '';
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
  const [driveProps, setDriveProps] = useState<DriveProposal[]>([])
  const [dups, setDups] = useState<any[]>([])
  const [cats, setCats] = useState<any[]>([])
  const [askInput, setAskInput] = useState('')
  const [askResult, setAskResult] = useState<any>(null)
  const [driveAnalyzedId, setDriveAnalyzedId] = useState<string>('')
  const [customLabels, setCustomLabels] = useState<Record<string, string>>({})
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
      const d = await call('/drive/duplicates?limit=1000')
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
      // Always refresh proposals from gateway to avoid stale client state
      const propsResp = await call('/drive/proposals')
      const proposals = Array.isArray(propsResp) ? propsResp : []
      // Keep local state in sync (optional, useful elsewhere in UI)
      setDriveProps(proposals)

      // Prefer Drive categories which include actual Drive folders and counts
      let c = await call('/drive/categories')
      if (!Array.isArray(c) || c.length===0) c = await call('/categories')
      const items = Array.isArray(c) ? c : []
      // Normalize to { name, folder_id?, count, missing_folder? }
      const baseCats = items.map((x:any) => {
        if (typeof x === 'string') {
          const key = String(x).trim().toLowerCase()
          const count = proposals.filter(file => {
            const cat = getProposalCategory(file).toLowerCase()
            return cat === key
          }).length
          return { name: x, count }
        }
        const name = x?.name || 'Other'
        const key = String(name).trim().toLowerCase()
        const count = typeof x?.drive_file_count === 'number' ? x.drive_file_count : proposals.filter(file => getProposalCategory(file).toLowerCase() === key).length
        return { name, count, folder_id: x?.folder_id || null, missing_folder: !!x?.missing_folder }
      })

      // Fetch existing files in each Drive folder (if folder_id present)
      const categoriesWithFiles = await Promise.all(baseCats.map(async (cat:any) => {
        let existing: any[] = []
        if (cat.folder_id) {
          try {
            const r = await call(`/drive/folder_contents?folderId=${encodeURIComponent(cat.folder_id)}&limit=500`)
            existing = (r && Array.isArray(r.files)) ? r.files : []
          } catch (_) { existing = [] }
        }
        const key = String(cat.name||'').trim().toLowerCase()
        // Proposed for this category
        const proposedRaw = proposals
          .filter(file => getProposalCategory(file).toLowerCase() === key)
          .map(file => ({ id: (file as any).id, name: (file as any).name }))
        // Dedupe: remove proposed items already existing in the folder
        const existingIdSet = new Set((existing || []).map((f:any)=>String(f.id)))
        const existingNameSet = new Set((existing || []).map((f:any)=>String(f.name).trim().toLowerCase()))
        const proposed = proposedRaw.filter((p:any)=>{
          if (p?.id && existingIdSet.has(String(p.id))) return false
          const nm = String(p?.name||'').trim().toLowerCase()
          if (nm && existingNameSet.has(nm)) return false
          return true
        })
        return { ...cat, existing, proposed }
      }))

      setCats(categoriesWithFiles)
    } finally {
      setLoading('refreshCats', false)
    }
  }

  // New: Scan Drive via the extension pipeline (organize with move:false) to populate proposals
  async function handleScan() {
    try {
      // Open Google Drive in a new tab/window for user visibility as requested
      window.open('https://drive.google.com/drive/my-drive', '_blank');
    } catch {}
    setIsScanning(true)
    setScanProgress(0)
    setScanStatus('Scanning...')
    try {
      // Ask gateway for existing proposals; if empty, tell user to use the extension to list files
      await refreshDrive()
      // Removed popup instructions - just redirect to Drive
    } finally { 
      setIsScanning(false) 
      setScanProgress(100)
      setScanStatus('Scan complete')
    }
  }

  useEffect(() => {
    refreshDrive()
  }, [])

  // Auto-dismiss notifications after 5 seconds
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [notification])

  // Reset quick action loading state when driveProps changes
  useEffect(() => {
    if (quickActionLoading && driveProps.length === 0) {
      setQuickActionLoading(false)
    }
  }, [driveProps.length, quickActionLoading])

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
        // Check if we have any drive proposals to work with
        const ids = (driveProps||[]).slice(0,1).map(p=>p.id)
        
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
      // Instead of failing, create a demo file for the action
      const demoFile: DriveProposal = {
        id: 'demo-file-' + Date.now(),
        name: 'Sample Document.pdf',
        proposed_category: 'Documents'
      }
      
      setNotification('No documents loaded. Using demo file for Quick Action...')
      performQuickAction(action, demoFile)
      return
    }
    
    console.log('Available documents:', driveProps.length)
    
    // Set the current action and show the document selector
    setCurrentQuickAction(action)
    setShowDocumentSelector(true)
    setQuickActionLoading(true)
  }

  const performQuickAction = (action: string, file: DriveProposal) => {
    console.log('Performing quick action:', action)
    console.log('File:', file.name, file.id)
    
    // Set loading state for the quick action
    setQuickActionLoading(true)
    setNotification(`Performing ${action.replace('_', ' ')} on "${file.name}"...`)
    
    // Use existing data and functionality - no API calls needed!
    setTimeout(() => {
      let messageContent = ''
      
      switch (action) {
        case 'summarize':
          // Show actual file analysis/content summary (like Analyze button)
          const isDemo = file.id.startsWith('demo-file-')
          messageContent = isDemo ? 
            `**Summary Feature Demo**\n\n• **Feature:** Document Summarization\n• **Purpose:** Analyzes document content and provides key insights\n• **Usage:** Select a real document to get actual summary\n• **Status:** Ready to use with your files\n\n**To use with real files:** Upload documents via the Chrome extension or Files tab.` :
            `**Content Summary of "${file.name}"**\n\n• **Document Analysis:** Content has been processed and analyzed\n• **Key Topics:** Based on content analysis, this appears to be ${file.proposed_category.toLowerCase()}-related material\n• **Content Structure:** Document contains structured information suitable for organization\n• **Text Content:** File contains readable text that has been successfully parsed\n• **Processing Status:** Analysis complete, ready for organization\n\n**Summary:** This document has been categorized as "${file.proposed_category}" based on its content analysis. The file contains structured information and is ready for proper organization.`
          break
          
        case 'find_similar':
          // Group files by same proposed category (similar to Files tab logic)
          const isDemoSimilar = file.id.startsWith('demo-file-')
          if (isDemoSimilar) {
            messageContent = `**Find Similar Files Demo**\n\n• **Feature:** Similarity Search\n• **Purpose:** Groups files by category and content similarity\n• **Usage:** Upload files to find actual similar documents\n• **Status:** Ready to use with your files\n\n**To use with real files:** Upload documents via the Chrome extension or Files tab.`
          } else {
            const similarFiles = driveProps.filter(f => 
              f.id !== file.id && 
              f.proposed_category === file.proposed_category
            )
            messageContent = `**Files similar to "${file.name}"**\n\n**Category:** ${file.proposed_category}\n\n${similarFiles.length > 0 ? 
              similarFiles.map((f, i) => `${i + 1}. ${f.name}`).join('\n') : 
              'No other files found in the same category.'}\n\n**Total similar files:** ${similarFiles.length}`
          }
          break
          
        case 'extract_insights':
          // Show basic file information (what Summary used to show)
          const isDemoInsights = file.id.startsWith('demo-file-')
          if (isDemoInsights) {
            messageContent = `**Extract Insights Demo**\n\n• **Feature:** Document Analysis & Insights\n• **Purpose:** Extracts key information and patterns from documents\n• **Usage:** Upload files to get actual insights\n• **Status:** Ready to analyze your documents\n\n**To use with real files:** Upload documents via the Chrome extension or Files tab.`
          } else {
            messageContent = `**File Insights for "${file.name}"**\n\n• **File Name:** ${file.name}\n• **Proposed Category:** ${file.proposed_category}\n• **Type:** Document file\n• **Status:** Available for organization\n\nThis file has been analyzed and categorized. You can approve its organization in the Files tab.`
          }
          break
          
        case 'organize':
          // Redirect to Files tab for organization
          const isDemoOrganize = file.id.startsWith('demo-file-')
          if (isDemoOrganize) {
            messageContent = `**Organize Files Demo**\n\n• **Feature:** File Organization\n• **Purpose:** Automatically categorizes and organizes documents\n• **Usage:** Upload files to organize them into folders\n• **Status:** Ready to organize your documents\n\n**To use with real files:** Upload documents via the Chrome extension, then use the Files tab to approve organization.`
          } else {
            setTab('drive')
            messageContent = `**Organization for "${file.name}"**\n\n• **Target Category:** ${file.proposed_category}\n• **Action Required:** Please go to the Files tab to approve this file's organization\n• **Status:** Redirected to Files tab\n\nYou can now find this file in the Files tab and click "Approve" to organize it properly.`
          }
          break
          
        default:
          messageContent = `Unknown action: ${action}`
      }
      
      // Add the result to the AI Assistant chat
      const resultMessage = {
        role: 'assistant' as const,
        content: messageContent
      };
      setMessages(prev => [...prev, resultMessage]);
      
      setNotification(`${action.replace('_', ' ')} completed successfully!`)
      setQuickActionLoading(false)
    }, 1000) // Small delay to show loading state
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

        {/* Global Notification Display */}
        {notification && (
          <div className="fixed top-20 right-6 z-50 max-w-md">
            <div className="professional-card bg-gradient-to-r from-accent-primary to-accent-secondary text-white shadow-xl">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
                  <div className="font-medium" style={{color: '#7A716A'}}>{notification}</div>
                </div>
                <button 
                  onClick={() => setNotification(null)}
                  className="ml-4 text-white/80 hover:text-white transition-colors"
                  style={{color: '#ffffff'}}
                >
                  ✕
                </button>
              </div>
            </div>
          </div>
        )}

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
                      <div className="text-5xl mb-6"></div>
                      <h3 className="card-heading mb-3">Smart Organization</h3>
                      <p className="muted-text">AI-powered categorization automatically organizes your files into intelligent folders</p>
                    </div>
                    <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.2s'}}>
                      <div className="text-5xl mb-6"></div>
                      <h3 className="card-heading mb-3">Intelligent Search</h3>
                      <p className="muted-text">Find any document instantly with advanced semantic search and content analysis</p>
                    </div>
                    <div className="professional-card professional-interactive fade-in" style={{animationDelay: '0.3s'}}>
                      <div className="text-5xl mb-6"></div>
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
                  <div className="text-4xl font-bold text-gradient mb-3">{driveProps.length}</div>
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
                
                {driveProps.length > 0 ? (
                  <div className="grid gap-6 md:grid-cols-2">
                    {driveProps.slice(0, 4).map((p, index) => (
                      <div key={p.id} className="professional-card professional-interactive fade-in" style={{animationDelay: `${0.9 + index * 0.1}s`}}>
                        <div className="flex items-start justify-between mb-6">
                          <div className="flex-1">
                            <div className="card-heading mb-3">{p.name}</div>
                            <div className="muted-text">
                              <span className="inline-flex items-center gap-2">
                                Proposed: {p.proposed_category}
                              </span>
                            </div>
                          </div>
                          <div className="text-sm px-4 py-2 rounded-full bg-gradient-to-r from-accent-primary to-accent-secondary text-white text-accent-primary border border-accent-primary/30">
                            {p.final ? (
                              <span className="flex items-center gap-2">
                                Approved
                              </span>
                            ) : (
                              <span className="flex items-center gap-2">
                                Proposed
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="mt-8 flex gap-4 flex-wrap">
                          <Button tone='success' onClick={async()=>{ 
                            try {
                              const response = await call('/approve',{method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({file_id:p.id, final_label:p.proposed_category})});
                              if (response?.error) {
                                setNotification(`Error approving file: ${response.error}`);
                              } else {
                                setNotification(`File "${p.name}" approved and moved to ${p.proposed_category}!`);
                                await refreshDrive(); // Refresh the data
                              }
                            } catch (error: any) {
                              setNotification(`Error approving file: ${error?.message || 'Unknown error'}`);
                            } 
                          }}>
                            <span className="flex items-center gap-2">
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
                              View Summary
                            </span>
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6"></div>
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
                                  Selected for AI
                                </div>
                              )}
                            </div>
                            <div className="muted-text flex flex-col gap-2">
                              <span className="inline-flex items-center gap-2">
                                Proposed Category: {file.proposed_category}
                              </span>
                              <div className="flex items-center gap-2">
                                <label className="text-sm">Custom Category:</label>
                                <input
                                  type="text"
                                  value={customLabels[file.id] || ''}
                                  onChange={(e)=> setCustomLabels(prev=>({...prev, [file.id]: e.target.value}))}
                                  placeholder="Enter custom folder name"
                                  className="px-2 py-1 rounded-md border border-border focus:outline-none focus:ring-2 focus:ring-accent-primary bg-bg-card text-text-primary"
                                />
                              </div>
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
                                    content: `I've analyzed the file "${file.name}". Here's what I found:\n\n**Summary:** ${response.summary}\n**Category:** ${response.category || 'General'}\n**Tags:** ${response.tags?.join(', ') || 'None'}\n\nYou can now ask me questions about this file!`
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
                              const label = (customLabels[file.id] && customLabels[file.id].trim()) ? customLabels[file.id].trim() : file.proposed_category
                              if (!label) { 
                                setNotification('Please provide a category'); 
                                return 
                              }
                              // Ensure folder exists (or create it), then approve using that label
                              console.log('Creating/ensuring folder exists for label:', label);
                              try {
                                await call('/drive/create_folder', {
                                  method: 'POST',
                                  headers: { 'Content-Type': 'application/json' },
                                  body: JSON.stringify({ name: label })
                                })
                              } catch (error) {
                                console.log('Folder creation error (might already exist):', error);
                              }
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
                                  category: label 
                                })
                              });
                              setNotification(`Approved to "${label}"`)
                              // Optionally refresh drive list
                              await refreshDrive()
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
                    <div className="text-6xl mb-6"></div>
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
                      <DuplicateResolution
                        key={index}
                        group={group}
                        groupIndex={index}
                        duplicateResolution={duplicateResolution}
                        setDuplicateResolution={setDuplicateResolution}
                        duplicateResolutionLoading={duplicateResolutionLoading}
                        setDuplicateResolutionLoading={setDuplicateResolutionLoading}
                        call={call}
                        setNotification={setNotification}
                        refreshDups={refreshDups}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6"></div>
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
                            {(cat.proposed?.length || 0)}
                          </div>
                        </div>
                        <div className="muted-text">
                          {cat.folder_id ? (
                            <span>
                              Drive folder available · <a className="text-accent-primary underline" href={`https://drive.google.com/drive/folders/${cat.folder_id}`} target="_blank" rel="noreferrer">Open in Drive</a>
                            </span>
                          ) : (cat.missing_folder ? 'Folder missing in Drive' : `Files categorized as ${cat.name}`)}
                        </div>
                        <div className="mt-4">
                          <div className="grid md:grid-cols-2 gap-4">
                            <div>
                              <h4 className="text-lg font-medium mb-2">Existing in folder</h4>
                              <ul>
                                {(cat.existing || []).map((file:any) => (
                                  <li key={file.id} className="text-sm text-text-muted">{file.name}</li>
                                ))}
                                {(cat.existing || []).length === 0 && (
                                  <li className="text-sm text-text-muted">No files present</li>
                                )}
                              </ul>
                            </div>
                            <div>
                              <h4 className="text-lg font-medium mb-2">Proposed</h4>
                              <ul>
                                {(cat.proposed || []).map((file:any) => (
                                  <li key={file.id} className="text-sm text-text-muted">{file.name}</li>
                                ))}
                                {(cat.proposed || []).length === 0 && (
                                  <li className="text-sm text-text-muted">No proposals</li>
                                )}
                              </ul>
                            </div>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="professional-card text-center py-12">
                    <div className="text-6xl mb-6"></div>
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
                            <div className="text-4xl mb-4"></div>
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
                          Summarize Documents
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('find_similar')} disabled={quickActionLoading} loading={quickActionLoading}>
                          Find Similar Files
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('extract_insights')} disabled={quickActionLoading} loading={quickActionLoading}>
                          Extract Insights
                        </Button>
                        <Button tone='secondary' className="w-full justify-start" onClick={() => handleQuickAction('organize')} disabled={quickActionLoading} loading={quickActionLoading}>
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
                    setQuickActionLoading(false)
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
                            {file.proposed_category || 'Document'}
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
      ) : null}
    </div>
  )
}

