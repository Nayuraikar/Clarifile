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
  // First check if we have a direct proposed_label (from backend)
  if (p?.proposed_label && typeof p.proposed_label === 'string' && p.proposed_label.trim().length > 0) {
    return p.proposed_label.trim();
  }
  
  // Fallback to other possible fields
  const candidates = [
    p?.proposed_category,
    p?.proposed,
    p?.category,
    p?.label,
    p?.folder,
    p?.target_folder,
  ];
  
  // Return the first non-empty candidate
  for (const c of candidates) {
    if (typeof c === 'string' && c.trim().length > 0) return c.trim();
  }
  
  // Default fallback
  return 'Uncategorized';
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
  const [chats, setChats] = useState<Record<string, { file: DriveProposal, messages: { role: 'user'|'assistant', content: string }[] }>>({})
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)
  const [loadingStates, setLoadingStates] = useState<{[key: string]: boolean}>({})
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [showDocumentSelector, setShowDocumentSelector] = useState(false)
  const [currentQuickAction, setCurrentQuickAction] = useState<string | null>(null)
  const [quickActionLoading, setQuickActionLoading] = useState(false)
  const [duplicateResolution, setDuplicateResolution] = useState<{ [key: string]: boolean }>({})
  const [duplicateResolutionLoading, setDuplicateResolutionLoading] = useState(false)
  const [viewAllProposals, setViewAllProposals] = useState(false)
  const inputRef = React.useRef<HTMLInputElement>(null)

  // Helper function to set loading state for specific operations
  const setLoading = (operation: string, loading: boolean) => {
    setLoadingStates(prev => ({ ...prev, [operation]: loading }))
  }

  const isLoading = (operation: string) => loadingStates[operation] || false

  // Ensure a chat exists for a given file and select it
  const ensureChatForFile = (file: DriveProposal) => {
    setChats(prev => {
      if (prev[file.id]) {
        // If chat exists, make sure file info is up to date
        if (prev[file.id].file.name !== file.name || prev[file.id].file.proposed_category !== file.proposed_category) {
          return {
            ...prev,
            [file.id]: {
              ...prev[file.id],
              file: { ...file }
            }
          };
        }
        return prev;
      }
      // Create new chat if it doesn't exist
      return { 
        ...prev, 
        [file.id]: { 
          file: { ...file },
          messages: [] 
        } 
      };
    });
    setSelectedChatId(file.id);
    setDriveAnalyzedId(file.id);
    return file.id;
  }

  // Append a message to a specific chat
  const appendToChat = (chatId: string, message: { role: 'user'|'assistant', content: string }) => {
    setChats(prev => {
      const chat = prev[chatId];
      if (!chat) return prev;
      
      const updatedChat = {
        ...chat,
        messages: [...chat.messages, message]
      };
      
      // If this is a new message from the assistant, make sure the chat is selected
      if (message.role === 'assistant') {
        setSelectedChatId(chatId);
        setDriveAnalyzedId(chatId);
        
        // If we're not already on the AI Assistant tab, switch to it
        if (tab !== 'ai') {
          setTab('ai');
        }
      }
      
      return { ...prev, [chatId]: updatedChat };
    });
  }

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

      // If no proposals yet, return empty categories
      if (proposals.length === 0) {
        setCats([])
        return
      }

      // Extract unique categories from proposals
      const categoryMap = new Map<string, number>()
      proposals.forEach(file => {
        const category = getProposalCategory(file) || 'Uncategorized'
        categoryMap.set(category, (categoryMap.get(category) || 0) + 1)
      })

      // Convert to array of categories with counts
      const baseCats = Array.from(categoryMap.entries()).map(([name, count]) => ({
        name,
        count,
        folder_id: null,
        missing_folder: false
      }))

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
    
    if (!selectedChatId || !chats[selectedChatId]) {
      setNotification('Select a chat from the left to ask about a specific file.')
      return
    }

    // Add user message into the selected chat
    appendToChat(selectedChatId, { role: 'user', content: q })
    setAskInput('')
    
    // Add typing indicator into chat
    appendToChat(selectedChatId, { role: 'assistant', content: '::typing::' })
    const replaceTypingWith = (text: string) => {
      setChats(prev => {
        const chat = prev[selectedChatId!]
        if (!chat) return prev
        const msgs = [...chat.messages]
        if (msgs.length && msgs[msgs.length - 1].content === '::typing::' && msgs[msgs.length - 1].role === 'assistant') {
          msgs[msgs.length - 1] = { role: 'assistant', content: text }
        } else {
          msgs.push({ role: 'assistant', content: text })
        }
        return { ...prev, [selectedChatId!]: { ...chat, messages: msgs } }
      })
    }
    
    try {
      const chat = chats[selectedChatId]
      const file = chat.file
      if (!file?.id || String(file.id).startsWith('demo-file-')) {
        replaceTypingWith('This is a demo chat. Please pick a real document from Files or run Analyze/Summarize on a Drive file to chat about it.')
        return
      }
      const fileId = String(file.id)
      const response = await call('/drive/analyze', { 
        method:'POST', 
        headers:{'Content-Type':'application/json'}, 
        body: JSON.stringify({ 
          file:{ id:fileId, name:file.name||'', mimeType:'', parents:[] }, 
          q 
        }) 
      })
      if (response?.error) {
        replaceTypingWith(`Error: ${response.error}. If this persists, ensure Google Drive is connected and try reloading files.`)
        return
      }
      const ans = response?.qa?.answer || response?.summary || 'No answer available'
      replaceTypingWith(ans)
      setAskResult(response?.qa || { answer: ans })
      
    } catch (e:any) {
      console.error('Chatbot error:', e)
      replaceTypingWith(`Error: ${e?.message || 'Failed to get response from AI assistant.'} If this keeps happening, reselect the file in Files and click Analyze again.`)
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
          // For demo files, show demo in a chat; otherwise mirror Analyze into file chat
          {
            const isDemo = file.id.startsWith('demo-file-')
            ensureChatForFile(file)
            if (isDemo) {
              const demoContent = `**Summary Feature Demo**\n\n• **Feature:** Document Summarization\n• **Purpose:** Analyzes document content and provides key insights\n• **Usage:** Select a real document to get actual summary\n• **Status:** Ready to use with your files\n\n**To use with real files:** Upload documents via the Chrome extension or Files tab.`
              appendToChat(file.id, { role: 'assistant', content: demoContent })
              setQuickActionLoading(false)
              setNotification('Summary demo shown')
              return
            }
            call('/drive/analyze', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
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
              if (response && response.summary !== undefined) {
                appendToChat(file.id, {
                  role: 'assistant',
                  content: `I've analyzed the file \"${file.name}\". Here's what I found:\n\n**Summary:** ${response.summary}\n**Category:** ${response.category || 'General'}\n**Tags:** ${response.tags?.join(', ') || 'None'}\n\nYou can now ask me questions about this file!`
                })
                setNotification('Summary generated successfully!')
              } else if (response?.error) {
                appendToChat(file.id, { role: 'assistant', content: `Error: ${response.error}` })
              } else {
                appendToChat(file.id, { role: 'assistant', content: 'No summary available.' })
              }
              setQuickActionLoading(false)
            }).catch((e:any) => {
              appendToChat(file.id, { role: 'assistant', content: `Error analyzing file: ${e?.message || 'Unknown error'}` })
              setQuickActionLoading(false)
            })
            return
          }
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
      
      // Ensure we have a chat for this file
      ensureChatForFile(file);
      
      // Update the messages in the chat
      setChats(prev => {
        const updatedChats = {
          ...prev,
          [file.id]: {
            ...prev[file.id],
            file: file, // Make sure file info is up to date
            messages: [...(prev[file.id]?.messages || []), resultMessage]
          }
        };
        
        // Make sure this chat is selected
        setSelectedChatId(file.id);
        
        // If we're not already on the AI Assistant tab, switch to it
        if (tab !== 'ai') {
          setTab('ai');
        }
        
        return updatedChats;
      });
      
      // Set the drive analyzed ID to trigger any necessary effects
      setDriveAnalyzedId(file.id);
      
      setNotification(`${action.replace('_', ' ')} completed successfully!`);
      setQuickActionLoading(false);
    }, 1000) // Small delay to show loading state
  }

  return (
    <div className="min-h-screen relative overflow-hidden">
      {notification && (
        <div style={{ position: 'fixed', top: 16, left: 0, right: 0, display: 'flex', justifyContent: 'center', zIndex: 9999, pointerEvents: 'none' }}>
          <div className="notification-toast notification-info" style={{ pointerEvents: 'auto' }}>
            <div className="notification-content">
              <div className="notification-icon">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="white">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                </svg>
              </div>
              <span style={{color: '#4a4a4a', fontSize: '13px'}}>{notification}</span>
            </div>
            <button 
              onClick={() => setNotification(null)}
              className="notification-close"
              title="Close notification"
            >
              ×
            </button>
          </div>
        </div>
      )}
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
      
      {/* Beautiful Floating Orbs */}
      <div className="beautiful-orb"></div>
      <div className="beautiful-orb"></div>
      <div className="beautiful-orb"></div>
      
      {/* Professional Mouse Trail */}
      <div className="professional-mouse-trail" id="professional-mouse-trail"></div>
      
      {/* Main content with higher z-index to appear above background */}
      <div className="relative z-20">
        <header className="sticky top-0 z-30 professional-glass border-b border-white/20">
          <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-2xl bg-gradient-to-r from-accent-primary to-accent-secondary flex items-center justify-center shadow-lg">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z" fill="#ffffff"/>
                  <path d="M14 2V8H20" fill="#ffffff"/>
                  <path d="M16 13H8V15H16V13Z" fill="#8b7355"/>
                  <path d="M16 17H8V19H16V17Z" fill="#8b7355"/>
                  <path d="M10 9H8V11H10V9Z" fill="#8b7355"/>
                </svg>
              </div>
              <div>
                <div className="text-3xl font-extrabold" style={{color: '#8b7355'}}>Clarifile</div>
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
                  className={`professional-nav-item professional-button ${tab===id?'active':''}`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
        </header>

        {/* Beautiful Theme-Matching Notification - moved to global fixed container at bottom */}

        <div className="page-transition">
          {tab==='dashboard' && (
            <Section>
              {/* Enhanced Hero Section */}
              <div className="text-center py-16 fade-in relative">
                <div className="relative z-10 max-w-6xl mx-auto px-6">
                <h1 className="hero-heading mb-6">Welcome to Clarifile</h1>
                  <p className="body-text text-lg mb-12 max-w-3xl mx-auto">
                    Transform your digital chaos into organized perfection with AI-powered document management and intelligent file organization
                  </p>
                  
                  {/* Clean Action Cards */}
                  <div className="grid gap-6 md:grid-cols-3 mb-12">
                  <div className="professional-card dark-card fade-in" style={{animationDelay: '0.1s'}}>
                      <div className="cool-icon mb-6">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="currentColor"/>
                          <path d="M19 15L20.09 18.26L24 19L20.09 19.74L19 23L17.91 19.74L14 19L17.91 18.26L19 15Z" fill="currentColor"/>
                          <path d="M5 15L6.09 18.26L10 19L6.09 19.74L5 23L3.91 19.74L0 19L3.91 18.26L5 15Z" fill="currentColor"/>
                        </svg>
                      </div>
                      <h3 className="card-heading mb-3">Smart Organization</h3>
                      <p className="card-description">AI-powered categorization automatically organizes your files into intelligent folders</p>
                    </div>
                    <div className="professional-card dark-card fade-in" style={{animationDelay: '0.2s'}}>
                      <div className="cool-icon mb-6">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3S3 5.91 3 9.5S5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5S14 7.01 14 9.5S11.99 14 9.5 14Z" fill="currentColor"/>
                          <circle cx="9.5" cy="9.5" r="2.5" fill="currentColor"/>
                        </svg>
                      </div>
                      <h3 className="card-heading mb-3">Intelligent Search</h3>
                      <p className="card-description">Find any document instantly with advanced semantic search and content analysis</p>
                    </div>
                    <div className="professional-card dark-card fade-in" style={{animationDelay: '0.3s'}}>
                      <div className="cool-icon mb-6">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M7 2V13L10.5 9.5L14 13V2H7Z" fill="currentColor"/>
                          <path d="M19 7V9H17V7H19ZM19 11V13H17V11H19ZM19 15V17H17V15H19Z" fill="currentColor"/>
                          <path d="M3 7V9H15V7H3ZM3 11V13H15V11H3ZM3 15V17H15V15H3Z" fill="currentColor"/>
                          <path d="M21 19V21H3V19H21Z" fill="currentColor"/>
                        </svg>
                      </div>
                      <h3 className="card-heading mb-3">Lightning Fast</h3>
                      <p className="card-description">Process thousands of files in seconds with our optimized AI engine</p>
                    </div>
                  </div>
                  
                  {/* Clean Main Action Button */}
                  <Button tone='primary' onClick={handleScan} disabled={isScanning} loading={isScanning} className="professional-button">
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
                <div className="professional-card dark-card fade-in" style={{animationDelay: '0.4s'}}>
                  <div className="cool-icon mb-4">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z" fill="currentColor"/>
                      <path d="M14 2V8H20" fill="currentColor"/>
                    </svg>
                  </div>
                  <div className="text-3xl font-bold mb-2">{driveProps.length}</div>
                  <div className="card-description">Document Proposals</div>
                </div>
                <div className="professional-card dark-card fade-in" style={{animationDelay: '0.5s'}}>
                  <div className="cool-icon mb-4">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M10 4H4C2.89 4 2 4.89 2 6V18C2 19.11 2.89 20 4 20H20C21.11 20 22 19.11 22 18V8C22 6.89 21.11 6 20 6H12L10 4Z" fill="currentColor"/>
                    </svg>
                  </div>
                  <div className="text-3xl font-bold mb-2">{driveProps.length}</div>
                  <div className="card-description">Drive Files</div>
                </div>
                <div className="professional-card dark-card fade-in" style={{animationDelay: '0.6s'}}>
                  <div className="cool-icon mb-4">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3ZM19 19H5V5H19V19Z" fill="currentColor"/>
                      <path d="M19 3H5C3.9 3 3 3.9 3 5V19C3 20.1 3.9 21 5 21H19C20.1 21 21 20.1 21 19V5C21 3.9 20.1 3 19 3Z" fill="currentColor"/>
                    </svg>
                  </div>
                  <div className="text-3xl font-bold mb-2">{dups.length}</div>
                  <div className="card-description">Duplicate Groups</div>
                </div>
                <div className="professional-card dark-card fade-in" style={{animationDelay: '0.7s'}}>
                  <div className="cool-icon mb-4">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                      <path d="M12 2L15.09 8.26L22 9L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9L8.91 8.26L12 2Z" fill="currentColor"/>
                    </svg>
                  </div>
                  <div className="text-3xl font-bold mb-2">{cats.length}</div>
                  <div className="card-description">Categories</div>
                </div>
              </div>
              
              {/* Enhanced Document Proposals Section */}
              <div className="fade-in" style={{animationDelay: '0.8s'}}>
                <div className="proposals-section-header">
                  <h2 className="proposals-section-title">
                    {viewAllProposals ? 'All Document Proposals' : 'Recent Document Proposals'}
                  </h2>
                  {driveProps.length > 0 && (
                    <Button 
                      tone='secondary' 
                      className="professional-button"
                      onClick={() => setViewAllProposals(!viewAllProposals)}
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z" fill="currentColor"/>
                      </svg>
                      {viewAllProposals ? 'View Less' : `View All (${driveProps.length})`}
                    </Button>
                  )}
                </div>
                
                {driveProps.length > 0 ? (
                  <div className="proposals-grid">
                    {driveProps.slice(0, viewAllProposals ? driveProps.length : 4).map((p, index) => (
                      <div key={p.id} className="document-proposal-card fade-in" style={{animationDelay: `${0.9 + index * 0.1}s`}}>
                        <div className="proposal-header">
                          <div className="proposal-info">
                            <div className="proposal-title">{p.name}</div>
                            <div className="proposal-category">
                              Proposed: {p.proposed_category}
                            </div>
                          </div>
                          <div className={`proposal-status ${(p as any).final ? 'approved' : 'proposed'}`}>
                            {(p as any).final ? (
                              <span className="flex items-center gap-2">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                </svg>
                                Approved
                              </span>
                            ) : (
                              <span className="flex items-center gap-2">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                                </svg>
                                Proposed
                              </span>
                            )}
                          </div>
                        </div>
                        <div className="proposal-actions">
                          <button className="proposal-button approve" onClick={async()=>{ 
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
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                            </svg>
                            Approve
                          </button>
                          <button className="proposal-button secondary" onClick={async()=>{
                            const r = await fetch(`${BASE}/file_summary?file_id=${p.id}`); 
                            if(r.status===200){ 
                              const j = await r.json(); 
                              alert(`Summary: ${j.summary}`); 
                            }
                          }}>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                            </svg>
                            View Summary
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="proposals-empty-state">
                    <div className="proposals-empty-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z"/>
                        <path d="M14 2V8H20"/>
                        <circle cx="12" cy="15" r="2"/>
                      </svg>
                    </div>
                    <h3 className="proposals-empty-title">No Document Proposals Yet</h3>
                    <p className="proposals-empty-description">Start by scanning your files to get intelligent organization suggestions</p>
                    <Button tone='primary' onClick={handleScan} className="professional-button">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                      </svg>
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
                <div className="drive-section-header">
                  <h2 className="drive-section-title">Drive Files</h2>
                  <div className="button-group">
                    <Button tone='secondary' onClick={refreshDrive} disabled={isLoading('refreshDrive')} loading={isLoading('refreshDrive')} className="professional-button">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                      </svg>
                      Refresh Files
                    </Button>
                    <Button tone='primary' onClick={handleScan} disabled={isScanning} loading={isScanning} className="professional-button">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                      </svg>
                      Scan Drive
                    </Button>
                  </div>
                </div>
                
                {driveProps.length > 0 ? (
                  <div className="grid gap-4">
                    {driveProps.map((file, index) => (
                      <div key={file.id} className={`drive-file-card fade-in ${driveAnalyzedId === file.id ? 'selected' : ''}`} style={{animationDelay: `${index * 0.1}s`}}>
                        <div className="file-header">
                          <div className="file-info">
                            <div className="flex items-center gap-3 mb-2">
                              <div className="file-name">{file.name}</div>
                              {driveAnalyzedId === file.id && (
                                <div className="ai-selected-badge">
                                  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                  </svg>
                                  Selected for AI
                                </div>
                              )}
                            </div>
                            <div className="file-category">
                              Proposed Category: {file.proposed_category}
                            </div>
                            <div className="custom-category-container">
                              <label className="custom-category-label">Custom Category:</label>
                              <input
                                type="text"
                                value={customLabels[file.id] || ''}
                                onChange={(e)=> setCustomLabels(prev=>({...prev, [file.id]: e.target.value}))}
                                placeholder="Enter custom folder name"
                                className="custom-category-input"
                              />
                            </div>
                          </div>
                          <div className="file-actions">
                            <button className={`file-action-button analyze ${driveAnalyzedId === file.id ? 'selected' : ''}`} onClick={() => {
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
                                
                                // Ensure chat exists and append analysis to that chat
                                ensureChatForFile(file);
                                
                                // Switch to AI Assistant tab
                                setTab('ai');
                                
                                if (response.summary) {
                                  appendToChat(file.id, {
                                    role: 'assistant',
                                    content: `I've analyzed the file "${file.name}". Here's what I found:\n\n**Summary:** ${response.summary}\n**Category:** ${response.category || 'General'}\n**Tags:** ${response.tags?.join(', ') || 'None'}\n\nYou can now ask me questions about this file!`
                                  });
                                  console.log('Added analysis to AI Assistant chat');
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
                            }} disabled={isLoading('analyzeFile')}>
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                              </svg>
                              {driveAnalyzedId === file.id ? 'Selected' : 'Analyze'}
                            </button>
                            <button className="file-action-button approve" onClick={async () => {
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
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                              </svg>
                              Approve
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="drive-empty-state">
                    <div className="proposals-empty-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M10 4H4C2.89 4 2 4.89 2 6V18C2 19.11 2.89 20 4 20H20C21.11 20 22 19.11 22 18V8C22 6.89 21.11 6 20 6H12L10 4Z"/>
                      </svg>
                    </div>
                    <h3 className="proposals-empty-title">No Drive Files Found</h3>
                    <p className="proposals-empty-description">Connect your Google Drive and scan your files to get started</p>
                    <Button tone='primary' onClick={handleScan} className="professional-button">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                      </svg>
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
                <div className="duplicates-section-header">
                  <h2 className="duplicates-section-title">Duplicate Files</h2>
                  <Button tone='secondary' onClick={refreshDups} disabled={isLoading('refreshDups')} loading={isLoading('refreshDups')} className="professional-button">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                    </svg>
                    Refresh Duplicates
                  </Button>
                </div>
                
                {dups.length > 0 ? (
                  <div className="grid gap-4">
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
                  <div className="duplicates-empty-state">
                    <div className="duplicates-empty-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                      </svg>
                    </div>
                    <h3 className="duplicates-empty-title">No Duplicates Found</h3>
                    <p className="duplicates-empty-description">Your files are well organized with no duplicates detected</p>
                  </div>
                )}
              </div>
            </Section>
          )}

          {tab==='cats' && (
            <Section>
              <div className="fade-in">
                <div className="categories-section-header">
                  <h2 className="categories-section-title">Categories</h2>
                  <Button tone='secondary' onClick={refreshCats} disabled={isLoading('refreshCats')} loading={isLoading('refreshCats')} className="professional-button">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
                    </svg>
                    Refresh Categories
                  </Button>
                </div>
                
                {cats.length > 0 ? (
                  <div className="categories-grid">
                    {cats.map((cat, index) => (
                      <div key={cat.name} className="category-card fade-in" style={{animationDelay: `${index * 0.1}s`}}>
                        <div className="category-header">
                          <div className="category-name">{cat.name}</div>
                          <div className="category-count">
                            {(cat.proposed?.length || 0)}
                          </div>
                        </div>
                        <div className={`category-status ${cat.folder_id ? 'available' : 'missing'}`}>
                          {cat.folder_id ? (
                            <span>
                              Drive folder available · <a className="category-drive-link" href={`https://drive.google.com/drive/folders/${cat.folder_id}`} target="_blank" rel="noreferrer">Open in Drive</a>
                            </span>
                          ) : (cat.missing_folder ? 'Folder missing in Drive' : `Files categorized as ${cat.name}`)}
                        </div>
                        <div className="category-files">
                          <div className="file-list-section">
                            <h4 className="file-list-title">Existing in folder</h4>
                            <ul className="file-list">
                              {(cat.existing || []).map((file:any) => (
                                <li key={file.id} className="file-list-item">{file.name}</li>
                              ))}
                              {(cat.existing || []).length === 0 && (
                                <li className="file-list-empty">No files present</li>
                              )}
                            </ul>
                          </div>
                          <div className="file-list-section">
                            <h4 className="file-list-title">Proposed</h4>
                            <ul className="file-list">
                              {(cat.proposed || []).map((file:any) => (
                                <li key={file.id} className="file-list-item">{file.name}</li>
                              ))}
                              {(cat.proposed || []).length === 0 && (
                                <li className="file-list-empty">No proposals</li>
                              )}
                            </ul>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="drive-empty-state">
                    <div className="proposals-empty-icon">
                      <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M10 4H4C2.89 4 2 4.89 2 6V18C2 19.11 2.89 20 4 20H20C21.11 20 22 19.11 22 18V8C22 6.89 21.11 6 20 6H12L10 4Z"/>
                      </svg>
                    </div>
                    <h3 className="proposals-empty-title">No Categories Yet</h3>
                    <p className="proposals-empty-description">Scan your files to automatically generate smart categories</p>
                    <Button tone='primary' onClick={handleScan} className="professional-button">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2L13.09 8.26L20 9L13.09 9.74L12 16L10.91 9.74L4 9L10.91 8.26L12 2Z"/>
                      </svg>
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
                  <div className="ai-notification-compact">
                    <div className="notification-header">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                      </svg>
                      Status Update
                    </div>
                    <div className="notification-text">{notification}</div>
                  </div>
                )}
                
                <div className="grid gap-8 lg:grid-cols-3">
                  <div className="lg:col-span-2">
                    <div className="flex gap-4">
                      {/* Sidebar listing chats by file name */}
                      <div className="professional-card" style={{ width: 280, flex: '0 0 auto' }}>
                        <div className="card-heading mb-3">Chats</div>
                        <div className="space-y-2">
                          {Object.keys(chats).length === 0 ? (
                            <div className="chat-empty-state">
                              <div className="chat-empty-icon"></div>
                              <p className="chat-empty-text">Analyze a file to start a chat</p>
                            </div>
                          ) : (
                            Object.values(chats).map(({ file }) => (
                              <button
                                key={file.id}
                                onClick={() => { setSelectedChatId(file.id); setDriveAnalyzedId(file.id) }}
                                className={`w-full text-left professional-button ${selectedChatId === file.id ? 'active' : ''}`}
                              >
                                {file.name}
                              </button>
                            ))
                          )}
                        </div>
                      </div>
                      {/* Chat pane */}
                      <div className="flex-1">
                        <div className="chat-container">
                          <div className="chat-header"> {selectedChatId && chats[selectedChatId] ? chats[selectedChatId].file.name : 'Chat'}</div>
                          <div className="chat-messages">
                            {!selectedChatId || !chats[selectedChatId] || chats[selectedChatId].messages.length === 0 ? (
                              <div className="chat-empty-state">
                                <div className="chat-empty-icon"></div>
                                <p className="chat-empty-text">Select a chat or analyze a file to begin</p>
                              </div>
                            ) : (
                              <div className="chat-messages-container">
                                {chats[selectedChatId].messages.filter(m => m.content !== '::typing::').map((msg, index) => (
                                  <div key={index} className="message-wrapper">
                                    {msg.role === 'user' ? (
                                      <div className="user-message-container">
                                        <div className="user-message-bubble">
                                          <div className="message-text">{msg.content}</div>
                                        </div>
                                        <div className="chat-avatar user-avatar">U</div>
                                      </div>
                                    ) : (
                                      <div className="assistant-message-container">
                                        <div className="chat-avatar assistant-avatar">AI</div>
                                        <div className="assistant-message-bubble">
                                          {msg.content === '::typing::' ? (
                                            <div className="typing-indicator">
                                              <div className="typing-dots">
                                                <div className="typing-dot"></div>
                                                <div className="typing-dot"></div>
                                                <div className="typing-dot"></div>
                                              </div>
                                              <span className="typing-text">Typing...</span>
                                            </div>
                                          ) : (
                                            <div className="message-text">{msg.content}</div>
                                          )}
                                        </div>
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
                              placeholder="Ask about this file..."
                              className="flex-1 px-4 py-3 rounded-xl border border-border focus:outline-none focus:ring-2 focus:ring-accent-primary bg-bg-card text-text-primary"
                            />
                            <Button tone='primary' onClick={sendMsg} disabled={!askInput.trim() || !selectedChatId} className="professional-button">
                              Send
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="professional-card">
                      <div className="card-heading mb-4">Quick Actions</div>
                      <div className="space-y-3">
                        <Button tone='secondary' className="w-full justify-start professional-button" onClick={() => handleQuickAction('summarize')} disabled={quickActionLoading} loading={quickActionLoading}>
                          Summarize Documents
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-button" onClick={() => handleQuickAction('find_similar')} disabled={quickActionLoading} loading={quickActionLoading}>
                          Find Similar Files
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-button" onClick={() => handleQuickAction('extract_insights')} disabled={quickActionLoading} loading={quickActionLoading}>
                          Info Panel
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-button" onClick={() => handleQuickAction('organize')} disabled={quickActionLoading} loading={quickActionLoading}>
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
        <div className="quick-action-backdrop">
          <div className="quick-action-modal">
            <div className="quick-action-header">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="quick-action-title">
                    Select a Document
                  </h3>
                  <p className="quick-action-subtitle">
                    Choose a document to {currentQuickAction?.replace('_', ' ')}
                  </p>
                </div>
                <button 
                  onClick={() => {
                    setShowDocumentSelector(false)
                    setQuickActionLoading(false)
                  }}
                  className="quick-action-close"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            
            <div className="quick-action-content">
              {driveProps.length === 0 ? (
                <div className="documents-empty">
                  <div className="documents-empty-icon"></div>
                  <p className="documents-empty-text">No documents available</p>
                </div>
              ) : (
                <div>
                  {driveProps.map((file, index) => (
                    <button
                      key={file.id}
                      onClick={() => {
                        setShowDocumentSelector(false)
                        setNotification(`Selected: ${file.name}`)
                        performQuickAction(currentQuickAction!, file)
                      }}
                      className="document-item"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <div className="flex items-center">
                        <div className="document-avatar">
                          {file.name.charAt(0).toUpperCase()}
                        </div>
                        <div className="document-info">
                          <div className="document-name">
                            {file.name}
                          </div>
                          <div className="document-category">
                            {file.proposed_category || 'Document'}
                          </div>
                          <div className="document-id">
                            ID: {file.id}
                          </div>
                        </div>
                        <svg className="w-5 h-5 document-arrow" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}

