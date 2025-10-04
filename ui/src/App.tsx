import React, { useEffect, useMemo, useState } from 'react'
import DuplicateResolution from './DuplicateResolution'
import SearchFiles from './components/SearchFiles'

const BASE = 'http://127.0.0.1:4000'

async function call(path: string, opts?: RequestInit) {
  const res = await fetch(BASE + path, opts ? opts : {})
  if (res.status === 204) return { ok: true }
  const text = await res.text()
  try { return JSON.parse(text) } catch { return { error: 'Invalid JSON', raw: text } }
}
function getProposalCategory(p: any): string {
  const getCategory = (): string => {
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
  };

  // Get the category
  let category = getCategory();
  
  // Check if the category is in the format "Category: Category" and simplify it
  if (category.includes(':')) {
    const [main, sub] = category.split(':').map(s => s.trim());
    if (main.toLowerCase() === sub.toLowerCase()) {
      return main; 
    }
  }
  return category;
}

interface DriveProposal {
  id: string;
  name: string;
  proposed_category: string;
  approved?: boolean;
  final_category?: string;
  mimeType?: string;
  parents?: string[];
}

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
  const [tab, setTab] = useState<'dashboard'|'drive'|'dups'|'cats'|'ai'|'search'>('dashboard')
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
 
  //  download functionality
  const DownloadButton = ({ message, className = '' }: { message: ChatMessage, className?: string }) => {
    if (message.role !== 'assistant' || !message.assistant || message.assistant.type !== 'download') {
      return null;
    }

    const { kind, filename, data, base64, mime } = message.assistant;
    const downloadData = data || base64;
    const messageId = message.id || `${Date.now()}`;
    const isLoading = downloads[messageId];

    const handleDownload = () => {
      if (!downloadData) {
        console.error('No download data available');
        setNotification('No download data available');
        return;
      }

      try {
        // Decode base64 data
        const byteCharacters = atob(downloadData);
        const byteNumbers = new Array(byteCharacters.length);
        
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const mimeType = mime || `application/${kind}` || 'application/octet-stream';
        const blob = new Blob([byteArray], { type: mimeType });
        
        // Create download link
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename || `download.${kind}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        setNotification(`Downloaded ${filename || 'file'} successfully!`);
      } catch (error) {
        console.error('Download failed:', error);
        setNotification('Download failed. Please try again.');
      }
    };

    // Get appropriate icon based on file type
    const getFileIcon = () => {
      switch (kind) {
        case 'pdf':
          return <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8"/></svg>;
        case 'docx':
          return <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6 M16 13H8 M16 17H8 M10 9H8"/></svg>;
        case 'png':
        case 'jpg':
          return <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>;
        default:
          return <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z M14 2v6h6"/></svg>;
      }
    };

    return (
      <button
        onClick={handleDownload}
        disabled={isLoading}
        className={`text-sm text-white px-3 py-1.5 rounded-lg transition-colors shadow-sm ${
          isLoading ? 'opacity-50 cursor-not-allowed' : ''
        } ${className}`}
        style={{ backgroundColor: isLoading ? 'rgb(180, 180, 180)' : 'rgb(139, 115, 85)' }}
        onMouseEnter={(e) => {
          if (!isLoading) {
            e.currentTarget.style.backgroundColor = 'rgb(120, 100, 75)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isLoading) {
            e.currentTarget.style.backgroundColor = 'rgb(139, 115, 85)';
          }
        }}
        title={`Download as ${(kind || 'file').toUpperCase()}`}
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-amber-300 border-t-amber-600 rounded-full animate-spin"></div>
            <span>Preparing...</span>
          </>
        ) : (
          <>
            {getFileIcon()}
            <span>Download {(kind || 'file').toUpperCase()}</span>
          </>
        )}
      </button>
    );
  };

  const [notification, setNotification] = useState<string | null>(null)
  // Enhanced chat types and state
  interface ChatMessage {
    id?: string;
    role: 'user' | 'assistant';
    content: string;
    timestamp?: string;
    assistant?: {
      type: 'download' | 'analysis' | 'summary' | 'display';
      kind?: 'pdf' | 'docx' | 'text' | 'png' | 'jpg' | 'txt';
      filename?: string;
      data?: string; // Base64 encoded file data
      base64?: string; // Alternative base64 field
      mime?: string;
    };
  }

  interface Chat {
    id: string;
    file: DriveProposal;
    messages: ChatMessage[];
    originalFiles?: any[];
  }

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [chats, setChats] = useState<Record<string, Chat>>({})
  const [selectedChatId, setSelectedChatId] = useState<string | null>(null)
  const [downloads, setDownloads] = useState<Record<string, boolean>>({})
  const [loadingStates, setLoadingStates] = useState<{[key: string]: boolean}>({})
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null)
  const [showDocumentSelector, setShowDocumentSelector] = useState(false)
  const [currentQuickAction, setCurrentQuickAction] = useState<string | null>(null)
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [multiFileQuery, setMultiFileQuery] = useState<string>('')
  const [multiFileOutputType, setMultiFileOutputType] = useState<string>('detailed')
  const [multiFileFormat, setMultiFileFormat] = useState<string>('')
  const [quickActionLoading, setQuickActionLoading] = useState(false)
  const [duplicateResolution, setDuplicateResolution] = useState<{ [key: string]: boolean }>({})
  const [duplicateResolutionLoading, setDuplicateResolutionLoading] = useState(false)
  const [viewAllProposals, setViewAllProposals] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchStats, setSearchStats] = useState<{total_searched: number, matches_found: number} | null>(null)
  const [visualSearchImage, setVisualSearchImage] = useState<File | null>(null)
  const [visualSearchPreview, setVisualSearchPreview] = useState<string | null>(null)
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
          id: file.id, 
          file: { ...file },
          messages: [] 
        } 
      };
    });
    setDriveAnalyzedId(file.id);
    return file.id;
  }

  // Append a message to a specific chat
  const appendToChat = (chatId: string, message: { role: 'user'|'assistant', content: string, assistant?: any }) => {
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
      const [propsResp, categoriesResp] = await Promise.all([
        call('/drive/proposals'),
        call('/drive/categories')
      ]);      
      const proposals = Array.isArray(propsResp) ? propsResp : [];
      const categories = Array.isArray(categoriesResp) ? categoriesResp : [];
      setDriveProps(proposals);

      if (proposals.length === 0) {
        setCats(categories);
        return;
      }
      // Create a map of category names to their folder info
      const categoryInfoMap = new Map();
      categories.forEach((cat: any) => {
        categoryInfoMap.set(cat.name, {
          folder_id: cat.folder_id,
          missing_folder: cat.missing_folder || false
        });
      });

      // Extract unique categories from proposals
      const categoryMap = new Map<string, number>();
      proposals.forEach(file => {
        const category = getProposalCategory(file) || 'Uncategorized';
        categoryMap.set(category, (categoryMap.get(category) || 0) + 1);
      });

      // Fetch existing files for all folders in parallel
      const folderContents = new Map();
      const folderPromises = categories
        .filter((cat: any) => cat.folder_id)
        .map(async (cat: any) => {
          try {
            const r = await call(`/drive/folder_contents?folderId=${encodeURIComponent(cat.folder_id)}&limit=500`);
            folderContents.set(cat.name, Array.isArray(r?.files) ? r.files : []);
          } catch (_) {
            folderContents.set(cat.name, []);
          }
        });
      
      await Promise.all(folderPromises);

      // Process all categories, including those with and without folders
      const allCategories = new Set([
        ...Array.from(categoryMap.keys()),
        ...Array.from(categoryInfoMap.keys())
      ]);

      const categoriesWithFiles = await Promise.all(
        Array.from(allCategories).map(async (categoryName) => {
          const info = categoryInfoMap.get(categoryName) || {};
          const existing = folderContents.get(categoryName) || [];
          // Get proposed files for this category
          const proposedRaw = proposals
            .filter(file => {
              const fileCategory = getProposalCategory(file) || 'Uncategorized';
              return fileCategory === categoryName;
            })
            .map(file => ({
              id: (file as any).id,
              name: (file as any).name,
              mimeType: (file as any).mimeType
            }));

          // Dedupe: remove proposed items already existing in the folder
          const existingIdSet = new Set(existing.map((f: any) => String(f.id)));
          const existingNameSet = new Set(
            existing.map((f: any) => String(f.name).trim().toLowerCase())
          );
          const proposed = proposedRaw.filter((p: any) => {
            if (p?.id && existingIdSet.has(String(p.id))) return false;
            const nm = String(p?.name || '').trim().toLowerCase();
            if (nm && existingNameSet.has(nm)) return false;
            return true;
          });
          return {
            name: categoryName,
            count: proposed.length,
            folder_id: info.folder_id || null,
            missing_folder: info.missing_folder || false,
            existing: existing.map((f: any) => ({
              id: f.id,
              name: f.name,
              mimeType: f.mimeType
            })),
            proposed
          };
        })
      );

      // Sort categories by name for consistent display
      categoriesWithFiles.sort((a, b) => a.name.localeCompare(b.name));
      setCats(categoriesWithFiles);
    } catch (error) {
      console.error('Error refreshing categories:', error);
      // If there's an error, still try to show what we have
      setCats([]);
    } finally {
      setLoading('refreshCats', false);
    }
  }

  // New: Scan Drive via the extension pipeline (organize with move:false) to populate proposals
  async function handleScan() {
    try {
      // Open Google Drive in a new tab/window for user visibility 
      window.open('https://drive.google.com/drive/my-drive', '_blank');
    } catch {}
    setIsScanning(true)
    setScanProgress(0)
    setScanStatus('Scanning...')
    try {
      // Ask gateway for existing proposals; 
      await refreshDrive()
      
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
      }, 15000)
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
    const replaceTypingWith = (text: string, assistantData?: any) => {
      console.log('replaceTypingWith called with:', { text: text.substring(0, 100) + '...', assistantData });
      setChats(prev => {
        const chat = prev[selectedChatId!]
        if (!chat) return prev
        const msgs = [...chat.messages]
        if (msgs.length && msgs[msgs.length - 1].content === '::typing::' && msgs[msgs.length - 1].role === 'assistant') {
          msgs[msgs.length - 1] = { role: 'assistant', content: text, assistant: assistantData }
        } else {
          msgs.push({ role: 'assistant', content: text, assistant: assistantData })
        }
        console.log('replaceTypingWith: Updated message with assistant data:', msgs[msgs.length - 1]);
        return { ...prev, [selectedChatId!]: { ...chat, messages: msgs } }
      })
    }    
    try {
      const chat = chats[selectedChatId]
      const file = chat.file
      
      // Special handling for multi-file analysis chats
      if (file?.id && String(file.id).startsWith('multi_')) {
        // For multi-file chats, get the original selected files from the chat
        const multiFileChat = chats[selectedChatId]
        const originalFiles = multiFileChat.originalFiles || []        
        if (originalFiles.length === 0) {
          replaceTypingWith('Unable to find original files for this multi-file analysis. Please create a new multi-file analysis.')
          return
        }        
        const response = await call('/analyze_multi', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            files: originalFiles,
            query: q,
            output_type: 'detailed'
          })
        })
        
        if (response?.error) {
          replaceTypingWith(`Error: ${response.error}. Please try again.`)
          return
        }        
        const analysis = response?.analysis || 'No analysis available'
        replaceTypingWith(analysis)
        return
      }      
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
      
      // Check if this is an assistant generation response
      if (response?.assistant) {
        const assistant = response.assistant
        console.log('sendMsg: Received assistant data:', assistant);
        // Pass assistant data to the message
        replaceTypingWith(ans, assistant)
        setAskResult({ answer: ans, assistant: assistant })
      } else {
        console.log('sendMsg: No assistant data in response');
        replaceTypingWith(ans)
        setAskResult(response?.qa || { answer: ans })
      }
      
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
    // Only set loading for non-multi_files actions
    if (action !== 'multi_files') {
      setQuickActionLoading(true)
    }
  }

  const performQuickAction = (action: string, file: DriveProposal) => {
    console.log('Performing quick action:', action)
    console.log('File:', file.name, file.id)
    
    // Set loading state for the quick action
    setQuickActionLoading(true)
    setNotification(`Performing ${action.replace('_', ' ')} on "${file.name}"...`)
    
    // Use existing data and functionality 
    setTimeout(() => {
      let messageContent = ''
      
      switch (action) {
        case 'summarize':
          
          {
            const isDemo = file.id.startsWith('demo-file-')
            ensureChatForFile(file)
            if (isDemo) {
              const demoContent = ` Summary Feature Demo \n\n‚  Feature:  Document Summarization\n ‚  Purpose:  Analyzes document content and provides key insights\n ‚  Usage:  Select a real document to get actual summary\n ‚  Status:  Ready to use with your files\n\n To use with real files:  Upload documents via the Chrome extension or Files tab.`
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
                  content: `I've analyzed the file  "${file.name}" . Here's what I found:\n\n${response.summary}\n\nYou can now ask me questions about this file!`
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
          // Group files by same proposed category 
          const isDemoSimilar = file.id.startsWith('demo-file-')
          if (isDemoSimilar) {
            messageContent = ` Find Similar Files Demo \n\n  Feature:  Similarity Search\n  Purpose:  Groups files by category and content similarity\n  Usage:  Upload files to find actual similar documents\n  Status:  Ready to use with your files\n\n To use with real files:  Upload documents via the Chrome extension or Files tab.`
          } else {
            const similarFiles = driveProps.filter(f => 
              f.id !== file.id && 
              f.proposed_category === file.proposed_category
            )
            messageContent = ` Files similar to "${file.name}" \n\n Category:  ${file.proposed_category}\n\n${similarFiles.length > 0 ? 
              similarFiles.map((f, i) => `${i + 1}. ${f.name}`).join('\n') : 
              'No other files found in the same category.'}\n\n Total similar files:  ${similarFiles.length}`
          }
          break
          
        case 'extract_insights':
          // Show basic file information 
          const isDemoInsights = file.id.startsWith('demo-file-')
          if (isDemoInsights) {
            messageContent = ` Extract Insights Demo \n\n  Feature:  Document Analysis & Insights\n  Purpose:  Extracts key information and patterns from documents\n  Usage:  Upload files to get actual insights\n  Status:  Ready to analyze your documents\n\n To use with real files:  Upload documents via the Chrome extension or Files tab.`
          } else {
            messageContent = ` File Insights for "${file.name}" \n\n  File Name:  ${file.name}\n  Proposed Category:  ${file.proposed_category}\n  Type:  Document file\n  Status:  Available for organization\n\nThis file has been analyzed and categorized. You can approve its organization in the Files tab.`
          }
          break
          
        case 'organize':
          // Redirect to Files tab for organization
          const isDemoOrganize = file.id.startsWith('demo-file-')
          if (isDemoOrganize) {
            messageContent = ` Organize Files Demo \n\n  Feature:  File Organization\n  Purpose:  Automatically categorizes and organizes documents\n  Usage:  Upload files to organize them into folders\n  Status:  Ready to organize your documents\n\n To use with real files:  Upload documents via the Chrome extension, then use the Files tab to approve organization.`
          } else {
            setTab('drive')
            messageContent = ` Organization for "${file.name}" \n\n  Target Category:  ${file.proposed_category}\n  Action Required:  Please go to the Files tab to approve this file's organization\n  Status:  Redirected to Files tab\n\nYou can now find this file in the Files tab and click "Approve" to organize it properly.`
          }
          break
          
        case 'multi_files':
          // Always set up multi-file analysis 
          // Set up multi-file analysis
            setCurrentQuickAction('multi_files')
            setShowDocumentSelector(true)
            setQuickActionLoading(false)  // Reset loading state
            setSelectedFiles([])  // Reset selected files
            setMultiFileQuery('')  // Reset query
            setNotification('Select multiple files for analysis...')
            return
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
        <div className="notification-toast notification-info">
          <div className="notification-content">
            <div className="notification-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="white">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
              </svg>
            </div>
            <span>{notification}</span>
          </div>
          <button 
            onClick={() => setNotification(null)}
            className="notification-close"
            title="Close notification"
          >
            ×
          </button>
        </div>
      )}
      {/*  Background Effects */}
      <div className="professional-shapes">
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
        <div className="professional-shape"></div>
      </div>
      
      {/* Particle System */}
      <div className="professional-particles" id="professional-particles"></div>
      
      {/* Wave Effects */}
      <div className="professional-waves">
        <div className="professional-wave"></div>
        <div className="professional-wave"></div>
        <div className="professional-wave"></div>
      </div>
      
      {/*  Floating Orbs */}
      <div className="beautiful-orb"></div>
      <div className="beautiful-orb"></div>
      <div className="beautiful-orb"></div>
      
      {/* Mouse Trail */}
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
                ['search','Find Files'],
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
        {/*  Notification - moved to global fixed container at bottom */}
        <div className="page-transition">
          {tab==='dashboard' && (
            <Section>
              {/*  */}
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
                      <h3 className="card-heading mb-3">Go Beyond Storage</h3>
                      <p className="card-description">Summarize, Analyze, and Understand your documents</p>
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
                  
                  {/*  Progress Display */}
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
              
              {/*  Stats Section */}
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
              
              {/*  Document Proposals Section */}
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
    <div className="button-group space-x-2">
      <Button 
        tone='accent' 
        onClick={async () => {
          // Get all unapproved files
          const unapprovedFiles = driveProps.filter(f => !f.approved);
          if (unapprovedFiles.length === 0) {
            setNotification('All files are already approved');
            return;
          }
          
          setLoading('approveAll', true);
          try {
            let successCount = 0;
            for (const file of unapprovedFiles) {
              const label = (customLabels[file.id] && customLabels[file.id].trim()) 
                ? customLabels[file.id].trim() 
                : file.proposed_category;
                
              if (!label) continue;
              
              try {
                // First ensure the folder exists
                await call('/drive/create_folder', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ name: label })
                });
                
                // Then approve the file
                const response = await call('/drive/approve', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ 
                    file: { 
                      id: file.id, 
                      name: file.name, 
                      mimeType: file.mimeType || '', 
                      parents: file.parents || [] 
                    }, 
                    category: label 
                  })
                });
                
                if (!response?.error) {
                  successCount++;
                  // Update the file in the local state
                  setDriveProps(prev => 
                    prev.map(f => 
                      f.id === file.id 
                        ? { 
                            ...f, 
                            approved: true, 
                            final_category: response.file?.final_category || label 
                          } 
                        : f
                    )
                  );
                }
              } catch (error) {
                console.error(`Error approving file ${file.name}:`, error);
              }
            }
            
            if (successCount > 0) {
              setNotification(`Successfully approved ${successCount} file(s)`);
            } else {
              setNotification('Failed to approve any files. Please try again.');
            }
          } finally {
            setLoading('approveAll', false);
          }
        }}
        disabled={isLoading('approveAll') || isLoading('refreshDrive') || driveProps.every(f => f.approved)}
        loading={isLoading('approveAll')}
        className="professional-button"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/>
        </svg>
        Approve All
      </Button>
      <Button 
        tone='secondary' 
        onClick={refreshDrive} 
        disabled={isLoading('refreshDrive') || isLoading('approveAll')} 
        loading={isLoading('refreshDrive')} 
        className="professional-button"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
        </svg>
        Refresh Files
      </Button>
      <Button 
        tone='primary' 
        onClick={handleScan} 
        disabled={isScanning} 
        loading={isScanning} 
        className="professional-button"
      >
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
                      <div key={file.id} className={`drive-file-card fade-in ${driveAnalyzedId === file.id ? 'selected' : ''} ${file.approved ? 'approved-file' : ''}`} style={{animationDelay: `${index * 0.1}s`}}>
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
                              {file.approved ? (
                                <div className="file-approved-badge">
                                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                                  </svg>
                                  Approved to: {file.final_category || file.proposed_category}
                                </div>
                              ) : (
                                `Proposed Category: ${file.proposed_category}`
                              )}
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
                              
                              //  call the backend to analyze the file
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
                                // Check if category was auto-updated and refresh
                                if (response.category_auto_updated) {
                                  console.log('Category was auto-updated, refreshing drive files...');
                                  refreshDrive(); // Refresh to show updated category
                                  setNotification(`File analyzed and category updated to: ${response.category || 'Unknown'}`);
                                } else {
                                  setNotification('File analyzed successfully!');
                                }
                                
                                // Ensure chat exists and append analysis to that chat
                                ensureChatForFile(file);
                                
                                // Switch to AI Assistant tab
                                setTab('ai');
                                
                                if (response.summary) {
                                  appendToChat(file.id, {
                                    role: 'assistant',
                                    content: `I've analyzed the file  "${file.name}" . Here's what I found:\n\n${response.summary}\n\nYou can now ask me questions about this file!`
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
                            <button 
                              className={`file-action-button ${file.approved ? 'approved' : 'approve'}`}
                              onClick={async () => {
                                if (file.approved) return; 
                                
                                const label = (customLabels[file.id] && customLabels[file.id].trim()) ? customLabels[file.id].trim() : file.proposed_category;
                                if (!label) { 
                                  setNotification('Please provide a category'); 
                                  return;
                                }
                                
                                // Ensure folder exists (or create it), then approve using that label
                                console.log('Creating/ensuring folder exists for label:', label);
                                try {
                                  await call('/drive/create_folder', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ name: label })
                                  });
                                  
                                  const response = await call('/drive/approve', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ 
                                      file: { 
                                        id: file.id, 
                                        name: file.name, 
                                        mimeType: file.mimeType || '', 
                                        parents: file.parents || [] 
                                      }, 
                                      category: label 
                                    })
                                  });
                                  
                                  if (response?.error) {
                                    setNotification(`Error: ${response.error}`);
                                  } else {
                                    // Update the file in the local state instead of refreshing the whole list
                                    setDriveProps(prev => 
                                      prev.map(f => 
                                        f.id === file.id 
                                          ? { 
                                              ...f, 
                                              approved: true, 
                                              final_category: response.file?.final_category || label 
                                            } 
                                          : f
                                      )
                                    );
                                    setNotification(`Approved to "${label}"`);
                                  } 
                                } catch (error) {
                                  console.error('Error approving file:', error);
                                  setNotification('Error approving file. Please try again.');
                                }
                              }}
                              disabled={file.approved}
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                              </svg>
                              {file.approved ? 'Approved' : 'Approve'}
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
                              Drive folder available Ã‚Â· <a className="category-drive-link" href={`https://drive.google.com/drive/folders/${cat.folder_id}`} target="_blank" rel="noreferrer">Open in Drive</a>
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
                
                <div className="flex gap-6 h-[calc(100vh-200px)]">
                  {/* Chat Sidebar */}
                  <div className="w-80 flex-shrink-0">
                    <div className="professional-card h-full flex flex-col">
                      <div className="card-heading mb-4">Active Chats</div>
                      <div className="flex-1 overflow-y-auto space-y-2">
                        {Object.keys(chats).length === 0 ? (
                          <div className="text-center py-8">
                            <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-beige-100 to-beige-300 rounded-full flex items-center justify-center">
                              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-beige-700">
                                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z"/>
                              </svg>
                            </div>
                            <p className="text-gray-500 text-sm">Analyze a file to start chatting</p>
                          </div>
                        ) : (
                          Object.values(chats).map(({ file }) => (
                            <button
                              key={file.id}
                              onClick={() => { setSelectedChatId(file.id); setDriveAnalyzedId(file.id) }}
                              className={`w-full text-left p-3 rounded-lg transition-all ${
                                selectedChatId === file.id 
                                  ? 'bg-gradient-to-r from-amber-100 to-amber-200 border border-amber-300' 
                                  : 'hover:bg-gray-50 border border-transparent'
                              }`}
                            >
                              <div className="font-medium text-gray-800 truncate">{file.name}</div>
                              <div className="text-xs text-gray-500 mt-1">Active chat</div>
                            </button>
                          ))
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Main Chat Area */}
                  <div className="flex-1 flex flex-col">
                    <div className="professional-card flex-1 flex flex-col overflow-hidden">
                      {/* Chat Header with Export Options */}
                      <div className="bg-gradient-to-r from-stone-50 to-stone-100 p-4 border-b border-gray-200 flex justify-between items-center">
                        <div>
                          <h3 className="font-semibold text-gray-800">
                            {selectedChatId && chats[selectedChatId] ? chats[selectedChatId].file.name : 'AI Assistant'}
                          </h3>
                          <p className="text-sm text-gray-500">Ask questions about your document</p>
                        </div>
                        
                        {/* Chat Info */}
                        {selectedChatId && chats[selectedChatId] && chats[selectedChatId].messages.length > 0 && (
                          <div className="text-xs text-gray-500">
                            {chats[selectedChatId].messages.length} messages
                          </div>
                        )}
                      </div>

                      {/* Messages Area - Single Scrollbar */}
                      <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        {!selectedChatId || !chats[selectedChatId] || chats[selectedChatId].messages.length === 0 ? (
                          <div className="flex flex-col items-center justify-center h-full text-center">
                            <div className="w-20 h-20 bg-gradient-to-br from-grey-100 to-grey-200 rounded-full flex items-center justify-center mb-4">
                              <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor" className="text-grey-600">
                                <path d="M20 2H4C2.9 2 2 2.9 2 4V22L6 18H20C21.1 18 22 17.1 22 16V4C22 2.9 21.1 2 20 2Z"/>
                              </svg>
                            </div>
                            <h4 className="text-lg font-medium text-gray-700 mb-2">Ready to Chat</h4>
                            <p className="text-gray-500">Select a file from the sidebar or analyze a new document to begin</p>
                          </div>
                            ) : (
                          <>
                            {chats[selectedChatId].messages.filter(m => m.content !== '::typing::').map((msg, index) => (
                              <div key={index} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                {msg.role === 'user' ? (
                                  <div className="flex items-end gap-2 max-w-[70%]">
                                    <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-3 rounded-2xl rounded-br-md">
                                      <p className="text-sm">{msg.content}</p>
                                    </div>
                                    <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-blue-600 rounded-full flex items-center justify-center text-white text-xs font-semibold">
                                      U
                                    </div>
                                  </div>
                                ) : (
                                  <div className="flex items-start gap-2 max-w-[80%]">
                                    <div className="w-8 h-8 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center text-white text-xs font-semibold">
                                      AI
                                    </div>
                                    <div className="bg-white border border-gray-200 p-3 rounded-2xl rounded-bl-md shadow-sm">
                                      <p className="text-sm text-gray-800">{msg.content}</p>
                                      
                                      {/* AI Capabilities Info */}
                                      {selectedChatId && !msg.content.includes('::typing::') && msg.role === 'assistant' && (
                                        <div className="mt-3 pt-3 border-t border-gray-100">
                                          <div className="text-xs text-gray-500 leading-relaxed">
                                            <span className="font-medium text-gray-600">Ask me to create:</span>
                                            <span className="text-gray-500"> Flowcharts • Notes • Timelines • Insights • Summaries • Q&A</span>
                                            <br />
                                            <span className="text-gray-400">Available formats: PDF, Word, Text, Images</span>
                                          </div>
                                        </div>
                                      )}

                                      {/*  Download Button */}
                                      <DownloadButton message={msg} className="mt-3" />
                                      
                                      {/* Fallback Download for Flowcharts */}
                                      {!msg.assistant?.data && !msg.assistant?.base64 && msg.content && msg.content.includes('flowchart') && (
                                        <div className="mt-3">
                                          <button 
                                            className="professional-btn professional-btn-primary text-sm"
                                            type="button"
                                            onClick={() => {
                                              // Direct download 
                                              const mermaidUrl = 'https://mermaid.ink/img/Zmxvd2NoYXJ0IFRECiAgICBBW1N0YXJ0XSAtLT4gQltQcmVwYXJhdGlvbl0KICAgIEIgLS0-IENbIkJlZ2luIHByb2Nlc3MiXQogICAgQyAtLT4gRFsiRXhlY3V0ZSBtYWluIHRhc2tzIl0KICAgIEQgLS0-IEVbIkNvbXBsZXRlIG9iamVjdGl2ZXMiXQogICAgRSAtLT4gRltDb21wbGV0ZV0=';
                                              
                                              fetch(mermaidUrl)
                                                .then(response => response.blob())
                                                .then(blob => {
                                                  const url = URL.createObjectURL(blob);
                                                  const a = document.createElement('a');
                                                  a.href = url;
                                                  a.download = 'flowchart.png';
                                                  document.body.appendChild(a);
                                                  a.click();
                                                  document.body.removeChild(a);
                                                  URL.revokeObjectURL(url);
                                                  setNotification('Download started');
                                                })
                                                .catch(() => {
                                                  setNotification('Download failed');
                                                });
                                            }}
                                          >
                                            Download Flowchart
                                          </button>
                                        </div>
                                      )}
                                      
                                    </div>
                                  </div>
                                )}
                              </div>
                            ))}

                            {/* Typing Indicator */}
                            {selectedChatId && chats[selectedChatId].messages.some(m => m.content === '::typing::') && (
                              <div className="flex items-start gap-2">
                                <div className="w-8 h-8 bg-gradient-to-br from-amber-400 to-orange-500 rounded-full flex items-center justify-center text-white text-xs font-semibold">
                                  AI
                                </div>
                                <div className="bg-white border border-gray-200 p-3 rounded-2xl rounded-bl-md shadow-sm">
                                  <div className="flex items-center gap-2">
                                    <div className="flex gap-1">
                                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                                    </div>
                                    <span className="text-xs text-gray-500">Typing...</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </>
                        )}
                      </div>

                      {/* Input Area */}
                      <div className="border-t border-gray-200 p-4">
                        <div className="flex items-center gap-3 bg-gray-50 rounded-full px-4 py-2 border border-gray-200 focus-within:border-blue-300 focus-within:ring-2 focus-within:ring-blue-100">
                          <input
                            ref={inputRef}
                            type="text"
                            value={askInput}
                            onChange={(e) => setAskInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && sendMsg()}
                            placeholder="Ask about this file..."
                            className="flex-1 bg-transparent border-none outline-none text-gray-800 placeholder-gray-500"
                            disabled={!selectedChatId}
                          />
                          <button 
                            className="w-8 h-8 bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white rounded-full flex items-center justify-center transition-all disabled:opacity-50 disabled:cursor-not-allowed" 
                            onClick={sendMsg} 
                            disabled={!askInput.trim() || !selectedChatId}
                            title="Send message"
                          >
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M2.01 21L23 12L2.01 3L2 10L17 12L2 14L2.01 21Z"/>
                            </svg>
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quick Actions Sidebar */}
                  <div className="w-80 flex-shrink-0">
                    <div className="professional-card h-full">
                      <div className="card-heading mb-4">Quick Actions</div>
                      <div className="space-y-3">
                        <Button tone='secondary' className="w-full justify-start professional-btn professional-btn-secondary" onClick={() => handleQuickAction('summarize')} disabled={quickActionLoading} loading={quickActionLoading}>
                          {quickActionLoading ? (
                            <div className="loading-dots">
                              <div className="loading-dot"></div>
                              <div className="loading-dot"></div>
                              <div className="loading-dot"></div>
                            </div>
                          ) : (
                            <>
                              <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z"/>
                                <path d="M14 2V8H20"/>
                              </svg>
                              Summarize Documents
                            </>
                          )}
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-btn professional-btn-secondary" onClick={() => handleQuickAction('find_similar')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M15.5 14H14.71L14.43 13.73C15.41 12.59 16 11.11 16 9.5C16 5.91 13.09 3 9.5 3S3 5.91 3 9.5S5.91 16 9.5 16C11.11 16 12.59 15.41 13.73 14.43L14 14.71V15.5L19 20.49L20.49 19L15.5 14ZM9.5 14C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5S14 7.01 14 9.5S11.99 14 9.5 14Z"/>
                          </svg>
                          Find Similar Files
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-btn professional-btn-secondary" onClick={() => handleQuickAction('extract_insights')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>
                          </svg>
                          Extract Insights
                        </Button>
                        <Button tone='secondary' className="w-full justify-start professional-btn professional-btn-secondary" onClick={() => handleQuickAction('organize')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M10 4H4C2.89 4 2 4.89 2 6V18C2 19.11 2.89 20 4 20H20C21.11 20 22 19.11 22 18V8C22 6.89 21.11 6 20 6H12L10 4Z"/>
                          </svg>
                          Organize Files
                        </Button>
                        <Button tone='accent' className="w-full justify-start professional-btn professional-btn-accent" onClick={() => handleQuickAction('multi_files')} disabled={quickActionLoading} loading={quickActionLoading}>
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M14 2H6C4.9 2 4 2.9 4 4V20C4 21.1 4.89 22 5.99 22H18C19.1 22 20 21.1 20 20V8L14 2Z"/>
                            <path d="M16 0H8C6.9 0 6 0.9 6 2V18C6 19.1 6.89 20 7.99 20H20C21.1 20 22 19.1 22 18V6L16 0Z"/>
                          </svg>
                          Multi-Files Analysis
                        </Button>
                      </div>
                    </div>
                    
                  </div>
                </div>
              </div>
            </Section>
          )}

          {tab === 'search' && (
            <SearchFiles />
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
                    {currentQuickAction === 'multi_files' ? 'Select Multiple Documents' : 'Select a Document'}
                  </h3>
                  <p className="quick-action-subtitle">
                    {currentQuickAction === 'multi_files' ? 'Choose multiple documents to analyze together' : `Choose a document to ${currentQuickAction?.replace('_', ' ')}`}
                  </p>
                </div>
                <button 
                  onClick={() => {
                    setShowDocumentSelector(false)
                    setQuickActionLoading(false)
                    setSelectedFiles([])  // Reset selected files
                    setMultiFileQuery('')  // Reset query
                    setMultiFileOutputType('detailed')  // Reset output type
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
              {currentQuickAction === 'multi_files' && (
                <div className="mb-4 space-y-3 p-4 bg-gray-50 rounded-lg">
                  <div>
                    <label className="block text-sm font-medium mb-2">Query:</label>
                    <input
                      type="text"
                      value={multiFileQuery}
                      onChange={(e) => setMultiFileQuery(e.target.value)}
                      placeholder="What would you like to analyze across these files?"
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Output Type:</label>
                    <select
                      value={multiFileOutputType}
                      onChange={(e) => setMultiFileOutputType(e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="detailed">Detailed Report</option>
                      <option value="flowchart">Flowchart</option>
                      <option value="timeline">Timeline</option>
                      <option value="short_notes">Short Notes</option>
                      <option value="key_insights">Key Insights</option>
                      <option value="flashcards">Flashcards</option>
                    </select>
                  </div>
                  {['flowchart', 'flashcards', 'key_insights', 'timeline', 'short_notes'].includes(multiFileOutputType) && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Download Format (Optional):</label>
                      <select
                        value={multiFileFormat}
                        onChange={(e) => setMultiFileFormat(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">Display Only</option>
                        <option value="md">Markdown (.md)</option>
                        <option value="txt">Text (.txt)</option>
                        <option value="docx">Word Document (.docx)</option>
                        <option value="pdf">PDF (.pdf)</option>
                      </select>
                    </div>
                  )}
                  <div className="text-sm text-gray-600">
                    Selected files: {selectedFiles.length} / {driveProps.length}
                  </div>
                </div>
              )}
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
                        if (currentQuickAction === 'multi_files') {
                          // Toggle selection for multi-file
                          setSelectedFiles(prev => 
                            prev.includes(file.id) 
                              ? prev.filter(id => id !== file.id)
                              : [...prev, file.id]
                          )
                        } else {
                          // Single file selection
                          setShowDocumentSelector(false)
                          setNotification(`Selected: ${file.name}`)
                          performQuickAction(currentQuickAction!, file)
                        }
                      }}
                      className="document-item"
                      style={{ animationDelay: `${index * 100}ms` }}
                    >
                      <div className="flex items-center">
                        {currentQuickAction === 'multi_files' && (
                          <div className="mr-3">
                            <input
                              type="checkbox"
                              checked={selectedFiles.includes(file.id)}
                              onChange={() => {}}
                              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
                            />
                          </div>
                        )}
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
            {currentQuickAction === 'multi_files' && (
              <div className="multi-file-button-area">
                <button
                  onClick={async () => {
                    if (selectedFiles.length === 0) {
                      setNotification('Please select at least one file')
                      return
                    }
                    if (!multiFileQuery.trim()) {
                      setNotification('Please enter a query')
                      return
                    }
                    
                    setQuickActionLoading(true)
                    setShowDocumentSelector(false)
                    setNotification(`Analyzing ${selectedFiles.length} files...`)
                    
                    try {
                      const selectedFileObjects = driveProps.filter(f => f && f.id && selectedFiles.includes(f.id))
                      
                      if (selectedFileObjects.length === 0) {
                        setNotification('No valid files found for analysis')
                        setQuickActionLoading(false)
                        return
                      }
                      
                      console.log('Selected file objects:', selectedFileObjects)
                      
                      const response = await call('/analyze_multi', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          files: selectedFileObjects,
                          query: multiFileQuery,
                          output_type: multiFileOutputType,
                          format: multiFileFormat || undefined
                        })
                      })
                      
                      // Create a combined chat for multi-file analysis
                      const multiFileId = `multi_${Date.now()}`
                      let analysisContent = ` Multi-File Analysis Results \n\n Query:  ${multiFileQuery}\n Output Type:  ${multiFileOutputType}\n Files Analyzed:  ${selectedFiles.length}\n\n Analysis: \n\n${response.analysis}`
                      
                      // Add download info if assistant data is available
                      if (response.assistant && response.assistant.type === 'download') {
                        analysisContent += `\n\n 📥 Downloadable ${response.assistant.kind.replace('_', ' ')} available as ${response.assistant.filename} `
                      }
                      
                      // Create a dummy file object for the multi-file chat with original files stored
                      const multiFileObject = {
                        id: multiFileId,
                        name: `Multi-File Analysis (${selectedFiles.length} files)`,
                        proposed_category: 'Multi-File Analysis',
                        approved: false,
                        originalFiles: selectedFileObjects, // Store original files for follow-up questions
                        originalQuery: multiFileQuery,
                        originalOutputType: multiFileOutputType
                      }
                      
                      setChats(prev => ({
                        ...prev,
                        [multiFileId]: {
                          id: multiFileId,
                          file: multiFileObject,
                          originalFiles: selectedFileObjects, // Store original files for follow-up questions
                          messages: [
                            { role: 'assistant', content: analysisContent, assistant: response.assistant }
                          ]
                        }
                      }))
                      
                      setSelectedChatId(multiFileId)
                      setTab('ai')
                      setNotification('Multi-file analysis completed!')
                      
                      // Reset state
                      setSelectedFiles([])
                      setMultiFileQuery('')
                      setMultiFileOutputType('detailed')
                      
                    } catch (error: any) {
                      setNotification(`Analysis failed: ${error?.message || 'Unknown error'}`)
                    } finally {
                      setQuickActionLoading(false)
                    }
                  }}
                  disabled={selectedFiles.length === 0 || !multiFileQuery.trim() || quickActionLoading}
                  className="multi-file-analyze-button"
                  title={
                    selectedFiles.length === 0 
                      ? 'Please select at least one file' 
                      : !multiFileQuery.trim() 
                        ? 'Please enter a query' 
                        : quickActionLoading 
                          ? 'Analysis in progress...'
                          : 'Click to start analysis'
                  }
                >
                  {quickActionLoading 
                    ? 'Analyzing...' 
                    : selectedFiles.length === 0 
                      ? 'Select Files First' 
                      : !multiFileQuery.trim()
                        ? 'Enter Query First'
                        : `SUBMIT (${selectedFiles.length} File${selectedFiles.length !== 1 ? 's' : ''})`
                  }
                </button>
              </div>
            )}
          </div>
        </div>
      ) : null}
    </div>
  )
}