import React, { useState, useEffect, useRef } from 'react';
import { Search, X, FileText, Image as ImageIcon, Music, Film, FileSpreadsheet, Presentation as FilePresentation, Code, Archive, Mail, BookOpen, Loader2, Sparkles, Eye, MessageCircle, ExternalLink, FolderPlus, Trash2, Move, Check, Square } from 'lucide-react';

interface SearchResult {
  id: string;
  name: string;
  mimeType: string;
  size?: number;
  modifiedTime?: string;
  score: number;
  match_score?: number;
  confidence?: number;
  match_type: string;
  context: string;
  drive_url: string;
}

interface SearchResponse {
  query: string;
  file_type?: string;
  results: SearchResult[];
  search_terms: string[];
}

const BASE = 'http://127.0.0.1:4000';

async function call(path: string, opts?: RequestInit) {
  const res = await fetch(BASE + path, opts ? opts : {});
  if (res.status === 204) return { ok: true };
  const text = await res.text();
  try { return JSON.parse(text) } catch { return { error: 'Invalid JSON', raw: text } }
}

const SearchFiles: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchStats, setSearchStats] = useState<{ total: number; shown: number; matchRate: number } | null>(null);
  const [selectedFileType, setSelectedFileType] = useState<string>('all');
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const searchTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  
  // Bulk operations state
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [showFolderModal, setShowFolderModal] = useState(false);
  const [showNewFolderModal, setShowNewFolderModal] = useState(false);
  const [existingFolders, setExistingFolders] = useState<{id: string, name: string}[]>([]);
  const [newFolderName, setNewFolderName] = useState('');
  const [bulkOperationLoading, setBulkOperationLoading] = useState(false);
  const [notification, setNotification] = useState<string | null>(null);

  // File type options with icons
  const fileTypes = [
    { id: 'all', name: 'All Files', icon: <FileText className="w-4 h-4" /> },
    { id: 'document', name: 'Documents', icon: <FileText className="w-4 h-4" /> },
    { id: 'spreadsheet', name: 'Sheets', icon: <FileSpreadsheet className="w-4 h-4" /> },
    { id: 'presentation', name: 'Slides', icon: <FilePresentation className="w-4 h-4" /> },
    { id: 'pdf', name: 'PDFs', icon: <FileText className="w-4 h-4" /> },
    { id: 'image', name: 'Images', icon: <ImageIcon className="w-4 h-4" /> },
    { id: 'video', name: 'Videos', icon: <Film className="w-4 h-4" /> },
    { id: 'audio', name: 'Audio', icon: <Music className="w-4 h-4" /> },
  ];

  // Format file size
  const formatFileSize = (bytes?: number): string => {
    if (!bytes) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
   
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
   
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  // Format date
  const formatDate = (dateString?: string): string => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };


  // Get file icon based on mime type
  const getFileIcon = (mimeType: string) => {
    if (!mimeType) return <FileText className="w-6 h-6" />;
    
    if (mimeType.includes('spreadsheet') || mimeType.includes('excel') || mimeType.includes('sheet')) {
      return <FileSpreadsheet className="w-6 h-6" />;
    } else if (mimeType.includes('document') || mimeType.includes('word') || mimeType.includes('text/plain') || mimeType.endsWith('pdf')) {
      return <FileText className="w-6 h-6" />;
    } else if (mimeType.includes('presentation') || mimeType.includes('powerpoint')) {
      return <FilePresentation className="w-6 h-6" />;
    } else if (mimeType.startsWith('image/')) {
      return <ImageIcon className="w-6 h-6" />;
    } else if (mimeType.startsWith('audio/')) {
      return <Music className="w-6 h-6" />;
    } else if (mimeType.startsWith('video/')) {
      return <Film className="w-6 h-6" />;
    } else if (mimeType.includes('code') || mimeType.includes('javascript') || mimeType.includes('python') || mimeType.includes('json') || mimeType.includes('xml')) {
      return <Code className="w-6 h-6" />;
    } else if (mimeType.includes('zip') || mimeType.includes('compressed') || mimeType.includes('archive')) {
      return <Archive className="w-6 h-6" />;
    } else if (mimeType.includes('email') || mimeType.includes('message')) {
      return <Mail className="w-6 h-6" />;
    } else if (mimeType.includes('pdf')) {
      return <FileText className="w-6 h-6" />;
    } else {
      return <FileText className="w-6 h-6" />;
    }
  };

  // Handle search
  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!query.trim()) return;
   
    setIsSearching(true);
    setShowSuggestions(false);
    setHasSearched(true);
   
    try {
      let response: Response;
     
      // Handle text search - Use the search_files endpoint from your backend
        // Try to get auth token from localStorage or URL params
        let authToken = localStorage.getItem('drive_token') || '';
        
        // If no token in localStorage, try to get from URL params
        if (!authToken) {
          const urlParams = new URLSearchParams(window.location.search);
          authToken = urlParams.get('auth_token') || '';
        }
        
        console.log('SearchFiles: Using auth token:', authToken ? 'Token found' : 'No token');
        
        const searchUrl = `http://localhost:4000/search_files${authToken ? `?auth_token=${encodeURIComponent(authToken)}` : ''}`;
        console.log('SearchFiles: Making request to:', searchUrl);
        console.log('SearchFiles: Request body:', { query, file_type: selectedFileType === 'all' ? null : selectedFileType, limit: 50 });
        
        response = await fetch(searchUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query: query,
            file_type: selectedFileType === 'all' ? null : selectedFileType,
            limit: 50,
          }),
        });
     
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
     
      const data = await response.json();
      console.log('SearchFiles: Received response data:', data);
      
      // Handle different possible response formats
      let searchResults: SearchResult[] = [];
      let totalMatches = 0;
      
      console.log('SearchFiles: Raw response data structure:', data);
      
      if (data.results && Array.isArray(data.results)) {
        // Standard format: { results: [...], total_matches: N }
        searchResults = data.results;
        totalMatches = data.total_matches || data.results.length;
      } else if (Array.isArray(data)) {
        // Direct array format: [...]
        searchResults = data;
        totalMatches = data.length;
      } else if (data.files && Array.isArray(data.files)) {
        // Alternative format: { files: [...] }
        searchResults = data.files;
        totalMatches = data.total_searched || data.matches_found || data.files.length;
      }
      
      // Log individual results to debug confidence values
      searchResults.forEach((result, index) => {
        console.log(`SearchFiles: Result ${index}:`, {
          name: result.name,
          score: result.score,
          match_score: result.match_score,
          confidence: result.confidence,
          match_type: result.match_type
        });
      });
      
      console.log('SearchFiles: Parsed search results:', searchResults);
      console.log('SearchFiles: Total matches:', totalMatches);
      
      setResults(searchResults);
      
      // Calculate match rate - fix NaN issue
      const shownResults = searchResults.length;
      let matchRate = 0;
      if (totalMatches > 0 && shownResults > 0) {
        matchRate = Math.round((shownResults / totalMatches) * 100);
      } else if (shownResults > 0) {
        matchRate = 100; // If we have results but no total count, assume 100%
      }
      
      console.log('SearchFiles: Setting stats - total:', totalMatches, 'shown:', shownResults, 'rate:', matchRate);
      
      setSearchStats({
        total: totalMatches,
        shown: shownResults,
        matchRate: matchRate
      });
     
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setSearchStats({ total: 0, shown: 0, matchRate: 0 });
    } finally {
      setIsSearching(false);
    }
  };

  // Handle input change with debounce
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setQuery(value);
   
    // Clear any pending search
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }
   
    // Show suggestions if query is not empty
    if (value.trim()) {
      setShowSuggestions(true);
     
      // Generate some example suggestions based on input
      const exampleQueries = [
        `Find documents about ${value}`,
        `Show me ${value} from last month`,
        `Search for ${value} in PDFs`,
        `Find images related to ${value}`,
      ];
     
      setSuggestions(exampleQueries);
    } else {
      setShowSuggestions(false);
      setSuggestions([]);
      setResults([]);
      setSearchStats(null);
      setHasSearched(false);
    }
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch();
  };

  // Clear search
  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setSearchStats(null);
    setShowSuggestions(false);
    setHasSearched(false);
    setSelectedFiles(new Set());
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  // Bulk operations functions
  const toggleFileSelection = (fileId: string) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fileId)) {
        newSet.delete(fileId);
      } else {
        newSet.add(fileId);
      }
      return newSet;
    });
  };

  const selectAllFiles = () => {
    if (selectedFiles.size === results.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(results.map(r => r.id)));
    }
  };

  const getAuthToken = () => {
    return localStorage.getItem('drive_token') || new URLSearchParams(window.location.search).get('auth_token') || '';
  };

  const fetchExistingFolders = async () => {
    try {
      const authToken = getAuthToken();
      const data = await call(`/drive_folders?auth_token=${encodeURIComponent(authToken)}`);
      if (data.success) {
        setExistingFolders(data.folders);
      }
    } catch (error) {
      console.error('Error fetching folders:', error);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedFiles.size === 0) return;
    
    const fileNames = results.filter(r => selectedFiles.has(r.id)).map(r => r.name).join(', ');
    if (!confirm(`Delete ${selectedFiles.size} file(s) from Drive? This action cannot be undone.\n\nFiles: ${fileNames}`)) return;
    
    setBulkOperationLoading(true);
    try {
      const authToken = getAuthToken();
      const data = await call('/bulk_delete_files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_ids: Array.from(selectedFiles),
          auth_token: authToken
        })
      });
      
      if (data.success) {
        setNotification(`Successfully deleted ${data.deleted_count} file(s)`);
        // Remove deleted files from results
        setResults(prev => prev.filter(r => !selectedFiles.has(r.id)));
        setSelectedFiles(new Set());
      } else {
        setNotification('Error deleting files. Please try again.');
      }
    } catch (error) {
      console.error('Bulk delete error:', error);
      setNotification('Error deleting files. Please try again.');
    } finally {
      setBulkOperationLoading(false);
    }
  };

  const handleMoveToExistingFolder = async (folderId: string, folderName: string) => {
    if (selectedFiles.size === 0) return;
    
    setBulkOperationLoading(true);
    try {
      const authToken = getAuthToken();
      const data = await call('/bulk_move_to_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_ids: Array.from(selectedFiles),
          folder_name: folderName,
          auth_token: authToken
        })
      });
      
      if (data.success) {
        setNotification(`Successfully moved ${data.moved_count} file(s) to "${folderName}"`);
        setSelectedFiles(new Set());
        setShowFolderModal(false);
      } else {
        setNotification('Error moving files. Please try again.');
      }
    } catch (error) {
      console.error('Move to folder error:', error);
      setNotification('Error moving files. Please try again.');
    } finally {
      setBulkOperationLoading(false);
    }
  };

  const handleCreateNewFolder = async () => {
    if (selectedFiles.size === 0 || !newFolderName.trim()) return;
    
    setBulkOperationLoading(true);
    try {
      const authToken = getAuthToken();
      const data = await call('/bulk_move_to_folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_ids: Array.from(selectedFiles),
          folder_name: newFolderName.trim(),
          auth_token: authToken,
          create_new: true
        })
      });
      
      if (data.success) {
        setNotification(`Successfully moved ${data.moved_count} file(s) to new folder "${newFolderName}"`);
        setSelectedFiles(new Set());
        setShowNewFolderModal(false);
        setNewFolderName('');
      } else {
        setNotification('Error creating folder and moving files. Please try again.');
      }
    } catch (error) {
      console.error('Create folder error:', error);
      setNotification('Error creating folder and moving files. Please try again.');
    } finally {
      setBulkOperationLoading(false);
    }
  };


  // Handle click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (inputRef.current && !inputRef.current.contains(e.target as Node)) {
        setShowSuggestions(false);
      }
    };
   
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus the input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  // Auto-dismiss notifications
  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => {
        setNotification(null);
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgb(245, 240, 230)' }}>
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Notification */}
      {notification && (
        <div className="fixed top-4 right-4 z-50 text-white px-4 py-2 rounded-lg shadow-lg" style={{ backgroundColor: 'rgb(101, 67, 33)' }}>
          {notification}
        </div>
      )}
      
      {/* Search Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2" style={{ color: 'rgb(101, 67, 33)' }}>Find Your Files</h1>
        <p className="text-lg" style={{ color: 'rgb(139, 115, 85)' }}>Search across your documents, PDFs, and presentations with AI-powered precision</p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div
          className="relative flex items-center rounded-2xl shadow-lg border-2 transition-all duration-300"
          style={{
            backgroundColor: 'rgb(250, 248, 240)',
            borderColor: isFocused ? 'rgb(139, 115, 85)' : 'rgb(210, 180, 140)'
          }}
        >
          <div className="absolute left-4" style={{ color: 'rgb(139, 115, 85)' }}>
            {isSearching ? (
              <Loader2 className="w-5 h-5 animate-spin" style={{ color: 'rgb(139, 115, 85)' }} />
            ) : (
              <Search className="w-5 h-5" />
            )}
          </div>
         
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Search documents, PDFs, presentations, and more..."
            className="w-full pl-12 pr-16 py-4 text-lg bg-transparent border-none focus:ring-0 focus:outline-none placeholder-opacity-75"
            style={{ 
              color: 'rgb(101, 67, 33)'
            }}
            disabled={isSearching}
          />
         
          {query && !isSearching && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-16 p-1 transition-colors"
              style={{ color: 'rgb(139, 115, 85)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgb(101, 67, 33)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgb(139, 115, 85)'}
              aria-label="Clear search"
            >
              <X className="w-5 h-5" />
            </button>
          )}
         
          <div className="absolute right-2">
            <button
              type="submit"
              disabled={isSearching || !query}
              className={`p-2 rounded-full transition-all ${
                isSearching || !query
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'text-white shadow-md'
              }`}
              style={!(isSearching || !query) ? { backgroundColor: 'rgb(139, 115, 85)' } : {}}
              onMouseEnter={(e) => {
                if (!(isSearching || !query)) {
                  e.currentTarget.style.backgroundColor = 'rgb(120, 100, 75)';
                }
              }}
              onMouseLeave={(e) => {
                if (!(isSearching || !query)) {
                  e.currentTarget.style.backgroundColor = 'rgb(139, 115, 85)';
                }
              }}
              title="Search"
            >
              {isSearching ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
            </button>
          </div>
        </div>
       
        {/* Search Suggestions */}
        {showSuggestions && suggestions.length > 0 && (
          <div className="mt-2 rounded-lg shadow-lg border overflow-hidden z-50 relative" style={{ backgroundColor: 'rgb(250, 248, 240)', borderColor: 'rgb(210, 180, 140)' }}>
            <div className="py-1">
              <div className="px-4 py-2 text-xs font-semibold" style={{ color: 'rgb(139, 115, 85)', backgroundColor: 'rgb(245, 240, 230)' }}>
                Try searching for:
              </div>
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="w-full text-left px-4 py-2 flex items-center transition-colors"
                  style={{ color: 'rgb(101, 67, 33)' }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgb(245, 240, 230)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  <Search className="w-4 h-4 mr-2" style={{ color: 'rgb(139, 115, 85)' }} />
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </form>


      {/* Search Stats */}
      {searchStats && hasSearched && (
        <div className="mb-8">
          <div className="flex justify-center">
            <div className="inline-flex items-center gap-8 px-8 py-4 rounded-2xl shadow-sm border" style={{ backgroundColor: 'rgb(250, 248, 240)', borderColor: 'rgb(210, 180, 140)' }}>
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 rounded-full mb-2" style={{ backgroundColor: 'rgb(245, 240, 230)' }}>
                  <Search className="w-6 h-6" style={{ color: 'rgb(139, 115, 85)' }} />
                </div>
                <div className="text-2xl font-bold" style={{ color: 'rgb(101, 67, 33)' }}>{searchStats.total}</div>
                <div className="text-sm" style={{ color: 'rgb(139, 115, 85)' }}>Files Searched</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 rounded-full mb-2" style={{ backgroundColor: 'rgb(245, 240, 230)' }}>
                  <Eye className="w-6 h-6" style={{ color: 'rgb(139, 115, 85)' }} />
                </div>
                <div className="text-2xl font-bold" style={{ color: 'rgb(101, 67, 33)' }}>{searchStats.shown}</div>
                <div className="text-sm" style={{ color: 'rgb(139, 115, 85)' }}>Matches Found</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 rounded-full mb-2" style={{ backgroundColor: 'rgb(245, 240, 230)' }}>
                  <Sparkles className="w-6 h-6" style={{ color: 'rgb(139, 115, 85)' }} />
                </div>
                <div className="text-2xl font-bold" style={{ color: 'rgb(101, 67, 33)' }}>{searchStats.matchRate}%</div>
                <div className="text-sm" style={{ color: 'rgb(139, 115, 85)' }}>Match Rate</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search Results Header with Bulk Actions */}
      {hasSearched && (
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold" style={{ color: 'rgb(101, 67, 33)' }}>
              Search Results
              {query && <span className="font-normal" style={{ color: 'rgb(139, 115, 85)' }}> for "{query}"</span>}
            </h2>
            <div className="text-sm" style={{ color: 'rgb(139, 115, 85)' }}>
              Showing {results.length} results
            </div>
          </div>
          
          {/* Bulk Actions Bar */}
          {results.length > 0 && (
            <div className="mt-4 p-4 rounded-lg border" style={{ backgroundColor: 'rgb(245, 240, 230)', borderColor: 'rgb(210, 180, 140)' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <button
                    onClick={selectAllFiles}
                    className="flex items-center gap-2 px-3 py-2 text-sm border rounded-md transition-colors"
                    style={{ 
                      backgroundColor: 'rgb(250, 248, 240)', 
                      borderColor: 'rgb(210, 180, 140)',
                      color: 'rgb(101, 67, 33)'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgb(245, 240, 230)'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'rgb(250, 248, 240)'}
                  >
                    <div className="w-4 h-4 rounded border-2 flex items-center justify-center" style={{
                      backgroundColor: selectedFiles.size === results.length ? 'rgb(101, 67, 33)' : 'transparent',
                      borderColor: selectedFiles.size === results.length ? 'rgb(101, 67, 33)' : 'rgb(139, 115, 85)'
                    }}>
                      {selectedFiles.size === results.length && (
                        <Check className="w-3 h-3 text-white" />
                      )}
                    </div>
                    {selectedFiles.size === results.length ? 'Deselect All' : 'Select All'}
                  </button>
                  
                  {selectedFiles.size > 0 && (
                    <span className="text-sm" style={{ color: 'rgb(139, 115, 85)' }}>
                      {selectedFiles.size} file{selectedFiles.size !== 1 ? 's' : ''} selected
                    </span>
                  )}
                </div>
                
                {selectedFiles.size > 0 && (
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => {
                        fetchExistingFolders();
                        setShowFolderModal(true);
                      }}
                      disabled={bulkOperationLoading}
                      className="flex items-center gap-2 px-4 py-2 text-sm text-black rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium"
                      style={{ backgroundColor: 'rgb(101, 67, 33)' }}
                      onMouseEnter={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(92, 57, 25)';
                          e.currentTarget.style.transform = 'translateY(-1px)';
                          e.currentTarget.style.boxShadow = '0 4px 12px rgba(101, 67, 33, 0.3)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(101, 67, 33)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
                        }
                      }}
                    >
                      <Move className="w-4 h-4" />
                      Move to Folder
                    </button>
                    
                    <button
                      onClick={() => setShowNewFolderModal(true)}
                      disabled={bulkOperationLoading}
                      className="flex items-center gap-2 px-4 py-2 text-sm text-black rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium"
                      style={{ backgroundColor: 'rgb(101, 67, 33)' }}
                      onMouseEnter={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(92, 57, 25)';
                          e.currentTarget.style.transform = 'translateY(-1px)';
                          e.currentTarget.style.boxShadow = '0 4px 12px rgba(101, 67, 33, 0.3)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(101, 67, 33)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
                        }
                      }}
                    >
                      <FolderPlus className="w-4 h-4" />
                      New Folder
                    </button>
                    
                    <button
                      onClick={handleBulkDelete}
                      disabled={bulkOperationLoading}
                      className="flex items-center gap-2 px-4 py-2 text-sm text-black rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium"
                      style={{ backgroundColor: 'rgb(101, 67, 33)' }}
                      onMouseEnter={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(92, 57, 25)';
                          e.currentTarget.style.transform = 'translateY(-1px)';
                          e.currentTarget.style.boxShadow = '0 4px 12px rgba(101, 67, 33, 0.3)';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!bulkOperationLoading) {
                          e.currentTarget.style.backgroundColor = 'rgb(101, 67, 33)';
                          e.currentTarget.style.transform = 'translateY(0)';
                          e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
                        }
                      }}
                    >
                      {bulkOperationLoading ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                      Delete All
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Search Results Container */}
      {hasSearched && (
        <div className="rounded-2xl shadow-sm border overflow-hidden" style={{ backgroundColor: 'rgb(250, 248, 240)', borderColor: 'rgb(210, 180, 140)' }}>
          {isSearching ? (
            <div className="flex items-center justify-center py-16">
              <div className="text-center">
                <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4" style={{ color: 'rgb(139, 115, 85)' }} />
                <p style={{ color: 'rgb(139, 115, 85)' }}>Searching through your files...</p>
              </div>
            </div>
          ) : results.length > 0 ? (
            <div className="divide-y" style={{ borderColor: 'rgb(210, 180, 140)' }}>
              {results.map((result, index) => (
                <div
                  key={`${result.id}-${index}`}
                  className="p-6 transition-colors duration-200"
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgb(245, 240, 230)'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 mt-1">
                      <button
                        onClick={() => toggleFileSelection(result.id)}
                        className="mr-2 transition-all duration-200"
                      >
                        <div className="w-4 h-4 rounded border-2 flex items-center justify-center" style={{
                          backgroundColor: selectedFiles.has(result.id) ? 'rgb(101, 67, 33)' : 'transparent',
                          borderColor: selectedFiles.has(result.id) ? 'rgb(101, 67, 33)' : 'rgb(139, 115, 85)'
                        }}>
                          {selectedFiles.has(result.id) && (
                            <Check className="w-3 h-3 text-white" />
                          )}
                        </div>
                      </button>
                    </div>
                    <div className="flex-shrink-0 mt-1">
                      <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'rgb(245, 240, 230)', color: 'rgb(139, 115, 85)' }}>
                        {getFileIcon(result.mimeType)}
                      </div>
                    </div>
                   
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-lg font-semibold truncate" style={{ color: 'rgb(101, 67, 33)' }}>
                            {result.name}
                          </h3>
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium" style={{ backgroundColor: 'rgb(245, 240, 230)', color: 'rgb(139, 115, 85)' }}>
                            {result.mimeType?.split('/').pop()?.toUpperCase() || 'FILE'}
                          </span>
                        </div>
                     
                      <p className="mb-3 line-clamp-2 leading-relaxed" style={{ color: 'rgb(139, 115, 85)' }}>
                        {result.context}
                      </p>
                     
                      <div className="flex flex-wrap items-center gap-4 text-sm" style={{ color: 'rgb(139, 115, 85)' }}>
                        <div className="flex items-center gap-1">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          {formatDate(result.modifiedTime)}
                        </div>
                        
                        {result.size && (
                          <div className="flex items-center gap-1">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            {formatFileSize(result.size)}
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex-shrink-0">
                      <a
                        href={result.drive_url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center justify-center px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
                        style={{ backgroundColor: 'rgb(139, 115, 85)' }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgb(120, 100, 75)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgb(139, 115, 85)';
                        }}
                      >
                        <ExternalLink className="w-4 h-4 mr-1.5" />
                        Open in Drive
                      </a>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-16">
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-full mb-4" style={{ backgroundColor: 'rgb(245, 240, 230)', color: 'rgb(139, 115, 85)' }}>
                <Search className="w-10 h-10" />
              </div>
              <h3 className="text-lg font-medium mb-2" style={{ color: 'rgb(101, 67, 33)' }}>No results found</h3>
              <p className="max-w-md mx-auto" style={{ color: 'rgb(139, 115, 85)' }}>
                We couldn't find any files matching "{query}". Try different keywords or check your spelling.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Empty State - Before Search */}
      {!hasSearched && !isSearching && (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-full mb-6" style={{ backgroundColor: 'rgb(245, 240, 230)', color: 'rgb(139, 115, 85)' }}>
            <Search className="w-12 h-12" />
          </div>
          <h3 className="text-2xl font-medium mb-2" style={{ color: 'rgb(101, 67, 33)' }}>Search your files</h3>
          <p className="max-w-lg mx-auto mb-8" style={{ color: 'rgb(139, 115, 85)' }}>
            Use natural language to find documents, images, and files across your Google Drive.
          </p>
         
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
            {[
              { icon: <FileText className="w-5 h-5" />, text: 'Search by content, not just filenames' },
              { icon: <Sparkles className="w-5 h-5" />, text: 'AI-powered semantic understanding' },
              { icon: <Search className="w-5 h-5" />, text: 'Natural language search queries' },
            ].map((item, index) => (
              <div key={index} className="flex items-center gap-3 p-4 rounded-xl border shadow-sm transition-shadow" style={{ backgroundColor: 'rgb(250, 248, 240)', borderColor: 'rgb(210, 180, 140)' }} onMouseEnter={(e) => e.currentTarget.style.boxShadow = '0 4px 6px -1px rgba(0, 0, 0, 0.1)'} onMouseLeave={(e) => e.currentTarget.style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1)'}>
                <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: 'rgb(245, 240, 230)', color: 'rgb(139, 115, 85)' }}>
                  {item.icon}
                </div>
                <span className="text-sm font-medium" style={{ color: 'rgb(101, 67, 33)' }}>{item.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Existing Folder Modal */}
      {showFolderModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="rounded-lg p-6 w-full max-w-md mx-4" style={{ backgroundColor: 'rgb(250, 248, 240)' }}>
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'rgb(101, 67, 33)' }}>Move to Existing Folder</h3>
            <p className="text-sm mb-4" style={{ color: 'rgb(139, 115, 85)' }}>
              Select a folder to move {selectedFiles.size} selected file{selectedFiles.size !== 1 ? 's' : ''} to:
            </p>
            
            <div className="max-h-64 overflow-y-auto border rounded-md" style={{ borderColor: 'rgb(210, 180, 140)' }}>
              {existingFolders.length > 0 ? (
                existingFolders.map((folder) => (
                  <button
                    key={folder.id}
                    onClick={() => handleMoveToExistingFolder(folder.id, folder.name)}
                    disabled={bulkOperationLoading}
                    className="w-full text-left px-4 py-3 border-b last:border-b-0 disabled:opacity-50 transition-colors"
                    style={{ borderColor: 'rgb(210, 180, 140)', color: 'rgb(101, 67, 33)' }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'rgb(245, 240, 230)'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
                  >
                    <div className="flex items-center gap-3">
                      <FolderPlus className="w-5 h-5" style={{ color: 'rgb(139, 115, 85)' }} />
                      <span className="font-medium">{folder.name}</span>
                    </div>
                  </button>
                ))
              ) : (
                <div className="p-4 text-center" style={{ color: 'rgb(139, 115, 85)' }}>
                  No folders found. Create a new folder instead.
                </div>
              )}
            </div>
            
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowFolderModal(false)}
                disabled={bulkOperationLoading}
                className="px-4 py-2 text-white rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium"
                style={{ backgroundColor: 'rgb(101, 67, 33)' }}
                onMouseEnter={(e) => {
                  if (!bulkOperationLoading) {
                    e.currentTarget.style.backgroundColor = 'rgb(92, 57, 25)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!bulkOperationLoading) {
                    e.currentTarget.style.backgroundColor = 'rgb(101, 67, 33)';
                  }
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* New Folder Modal */}
      {showNewFolderModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="rounded-lg p-6 w-full max-w-md mx-4" style={{ backgroundColor: 'rgb(250, 248, 240)' }}>
            <h3 className="text-lg font-semibold mb-4" style={{ color: 'rgb(101, 67, 33)' }}>Create New Folder</h3>
            <p className="text-sm mb-4" style={{ color: 'rgb(139, 115, 85)' }}>
              Enter a name for the new folder to move {selectedFiles.size} selected file{selectedFiles.size !== 1 ? 's' : ''} to:
            </p>
            
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              placeholder="Enter folder name"
              className="w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:border-transparent"
              style={{ 
                borderColor: 'rgb(210, 180, 140)', 
                backgroundColor: 'rgb(255, 253, 248)',
                color: 'rgb(101, 67, 33)'
              }}
              onFocus={(e) => {
                e.currentTarget.style.borderColor = 'rgb(139, 115, 85)';
                e.currentTarget.style.boxShadow = '0 0 0 2px rgba(139, 115, 85, 0.2)';
              }}
              onBlur={(e) => {
                e.currentTarget.style.borderColor = 'rgb(210, 180, 140)';
                e.currentTarget.style.boxShadow = 'none';
              }}
              disabled={bulkOperationLoading}
            />
            
            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => {
                  setShowNewFolderModal(false);
                  setNewFolderName('');
                }}
                disabled={bulkOperationLoading}
                className="px-4 py-2 text-white rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium"
                style={{ backgroundColor: 'rgb(120, 113, 108)' }}
                onMouseEnter={(e) => {
                  if (!bulkOperationLoading) {
                    e.currentTarget.style.backgroundColor = 'rgb(107, 100, 95)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!bulkOperationLoading) {
                    e.currentTarget.style.backgroundColor = 'rgb(120, 113, 108)';
                  }
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleCreateNewFolder}
                disabled={bulkOperationLoading || !newFolderName.trim()}
                className="px-4 py-2 text-white rounded-lg shadow-sm disabled:opacity-50 transition-all duration-200 font-medium flex items-center gap-2"
                style={{ backgroundColor: 'rgb(101, 67, 33)' }}
                onMouseEnter={(e) => {
                  if (!bulkOperationLoading && newFolderName.trim()) {
                    e.currentTarget.style.backgroundColor = 'rgb(92, 57, 25)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!bulkOperationLoading && newFolderName.trim()) {
                    e.currentTarget.style.backgroundColor = 'rgb(101, 67, 33)';
                  }
                }}
              >
                {bulkOperationLoading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <FolderPlus className="w-4 h-4" />
                )}
                Create & Move
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
    </div>
  );
};

export default SearchFiles;
