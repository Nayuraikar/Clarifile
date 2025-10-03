import React, { useState, useEffect, useRef } from 'react';
import { assistantGenerate, downloadFromResponse } from '../assistantApi';
import { Search, X, FileText, Image as ImageIcon, Music, Film, FileSpreadsheet, FilePresentation, Code, Archive, Mail, BookOpen, Loader2, Sparkles } from 'lucide-react';

interface SearchResult {
  id: string;
  name: string;
  mimeType: string;
  size?: number;
  modifiedTime?: string;
  score: number;
  match_type: string;
  context: string;
  drive_url: string;
}

interface SearchResponse {
  query: string;
  file_type?: string;
  total_matches: number;
  results: SearchResult[];
  search_terms: string[];
}

const SearchComponent: React.FC = () => {
  const [query, setQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchStats, setSearchStats] = useState<{ total: number; shown: number } | null>(null);
  const [selectedFileType, setSelectedFileType] = useState<string | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [isFocused, setIsFocused] = useState(false);
  const [isVisualSearch, setIsVisualSearch] = useState(false);
  const [visualSearchPreview, setVisualSearchPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Assistant Generator state
  const [assistantKind, setAssistantKind] = useState<'flowchart' | 'short_notes' | 'detailed_notes' | 'timeline' | 'key_insights' | 'flashcards'>('flowchart');
  const [assistantOutput, setAssistantOutput] = useState<any>(null);
  const [assistantError, setAssistantError] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);

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

  // Handle search
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() && !isVisualSearch) return;
    
    setIsSearching(true);
    setShowSuggestions(false);
    
    try {
      let response: Response;
      
      if (isVisualSearch && visualSearchPreview) {
        // Handle visual search
        const formData = new FormData();
        const blob = await fetch(visualSearchPreview).then(r => r.blob());
        formData.append('image', blob, 'search-image.jpg');
        
        response = await fetch('http://localhost:4000/visual_search', {
          method: 'POST',
          body: formData,
        });
      } else {
        // Handle text search
        response = await fetch('http://localhost:4000/search_files', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            query,
            use_semantic: true,
            semantic_weight: 0.7,
            min_score: 0.2,
            top_k: 50,
          }),
        });
      }
      
      if (!response.ok) {
        throw new Error(`Search failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      // Transform the new API response to match the expected format
      const transformedResults = (data.files || []).map((file: any) => ({
        id: file.id,
        name: file.name,
        mimeType: file.mimeType,
        size: file.size,
        modifiedTime: file.modifiedTime,
        score: file.match_score || file.confidence / 100 || 0,
        match_type: file.match_type || 'semantic',
        context: file.context || '',
        drive_url: file.drive_url || `https://drive.google.com/file/d/${file.id}/view`
      }));
      
      setResults(transformedResults);
      setSearchStats({
        total: data.total_searched || 0,
        shown: data.matches_found || 0,
      });
      
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
      setSearchStats(null);
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
      
      // Auto-search after delay (debounce)
      searchTimeout.current = setTimeout(() => {
        handleSearch(e);
      }, 500);
    } else {
      setShowSuggestions(false);
      setSuggestions([]);
    }
  };

  // Handle suggestion click
  const handleSuggestionClick = (suggestion: string) => {
    setQuery(suggestion);
    setShowSuggestions(false);
    handleSearch({ preventDefault: () => {} } as React.FormEvent);
  };

  // Clear search
  const clearSearch = () => {
    setQuery('');
    setResults([]);
    setSearchStats(null);
    setVisualSearchPreview(null);
    setIsVisualSearch(false);
    setShowSuggestions(false);
    if (inputRef.current) {
      inputRef.current.focus();
    }
  };

  // Assistant generate handler (text-only quick path)
  const handleAssistantGenerate = async (fmt: 'md' | 'txt' | 'docx' | 'pdf' = 'md') => {
    const text = query?.trim();
    if (!text) {
      setAssistantError('Enter some text first (e.g., paste content or a brief).');
      return;
    }
    setAssistantError('');
    setIsGenerating(true);
    try {
      const resp = await assistantGenerate({ kind: assistantKind, text, format: fmt });
      setAssistantOutput(resp);
    } catch (e: any) {
      setAssistantError(e?.message || 'Generation failed');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleAssistantDownload = () => {
    if (!assistantOutput) return;
    downloadFromResponse(assistantOutput);
  };

  // Handle file upload for visual search
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    // Check if file is an image
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file');
      return;
    }
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file);
    setVisualSearchPreview(previewUrl);
    setIsVisualSearch(true);
    setQuery('Searching similar images...');
    
    // Trigger search
    setTimeout(() => {
      handleSearch({ preventDefault: () => {} } as React.FormEvent);
    }, 100);
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

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Search Header */}
      <div className="text-center mb-8">
        <div className="relative inline-block mb-2">
          <h1 className="text-4xl font-bold text-[#8b7355] mb-2">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-[#8b7355] to-[#c19a6b] animate-text-shimmer">
              Clarifile Search
            </span>
          </h1>
          <div className="absolute -top-2 -right-2 bg-gradient-to-r from-amber-200 to-amber-400 text-amber-900 text-xs font-bold px-2 py-1 rounded-full shadow-md transform rotate-6">
            AI-Powered
          </div>
        </div>
        <p className="text-[#8b7355] text-lg">Intelligent document search powered by AI</p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div 
          className={`relative flex items-center bg-white rounded-2xl shadow-lg border-2 transition-all duration-300 ${
            isFocused ? 'border-[#8b7355]' : 'border-[#e6dfd4]'
          }`}
        >
          <div className="absolute left-4 text-[#8b7355]">
            {isVisualSearch ? (
              <ImageIcon className="w-5 h-5" />
            ) : isSearching ? (
              <Loader2 className="w-5 h-5 animate-spin" />
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
            placeholder={isVisualSearch ? "Searching similar images..." : "Search documents, PDFs, presentations, images, audio, and more..."}
            className={`w-full pl-12 pr-16 py-4 text-lg text-[#2d2416] bg-transparent border-none focus:ring-0 focus:outline-none placeholder-[#8b7355] ${
              isVisualSearch ? 'opacity-70' : ''
            }`}
            disabled={isSearching || isVisualSearch}
          />
          
          {visualSearchPreview && (
            <div className="absolute right-16 w-8 h-8 rounded-full overflow-hidden border-2 border-white shadow-md">
              <img 
                src={visualSearchPreview} 
                alt="Search preview" 
                className="w-full h-full object-cover"
              />
            </div>
          )}
          
          {(query || visualSearchPreview) && !isSearching && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute right-14 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Clear search"
            >
              <X className="w-5 h-5" />
            </button>
          )}
          
          <div className="absolute right-2 flex space-x-1">
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={handleFileUpload}
              className="hidden"
              id="visual-search-input"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className={`p-2 rounded-full transition-all ${
                isVisualSearch 
                  ? 'bg-[#8b7355] text-white' 
                  : 'text-[#8b7355] hover:bg-gray-100'
              }`}
              title="Search by image"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
            
            <button
              type="submit"
              disabled={isSearching || (!query && !visualSearchPreview)}
              className={`p-2 rounded-full transition-all ${
                isSearching || (!query && !visualSearchPreview)
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-[#8b7355] text-white hover:bg-[#6b5a45] shadow-md'
              }`}
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
          <div className="mt-1 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden">
            <div className="py-1">
              <div className="px-4 py-2 text-xs font-semibold text-gray-500 bg-gray-50">
                Try searching for:
              </div>
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => handleSuggestionClick(suggestion)}
                  className="w-full text-left px-4 py-2 hover:bg-gray-50 flex items-center text-gray-700"
                >
                  <Search className="w-4 h-4 mr-2 text-gray-400" />
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}
      </form>

      {/* File Type Filters */}
      <div className="flex flex-wrap gap-2 mb-6 justify-center">
        {fileTypes.map((type) => (
          <button
            key={type.id}
            type="button"
            onClick={() => {
              setSelectedFileType(type.id === selectedFileType ? null : type.id);
              if (query) {
                handleSearch({ preventDefault: () => {} } as React.FormEvent);
              }
            }}
            className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium transition-all ${
              selectedFileType === type.id
                ? 'bg-[#8b7355] text-white shadow-md'
                : 'bg-white text-[#8b7355] border border-[#e6dfd4] hover:border-[#8b7355] hover:bg-gray-50'
            }`}
          >
            {React.cloneElement(type.icon, {
              className: `w-4 h-4 ${selectedFileType === type.id ? 'text-white' : 'text-[#8b7355]'}`
            })}
            {type.name}
          </button>
        ))}
      </div>

      {/* Search Stats */}
      {searchStats && (
        <div className="mb-6 text-center">
          <div className="inline-flex items-center gap-6 bg-white/80 backdrop-blur-sm px-6 py-3 rounded-full shadow-sm border border-[#e6dfd4]">
            <div className="text-center">
              <div className="text-2xl font-bold text-[#8b7355]">{searchStats.total.toLocaleString()}</div>
              <div className="text-xs text-gray-500">Total Matches</div>
            </div>
            <div className="h-8 w-px bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-[#8b7355]">{searchStats.shown.toLocaleString()}</div>
              <div className="text-xs text-gray-500">Showing</div>
            </div>
            <div className="h-8 w-px bg-gray-200"></div>
            <div className="text-center">
              <div className="text-2xl font-bold text-[#8b7355]">
                {searchStats.total > 0 ? Math.min(100, Math.round((searchStats.shown / searchStats.total) * 100)) : 0}%
              </div>
              <div className="text-xs text-gray-500">of Results</div>
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      <div className="space-y-4">
        {results.map((result, index) => (
          <div 
            key={`${result.id}-${index}`}
            className="group relative bg-white rounded-xl shadow-sm border border-[#e6dfd4] hover:shadow-md transition-all duration-200 overflow-hidden hover:-translate-y-0.5"
          >
            <div className="p-5">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 mt-1">
                  <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-700">
                    {result.mimeType?.includes('pdf') ? (
                      <FileText className="w-5 h-5" />
                    ) : result.mimeType?.includes('spreadsheet') ? (
                      <FileSpreadsheet className="w-5 h-5" />
                    ) : result.mimeType?.includes('presentation') ? (
                      <FilePresentation className="w-5 h-5" />
                    ) : result.mimeType?.includes('image') ? (
                      <ImageIcon className="w-5 h-5" />
                    ) : result.mimeType?.includes('video') ? (
                      <Film className="w-5 h-5" />
                    ) : result.mimeType?.includes('audio') ? (
                      <Music className="w-5 h-5" />
                    ) : (
                      <FileText className="w-5 h-5" />
                    )}
                  </div>
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <h3 className="text-lg font-medium text-gray-900 truncate">
                      {result.name}
                    </h3>
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-100 text-amber-800">
                      {result.mimeType?.split('/').pop()?.toUpperCase() || 'FILE'}
                    </span>
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-2 line-clamp-2">
                    {result.context}
                  </p>
                  
                  <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500">
                    <div className="flex items-center gap-1">
                      <span className="text-gray-400">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </span>
                      {formatDate(result.modifiedTime)}
                    </div>
                    
                    {result.size && (
                      <div className="flex items-center gap-1">
                        <span className="text-gray-400">
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                        </span>
                        {formatFileSize(result.size)}
                      </div>
                    )}
                    
                    <div className="flex items-center gap-1">
                      <span className="text-gray-400">
                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 15.536c-1.171 1.952-3.07 1.952-4.242 0-1.172-1.953-1.172-5.119 0-7.072 1.171-1.952 3.07-1.952 4.242 0M8 10.5h4m-4 3h4m9-1.5a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                      </span>
                      {Math.round(result.score * 100)}% Match
                    </div>
                  </div>
                </div>
                
                <div className="flex-shrink-0 flex flex-col gap-2">
                  <a
                    href={result.drive_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-[#8b7355] hover:bg-[#6b5a45] shadow-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#8b7355] transition-colors"
                  >
                    Open in Drive
                  </a>
                  
                  <button
                    type="button"
                    className="inline-flex items-center justify-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-[#8b7355] bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#8b7355] transition-colors"
                  >
                    <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    Chat
                  </button>
                </div>
              </div>
            </div>
            
            {/* Match indicator */}
            <div 
              className="h-1 bg-gradient-to-r from-green-400 to-blue-500"
              style={{ width: `${Math.round(result.score * 100)}%` }}
            ></div>
          </div>
        ))}
        
        {/* Empty States */}
        {!isSearching && results.length === 0 && query && (
          <div className="text-center py-12">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-amber-50 text-amber-500 mb-4">
              <Search className="w-10 h-10" />
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-1">No results found</h3>
            <p className="text-gray-500 max-w-md mx-auto">
              We couldn't find any files matching "{query}". Try different keywords or check your spelling.
            </p>
          </div>
        )}
        
        {!isSearching && !query && results.length === 0 && (
          <div className="text-center py-16">
            <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-amber-50 to-amber-100 text-amber-500 mb-6 animate-float">
              <Search className="w-12 h-12" />
            </div>
            <h3 className="text-2xl font-medium text-gray-900 mb-2">Search your files</h3>
            <p className="text-gray-500 max-w-lg mx-auto mb-8">
              Use natural language to find documents, images, audio files, and more across your Google Drive.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
              {[
                { icon: <FileText className="w-5 h-5" />, text: 'Search by content, not just filenames' },
                { icon: <ImageIcon className="w-5 h-5" />, text: 'Search images with OCR and audio with transcription' },
                { icon: <Sparkles className="w-5 h-5" />, text: 'AI-powered semantic understanding' },
                { icon: <Code className="w-5 h-5" />, text: 'Filter by file type, date, and more' },
              ].map((item, index) => (
                <div key={index} className="flex items-center gap-3 p-3 bg-white rounded-lg border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-amber-50 flex items-center justify-center text-amber-600">
                    {item.icon}
                  </div>
                  <span className="text-sm text-gray-700">{item.text}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Assistant Generator */}
      <div className="mt-10 bg-white rounded-xl shadow-sm border border-[#e6dfd4] p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-[#8b7355]">Assistant Generator</h2>
          <div className="flex gap-2 items-center">
            <select
              value={assistantKind}
              onChange={(e) => setAssistantKind(e.target.value as any)}
              className="border rounded-md px-2 py-1 text-sm"
            >
              <option value="flowchart">Flowchart (Mermaid)</option>
              <option value="short_notes">Short Notes</option>
              <option value="detailed_notes">Detailed Notes</option>
              <option value="timeline">Timeline</option>
              <option value="key_insights">Key Insights</option>
              <option value="flashcards">Q&A Flashcards</option>
            </select>
            <button
              type="button"
              disabled={isGenerating}
              onClick={() => handleAssistantGenerate('md')}
              className={`px-3 py-2 rounded-md text-white ${isGenerating ? 'bg-gray-300' : 'bg-[#8b7355] hover:bg-[#6b5a45]'}`}
            >
              {isGenerating ? 'Generating...' : 'Generate'}
            </button>
            {assistantOutput && (
              <button
                type="button"
                onClick={handleAssistantDownload}
                className="px-3 py-2 rounded-md border text-[#8b7355] hover:bg-gray-50"
              >
                Download
              </button>
            )}
          </div>
        </div>
        <p className="text-xs text-gray-500 mb-3">Tip: Paste or type content into the main search input above to generate outputs without uploading a file.</p>
        {assistantError && <div className="text-sm text-red-600 mb-2">{assistantError}</div>}
        {assistantOutput && assistantOutput.kind === 'flowchart' && (
          <div className="bg-gray-50 rounded-md p-3 overflow-auto">
            <pre className="text-xs whitespace-pre-wrap">```mermaid
{assistantOutput.content}
```</pre>
          </div>
        )}
        {assistantOutput && assistantOutput.kind !== 'flowchart' && (
          <div className="bg-gray-50 rounded-md p-3 overflow-auto">
            <pre className="text-xs whitespace-pre-wrap">{typeof assistantOutput.content === 'string' ? assistantOutput.content : JSON.stringify(assistantOutput.content, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default SearchComponent;
