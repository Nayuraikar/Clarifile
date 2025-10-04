import React, { useState, useEffect, useRef } from 'react';
import { Search, X, FileText, Image as ImageIcon, Music, Film, FileSpreadsheet, Presentation as FilePresentation, Code, Archive, Mail, BookOpen, Loader2, Sparkles, Eye, MessageCircle, Download, ExternalLink } from 'lucide-react';

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
  const [isVisualSearch, setIsVisualSearch] = useState(false);
  const [visualSearchPreview, setVisualSearchPreview] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleDownload = async (fileId: string, fileName: string) => {
    try {
      const authToken = localStorage.getItem('drive_token') || '';
      if (!authToken) {
        const urlParams = new URLSearchParams(window.location.search);
        const urlToken = urlParams.get('auth_token');
        if (urlToken) {
          localStorage.setItem('drive_token', urlToken);
        }
      }
      
      const token = localStorage.getItem('drive_token') || '';
      const response = await fetch(`http://localhost:4000/download/${fileId}?auth_token=${encodeURIComponent(token)}`, {
        method: 'GET',
      });

      if (!response.ok) {
        throw new Error(`Failed to download file: ${response.statusText}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
    } catch (error) {
      console.error('Download error:', error);
      alert('Failed to download file. Please try again.');
    }
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
    if (!query.trim() && !isVisualSearch) return;
   
    setIsSearching(true);
    setShowSuggestions(false);
    setHasSearched(true);
   
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
      }
     
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
    setVisualSearchPreview(null);
    setIsVisualSearch(false);
    setShowSuggestions(false);
    setHasSearched(false);
    if (inputRef.current) {
      inputRef.current.focus();
    }
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
      handleSearch();
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
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Search Header */}
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold mb-2" style={{ color: 'rgb(139, 115, 85)' }}>Find Your Files</h1>
        <p className="text-gray-600 text-lg">Search across your documents, PDFs, and presentations with AI-powered precision</p>
      </div>

      {/* Search Bar */}
      <form onSubmit={handleSearch} className="mb-8">
        <div
          className={`relative flex items-center bg-white rounded-2xl shadow-lg border-2 transition-all duration-300 ${
            isFocused ? 'border-gray-400 shadow-gray-100' : 'border-gray-200'
          }`}
        >
          <div className="absolute left-4 text-gray-400">
            {isVisualSearch ? (
              <ImageIcon className="w-5 h-5" />
            ) : isSearching ? (
              <Loader2 className="w-5 h-5 animate-spin text-gray-500" />
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
            placeholder={isVisualSearch ? "Searching similar images..." : "Search documents, PDFs, presentations, and more..."}
            className={`w-full pl-12 pr-32 py-4 text-lg text-gray-800 bg-transparent border-none focus:ring-0 focus:outline-none placeholder-gray-400 ${
              isVisualSearch ? 'opacity-70' : ''
            }`}
            disabled={isSearching || isVisualSearch}
          />
         
          {visualSearchPreview && (
            <div className="absolute right-20 w-8 h-8 rounded-full overflow-hidden border-2 border-white shadow-md">
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
              className="absolute right-16 p-1 text-gray-400 hover:text-gray-600 transition-colors"
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
                  ? 'text-white'
                  : 'text-gray-500 hover:bg-gray-100'
              }`}
              style={isVisualSearch ? { backgroundColor: 'rgb(139, 115, 85)' } : {}}
              title="Visual search"
            >
              <ImageIcon className="w-5 h-5" />
            </button>
           
            <button
              type="submit"
              disabled={isSearching || (!query && !visualSearchPreview)}
              className={`p-2 rounded-full transition-all ${
                isSearching || (!query && !visualSearchPreview)
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'text-white shadow-md'
              }`}
              style={!(isSearching || (!query && !visualSearchPreview)) ? { backgroundColor: 'rgb(139, 115, 85)' } : {}}
              onMouseEnter={(e) => {
                if (!(isSearching || (!query && !visualSearchPreview))) {
                  e.currentTarget.style.backgroundColor = 'rgb(120, 100, 75)';
                }
              }}
              onMouseLeave={(e) => {
                if (!(isSearching || (!query && !visualSearchPreview))) {
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
          <div className="mt-2 bg-white rounded-lg shadow-lg border border-gray-200 overflow-hidden z-50 relative">
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


      {/* Search Stats */}
      {searchStats && hasSearched && (
        <div className="mb-8">
          <div className="flex justify-center">
            <div className="inline-flex items-center gap-8 bg-white px-8 py-4 rounded-2xl shadow-sm border border-gray-200">
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gray-100 rounded-full mb-2">
                  <Search className="w-6 h-6 text-gray-600" />
                </div>
                <div className="text-2xl font-bold text-gray-800">{searchStats.total}</div>
                <div className="text-sm text-gray-500">Files Searched</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gray-100 rounded-full mb-2">
                  <Eye className="w-6 h-6 text-gray-600" />
                </div>
                <div className="text-2xl font-bold text-gray-800">{searchStats.shown}</div>
                <div className="text-sm text-gray-500">Matches Found</div>
              </div>
              
              <div className="text-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gray-100 rounded-full mb-2">
                  <Sparkles className="w-6 h-6 text-gray-600" />
                </div>
                <div className="text-2xl font-bold text-gray-800">{searchStats.matchRate}%</div>
                <div className="text-sm text-gray-500">Match Rate</div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Search Results Header */}
      {hasSearched && (
        <div className="mb-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-800">
              Search Results
              {query && <span className="text-gray-500 font-normal"> for "{query}"</span>}
            </h2>
            <div className="text-sm text-gray-500">
              Showing {results.length} results
            </div>
          </div>
        </div>
      )}

      {/* Search Results Container */}
      {hasSearched && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
          {isSearching ? (
            <div className="flex items-center justify-center py-16">
              <div className="text-center">
                <Loader2 className="w-12 h-12 animate-spin text-gray-500 mx-auto mb-4" />
                <p className="text-gray-600">Searching through your files...</p>
              </div>
            </div>
          ) : results.length > 0 ? (
            <div className="divide-y divide-gray-100">
              {results.map((result, index) => (
                <div
                  key={`${result.id}-${index}`}
                  className="p-6 hover:bg-gray-50 transition-colors duration-200"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 mt-1">
                      <div className="w-12 h-12 rounded-xl bg-gray-100 flex items-center justify-center text-gray-600">
                        {getFileIcon(result.mimeType)}
                      </div>
                    </div>
                   
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-2">
                          <h3 className="text-lg font-semibold text-gray-900 truncate">
                            {result.name}
                          </h3>
                          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-700">
                            {result.mimeType?.split('/').pop()?.toUpperCase() || 'FILE'}
                          </span>
                        </div>
                     
                      <p className="text-gray-600 mb-3 line-clamp-2 leading-relaxed">
                        {result.context}
                      </p>
                     
                      <div className="flex flex-wrap items-center gap-4 text-sm text-gray-500">
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
                    
                    <div className="flex-shrink-0 flex gap-2">
                      <button
                        onClick={() => handleDownload(result.id, result.name)}
                        className="inline-flex items-center justify-center px-4 py-2 text-white text-sm font-medium rounded-lg transition-colors shadow-sm"
                        style={{ backgroundColor: 'rgb(100, 116, 139)' }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgb(80, 96, 119)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgb(100, 116, 139)';
                        }}
                      >
                        <Download className="w-4 h-4 mr-1.5" />
                        Download
                      </button>
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
              <div className="inline-flex items-center justify-center w-20 h-20 rounded-full bg-gray-100 text-gray-400 mb-4">
                <Search className="w-10 h-10" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
              <p className="text-gray-500 max-w-md mx-auto">
                We couldn't find any files matching "{query}". Try different keywords or check your spelling.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Empty State - Before Search */}
      {!hasSearched && !isSearching && (
        <div className="text-center py-16">
          <div className="inline-flex items-center justify-center w-24 h-24 rounded-full bg-gradient-to-br from-gray-100 to-gray-200 text-gray-600 mb-6">
            <Search className="w-12 h-12" />
          </div>
          <h3 className="text-2xl font-medium text-gray-900 mb-2">Search your files</h3>
          <p className="text-gray-500 max-w-lg mx-auto mb-8">
            Use natural language to find documents, images, and files across your Google Drive.
          </p>
         
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl mx-auto">
            {[
              { icon: <FileText className="w-5 h-5" />, text: 'Search by content, not just filenames' },
              { icon: <ImageIcon className="w-5 h-5" />, text: 'Find similar images with visual search' },
              { icon: <Sparkles className="w-5 h-5" />, text: 'AI-powered semantic understanding' },
              { icon: <Search className="w-5 h-5" />, text: 'Natural language search queries' },
            ].map((item, index) => (
              <div key={index} className="flex items-center gap-3 p-4 bg-white rounded-xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
                <div className="flex-shrink-0 w-10 h-10 rounded-full bg-gray-100 flex items-center justify-center text-gray-600">
                  {item.icon}
                </div>
                <span className="text-sm text-gray-700 font-medium">{item.text}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SearchFiles;
