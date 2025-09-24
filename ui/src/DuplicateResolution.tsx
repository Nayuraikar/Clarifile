import React from 'react';

interface File {
  id: string;
  name: string;
  size?: string;
  label?: string;
}

interface Group {
  group_id?: number;
  file_count: number;
  files: File[];
}

interface DuplicateResolutionProps {
  group: Group;
  groupIndex: number;
  duplicateResolution: { [key: string]: boolean };
  setDuplicateResolution: (resolution: { [key: string]: boolean }) => void;
  duplicateResolutionLoading: boolean;
  setDuplicateResolutionLoading: (loading: boolean) => void;
  call: (path: string, opts?: RequestInit) => Promise<any>;
  setNotification: (message: string) => void;
  refreshDups: () => Promise<void>;
}

const DuplicateResolution: React.FC<DuplicateResolutionProps> = ({
  group,
  groupIndex,
  duplicateResolution,
  setDuplicateResolution,
  duplicateResolutionLoading,
  setDuplicateResolutionLoading,
  call,
  setNotification,
  refreshDups
}) => {
  const handleResolveDuplicates = async () => {
    // Find which file is selected to keep
    const selectedFileId = Object.keys(duplicateResolution).find(id => 
      duplicateResolution[id] === true && group.files.some((f: File) => f.id === id)
    );
    
    if (!selectedFileId) {
      alert('Please select which file to keep');
      return;
    }
    
    const selectedFile = group.files.find((f: File) => f.id === selectedFileId);
    const filesToDelete = group.files.filter((f: File) => f.id !== selectedFileId);
    
    if (confirm(`Are you sure you want to keep "${selectedFile?.name}" and delete ${filesToDelete.length} duplicate(s)?`)) {
      setDuplicateResolutionLoading(true);
      try {
        // For each file to delete, call the backend to resolve the duplicate
        for (const fileToDelete of filesToDelete) {
          await call('/resolve_duplicate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              file_a: selectedFileId,
              file_b: fileToDelete.id,
              action: 'keep_a'
            })
          });
        }
        
        // Show success message
        setNotification(`Successfully resolved duplicates. Kept "${selectedFile?.name}" and deleted ${filesToDelete.length} duplicate(s).`);
        
        // Clear the selection
        setDuplicateResolution(prev => {
          const newResolution = { ...prev };
          group.files.forEach((f: File) => {
            delete newResolution[f.id];
          });
          return newResolution;
        });
        
        // Refresh the duplicates list
        await refreshDups();
        
      } catch (error) {
        console.error('Error resolving duplicates:', error);
        setNotification('Error resolving duplicates. Please try again.');
      } finally {
        setDuplicateResolutionLoading(false);
      }
    }
  };

  const handleClearSelection = () => {
    // Clear selection for this group
    setDuplicateResolution(prev => {
      const newResolution = { ...prev };
      group.files.forEach((f: File) => {
        delete newResolution[f.id];
      });
      return newResolution;
    });
  };

  const handleFileSelection = (fileId: string) => {
    setDuplicateResolution(prev => {
      const newResolution = { ...prev };
      // Clear all other selections in this group
      group.files.forEach((f: File) => {
        if (f.id !== fileId) {
          delete newResolution[f.id];
        }
      });
      // Set this file as the one to keep
      newResolution[fileId] = true;
      return newResolution;
    });
  };

  return (
    <div className="professional-card professional-interactive fade-in" style={{animationDelay: `${groupIndex * 0.1}s`}}>
      <div className="card-heading mb-4">Duplicate Group {groupIndex + 1}</div>
      <div className="text-sm text-text-muted mb-4">
        {group.file_count} files with identical content
      </div>
      
      <div className="space-y-3">
        {group.files.map((file: File, fileIndex: number) => (
          <div key={file.id} className="flex items-center justify-between p-4 bg-bg-secondary rounded-xl">
            <div className="flex items-center gap-3">
              <span className="text-2xl">📄</span>
              <div>
                <div className="font-medium text-text-primary">{file.name}</div>
                <div className="text-sm text-text-muted">
                  {file.size ? `${file.size} bytes` : 'Size unknown'}
                  {file.label && ` • Category: ${file.label}`}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="radio"
                name={`duplicate-group-${groupIndex}`}
                id={`keep-${file.id}`}
                checked={duplicateResolution[file.id] === true}
                onChange={() => handleFileSelection(file.id)}
                className="w-4 h-4 text-accent-primary bg-bg-secondary border-text-muted rounded focus:ring-accent-primary"
              />
              <label htmlFor={`keep-${file.id}`} className="text-sm font-medium text-text-primary cursor-pointer">
                Keep this one
              </label>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 flex justify-end gap-3">
        <button
          onClick={handleClearSelection}
          className="px-4 py-2 text-sm font-medium text-text-primary bg-bg-secondary border border-text-muted rounded-lg hover:bg-bg-tertiary focus:outline-none focus:ring-2 focus:ring-accent-primary"
        >
          Clear Selection
        </button>
        <button
          onClick={handleResolveDuplicates}
          disabled={duplicateResolutionLoading || !Object.keys(duplicateResolution).some(id => 
            duplicateResolution[id] === true && group.files.some((f: File) => f.id === id)
          )}
          className="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-accent-primary to-accent-secondary rounded-lg hover:opacity-90 focus:outline-none focus:ring-2 focus:ring-accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {duplicateResolutionLoading ? 'Resolving...' : 'Resolve Duplicates'}
        </button>
      </div>
    </div>
  );
};

export default DuplicateResolution;
