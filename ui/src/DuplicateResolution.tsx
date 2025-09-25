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
  const handleKeep = async (file: File) => {
    try {
      setDuplicateResolutionLoading(true);
      await call('/drive/keep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: file.id })
      });
      setNotification(`Kept "${file.name}"`);
      await refreshDups();
    } catch (e) {
      console.error('Error keeping file:', e);
      setNotification('Error keeping file. Please try again.');
    } finally {
      setDuplicateResolutionLoading(false);
    }
  };

  const handleDelete = async (file: File) => {
    if (!confirm(`Delete "${file.name}" from Drive? This action cannot be undone.`)) return;
    try {
      setDuplicateResolutionLoading(true);
      const resp = await call('/drive/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: file.id })
      });
      if (resp?.error) throw new Error(String(resp.error));
      setNotification(`Deleted "${file.name}"`);
      await refreshDups();
    } catch (e) {
      console.error('Error deleting file:', e);
      setNotification('Error deleting file. Please try again.');
    } finally {
      setDuplicateResolutionLoading(false);
    }
  };

  const handleClearSelection = () => {
    // No-op in Drive mode; preserved for compatibility
    setDuplicateResolution({});
  };

  const handleFileSelection = (_fileId: string) => {
    // No-op in Drive mode
  };

  return (
    <div className="duplicate-group-card fade-in" style={{animationDelay: `${groupIndex * 0.1}s`}}>
      <div className="duplicate-group-header">
        <div className="duplicate-group-title">🔄 Duplicate Group {groupIndex + 1}</div>
        <div className="duplicate-group-count">
          {group.file_count}
        </div>
      </div>
      <div className="duplicate-group-subtitle">
        
      </div>
      
      <div className="duplicate-files-list">
        {group.files.map((file: File, fileIndex) => (
          <div key={file.id} className="duplicate-file-item" style={{animationDelay: `${(groupIndex * 0.1) + (fileIndex * 0.05)}s`}}>
            <div className="duplicate-file-info">
              <div className="duplicate-file-icon">
                📄
              </div>
              <div className="duplicate-file-details">
                <div className="duplicate-file-name">{file.name}</div>
                <div className="duplicate-file-size">
                  {file.size ? `${file.size} bytes` : 'Size unknown'}
                </div>
              </div>
            </div>
            <div className="duplicate-actions">
              <button
                onClick={() => handleKeep(file)}
                disabled={duplicateResolutionLoading}
                className="duplicate-action-button keep"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
                Keep
              </button>
              <button
                onClick={() => handleDelete(file)}
                disabled={duplicateResolutionLoading}
                className="duplicate-action-button delete"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                </svg>
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DuplicateResolution;
