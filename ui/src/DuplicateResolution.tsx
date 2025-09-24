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
    <div className="professional-card professional-interactive fade-in" style={{animationDelay: `${groupIndex * 0.1}s`}}>
      <div className="card-heading mb-4">Duplicate Group {groupIndex + 1}</div>
      <div className="text-sm text-text-muted mb-4">
        {group.file_count} files with identical content
      </div>
      
      <div className="space-y-3">
        {group.files.map((file: File) => (
          <div key={file.id} className="flex items-center justify-between p-4 bg-bg-secondary rounded-xl">
            <div className="flex items-center gap-3">
              <div>
                <div className="font-medium text-text-primary">{file.name}</div>
                <div className="text-sm text-text-muted">
                  {file.size ? `${file.size} bytes` : 'Size unknown'}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => handleKeep(file)}
                disabled={duplicateResolutionLoading}
                className="px-3 py-2 text-sm font-medium text-text-primary bg-bg-secondary border border-text-muted rounded-lg hover:bg-bg-tertiary disabled:opacity-50"
              >
                Keep
              </button>
              <button
                onClick={() => handleDelete(file)}
                disabled={duplicateResolutionLoading}
                className="px-3 py-2 text-sm font-medium text-white bg-red-600 rounded-lg hover:opacity-90 disabled:opacity-50"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-6 flex justify-end gap-3"/>
    </div>
  );
};

export default DuplicateResolution;
