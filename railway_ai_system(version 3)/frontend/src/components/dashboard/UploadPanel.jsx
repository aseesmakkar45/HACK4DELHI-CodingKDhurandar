import React, { useState, useRef } from 'react';
import { UploadCloud, FileVideo, AlertCircle, Loader } from 'lucide-react';
import { useApi } from '../../hooks/useApi';
import './UploadPanel.css';

export default function UploadPanel({ onUploadStart, onUploadSuccess }) {
  const [file, setFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);
  const { request, loading, error } = useApi();

  const handleFileSelect = (e) => {
    const selected = e.target.files[0];
    if (selected && (selected.type.startsWith('video/') || selected.type.startsWith('image/') || selected.name.toLowerCase().endsWith('.mp4'))) {
      setFile(selected);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped && (dropped.type.startsWith('video/') || dropped.type.startsWith('image/') || dropped.name.toLowerCase().endsWith('.mp4'))) {
      setFile(dropped);
    }
  };

  const handeAnalyze = async () => {
    onUploadStart();
    
    try {
      const formData = new FormData();
      if (file) {
        formData.append('file', file);
      }

      const response = await request('/analyze', {
        method: 'POST',
        body: formData
      });
      
      if (response && response.run_id) {
        onUploadSuccess(response.run_id);
      }
    } catch (err) {
      console.error("Upload/Analyze failed");
    }
  };

  return (
    <div className="card-glass upload-panel">
      <h2 className="text-gradient" style={{ marginBottom: '2rem', textAlign: 'center', fontSize: '1.8rem' }}>Start Analysis</h2>
      
      <div 
        className={`drop-zone ${isDragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !file && fileInputRef.current?.click()}
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          onChange={handleFileSelect} 
          accept="video/*,image/*" 
          style={{ display: 'none' }} 
        />
        
        {file ? (
          <div className="file-preview">
            <FileVideo size={48} color="var(--accent-blue)" />
            <div className="file-info">
              <span className="file-name">{file.name}</span>
              <span className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
            </div>
            <button className="btn btn-clear" onClick={(e) => { e.stopPropagation(); setFile(null); }}>
              Remove
            </button>
          </div>
        ) : (
          <div className="upload-prompt">
            <UploadCloud size={48} color="var(--text-muted)" />
            <p><strong>Drag & Drop</strong> your railway inspection footage here</p>
            <span className="text-muted">or click to browse files</span>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <button 
        className={`btn btn-primary submit-btn ${loading ? 'btn-disabled' : ''}`}
        disabled={loading}
        onClick={handeAnalyze}
      >
        {loading ? <Loader className="animate-spin" size={18} /> : null}
        {loading ? 'Starting Analysis...' : 'Analyze Footage'}
      </button>
    </div>
  );
}
