import { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';
const getSessionId = () => {
  let id = sessionStorage.getItem('pdf_emb_session');
  if (!id) {
    id = `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    sessionStorage.setItem('pdf_emb_session', id);
  }
  return id;
};

export default function App() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [downloadFormat, setDownloadFormat] = useState('json');
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f && f.type === 'application/pdf') {
      setFile(f);
      setError(null);
      setResult(null);
    } else if (f) {
      setError('Please select a PDF file.');
      setFile(null);
    }
  };

  const processPdf = async () => {
    if (!file) {
      setError('Please select a PDF first.');
      return;
    }
    setProcessing(true);
    setError(null);
    setResult(null);
    const formData = new FormData();
    formData.append('file', file);
    try {
      const sessionId = getSessionId();
      const res = await axios.post(`${API_BASE}/api/process-pdf`, formData, {
        headers: { 'Content-Type': 'multipart/form-data', 'X-Session-ID': sessionId },
        timeout: 600000,
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Processing failed.');
    } finally {
      setProcessing(false);
    }
  };

  const downloadEmbeddings = async () => {
    if (!result) {
      setError('Process a PDF first, then download.');
      return;
    }
    setError(null);
    try {
      const sessionId = getSessionId();
      const url = `${API_BASE}/api/embeddings/download?format=${downloadFormat}&include_embedding_vector=true`;
      const res = await fetch(url, { method: 'GET', headers: { 'X-Session-ID': sessionId } });
      if (!res.ok) throw new Error(res.statusText);
      const blob = await res.blob();
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename=(.+)/i);
      const filename = match ? match[1].trim() : `embeddings.${downloadFormat}`;
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = filename;
      a.click();
      URL.revokeObjectURL(a.href);
    } catch (err) {
      setError(err.message || 'Download failed.');
    }
  };

  return (
    <div style={{
      maxWidth: 560,
      margin: '40px auto',
      padding: 24,
      background: 'white',
      borderRadius: 12,
      boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
    }}>
      <h1 style={{ margin: '0 0 8px', fontSize: 22, color: '#1a1a1a' }}>
        PDF Embeddings
      </h1>
      <p style={{ margin: '0 0 24px', fontSize: 14, color: '#666' }}>
        Upload a PDF to extract text and images, run OCR + BLIP, and download embeddings.
      </p>

      <div style={{
        border: '2px dashed #ccc',
        borderRadius: 8,
        padding: 24,
        textAlign: 'center',
        marginBottom: 16,
        background: '#fafafa',
      }}>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: '10px 20px',
            fontSize: 14,
            background: '#2563eb',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: 'pointer',
          }}
        >
          Choose PDF
        </button>
        {file && (
          <p style={{ margin: '12px 0 0', fontSize: 14, color: '#374151' }}>
            {file.name}
          </p>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 16 }}>
        <button
          type="button"
          onClick={processPdf}
          disabled={!file || processing}
          style={{
            padding: '10px 20px',
            fontSize: 14,
            background: processing ? '#9ca3af' : '#059669',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: processing ? 'not-allowed' : 'pointer',
          }}
        >
          {processing ? 'Processing…' : 'Process PDF'}
        </button>
        <button
          type="button"
          onClick={downloadEmbeddings}
          disabled={!result || processing}
          style={{
            padding: '10px 20px',
            fontSize: 14,
            background: !result ? '#d1d5db' : '#7c3aed',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            cursor: !result ? 'not-allowed' : 'pointer',
          }}
        >
          Download Embeddings
        </button>
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={{ fontSize: 13, color: '#6b7280', marginRight: 8 }}>Format:</label>
        <select
          value={downloadFormat}
          onChange={(e) => setDownloadFormat(e.target.value)}
          style={{ padding: '6px 10px', fontSize: 13, borderRadius: 4, border: '1px solid #d1d5db' }}
        >
          <option value="json">JSON</option>
          <option value="csv">CSV</option>
          <option value="xlsx">Excel (xlsx)</option>
        </select>
      </div>

      {result && (
        <div style={{
          padding: 12,
          background: '#f0fdf4',
          borderRadius: 6,
          fontSize: 13,
          color: '#166534',
        }}>
          <strong>Done.</strong> Rows: {result.total_rows ?? 0} · Text chunks: {result.total_text_chunks ?? 0} · Images: {result.total_images ?? 0}
        </div>
      )}

      {error && (
        <div style={{
          marginTop: 12,
          padding: 12,
          background: '#fef2f2',
          borderRadius: 6,
          fontSize: 13,
          color: '#b91c1c',
        }}>
          {error}
        </div>
      )}
    </div>
  );
}
