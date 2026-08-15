import { useState, useEffect, useRef } from 'react';

// ---------------------------------------------------------------------------
// Design tokens — mirrors the colour palette from tailwind.config.js but as
// JS constants so we can use them in inline styles (matching senior App.jsx).
// ---------------------------------------------------------------------------
const COLORS = {
  bgColor: '#0f111a',
  panelBg: '#1e2130',
  textPrimary: '#e2e8f0',
  textSecondary: '#94a3b8',
  accent: '#6366f1',
  accentHover: '#4f46e5',
  userMsg: '#312e81',
  aiMsg: '#1e293b',
  border: 'rgba(255, 255, 255, 0.1)',
};

// Protocol constant — must match the marker emitted by answer_generator.py
const IMAGES_MARKER = '__IMAGES_JSON__:';

// Smooth typing animation component for AI responses
function TypewriterText({ text, done }) {
  const [displayedText, setDisplayedText] = useState('');

  useEffect(() => {
    if (text.length > displayedText.length) {
      // Dynamic speed: if we fall behind the stream, reveal more characters at once
      const diff = text.length - displayedText.length;
      const charsToAdd = diff > 50 ? 5 : diff > 15 ? 2 : 1;
      
      const timeout = setTimeout(() => {
        setDisplayedText(text.slice(0, displayedText.length + charsToAdd));
      }, 15); // 15ms per tick is a very smooth 60fps-like aesthetic
      return () => clearTimeout(timeout);
    }
  }, [text, displayedText]);

  return <>{displayedText || (!done && 'Thinking...')}</>;
}

export default function App() {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [isThinking, setIsThinking] = useState(false);

  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isThinking]);

  useEffect(() => {
    if (!window.api) return;

    window.api.onPythonOutput((data) => {
      setChatHistory((prev) => {
        const newHistory = [...prev];
        const lastMsg = newHistory[newHistory.length - 1];

        if (lastMsg && lastMsg.sender === 'ai' && !lastMsg.done) {
          let updatedText = lastMsg.text + data;
          let images = [];

          const markerIndex = updatedText.lastIndexOf(IMAGES_MARKER);
          if (markerIndex !== -1) {
            const textContent = updatedText.substring(0, markerIndex).trim();
            try {
              const jsonStr = updatedText.substring(markerIndex + IMAGES_MARKER.length).trim();
              images = JSON.parse(jsonStr);
            } catch (e) {
              console.error('Failed to parse image references:', e);
            }
            updatedText = textContent;
            lastMsg.images = images;
          }

          lastMsg.text = updatedText;
          return newHistory;
        }
        return prev;
      });
    });

    window.api.onPythonDone((code) => {
      setIsThinking(false);
      setChatHistory((prev) => {
        const newHistory = [...prev];
        const lastMsg = newHistory[newHistory.length - 1];
        if (lastMsg && lastMsg.sender === 'ai') {
          lastMsg.done = true;
        }
        return newHistory;
      });
    });

    return () => {
      window.api.removeListeners();
    };
  }, []);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim() || isThinking) return;

    const userQuery = query.trim();
    setChatHistory((prev) => [...prev, { sender: 'user', text: userQuery }]);
    setQuery('');
    setIsThinking(true);
    setChatHistory((prev) => [...prev, { sender: 'ai', text: '', done: false }]);

    if (window.api) {
      window.api.askQuestion(userQuery);
    } else {
      // Mock fallback for browser dev without Electron
      setTimeout(() => {
        setChatHistory((prev) => {
          const newHistory = [...prev];
          newHistory[newHistory.length - 1].text = 'Mock response (Electron API not found).';
          newHistory[newHistory.length - 1].done = true;
          return newHistory;
        });
        setIsThinking(false);
      }, 1000);
    }
  };

  const getImagePath = (imgPath) => {
    // PROJECT_ROOT is exposed by preload.cjs as the PDF/ project root directory.
    const root = window.api?.PROJECT_ROOT || '';
    const clean = imgPath.startsWith('/') ? imgPath.slice(1) : imgPath;
    // Normalise slashes and avoid double-slashes.
    const full = `${root}/${clean}`.replace(/\/+/g, '/');
    return 'file:///' + full;
  };

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      width: '100vw',
      maxWidth: 960,
      height: '100vh',
      background: COLORS.panelBg,
      boxShadow: '0 25px 50px -12px rgba(0,0,0,0.5)',
      margin: '0 auto',
      overflow: 'hidden',
    }}>

      {/* Header */}
      <header style={{
        padding: '20px 32px',
        borderBottom: `1px solid ${COLORS.border}`,
        background: 'rgba(30, 33, 48, 0.8)',
        backdropFilter: 'blur(8px)',
        flexShrink: 0,
      }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 600, color: COLORS.textPrimary, letterSpacing: '-0.02em' }}>
          Document Knowledge Base
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: 13, color: COLORS.textSecondary }}>
          Ask anything about your processed documents.
        </p>
      </header>

      {/* Chat Area */}
      <div style={{
        flex: 1,
        overflowY: 'auto',
        padding: 32,
        display: 'flex',
        flexDirection: 'column',
        gap: 20,
      }}>
        {chatHistory.map((msg, idx) => (
          <div
            key={idx}
            style={{
              maxWidth: '80%',
              minWidth: 0,
              padding: '12px 16px',
              borderRadius: 12,
              lineHeight: 1.6,
              wordBreak: 'break-word',
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              background: msg.sender === 'user' ? COLORS.userMsg : COLORS.aiMsg,
              border: msg.sender === 'ai' ? `1px solid ${COLORS.border}` : 'none',
              borderBottomRightRadius: msg.sender === 'user' ? 2 : 12,
              borderBottomLeftRadius: msg.sender === 'ai' ? 2 : 12,
              color: COLORS.textPrimary,
              animation: 'fadeIn 0.3s ease-out',
            }}
          >
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {msg.sender === 'ai' ? (
                <TypewriterText text={msg.text} done={msg.done} />
              ) : (
                msg.text
              )}
            </div>
            {msg.images && msg.images.length > 0 && (
              <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {msg.images.map((img, i) => (
                  <img
                    key={i}
                    src={getImagePath(img)}
                    alt="Document reference"
                    style={{
                      maxWidth: '100%',
                      height: 'auto',
                      borderRadius: 8,
                      border: `1px solid ${COLORS.border}`,
                      cursor: 'zoom-in',
                    }}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      {/* Input Form */}
      <form
        onSubmit={handleSubmit}
        style={{
          padding: '20px 32px',
          borderTop: `1px solid ${COLORS.border}`,
          background: COLORS.panelBg,
          display: 'flex',
          gap: 16,
          alignItems: 'center',
          flexShrink: 0,
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Type your question..."
          disabled={isThinking}
          style={{
            flex: 1,
            background: 'rgba(255,255,255,0.05)',
            border: `1px solid ${COLORS.border}`,
            borderRadius: 8,
            padding: '14px 20px',
            fontSize: 15,
            color: COLORS.textPrimary,
            outline: 'none',
            transition: 'border-color 0.2s',
            opacity: isThinking ? 0.5 : 1,
          }}
        />
        <button
          type="submit"
          disabled={isThinking || !query.trim()}
          style={{
            background: isThinking || !query.trim() ? COLORS.textSecondary : COLORS.accent,
            color: 'white',
            border: 'none',
            borderRadius: 8,
            width: 54,
            height: 54,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            cursor: isThinking || !query.trim() ? 'not-allowed' : 'pointer',
            flexShrink: 0,
            transition: 'background 0.2s',
          }}
        >
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </form>
    </div>
  );
}
