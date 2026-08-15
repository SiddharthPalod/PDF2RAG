const { contextBridge, ipcRenderer } = require('electron');
const path = require('path');

// rag/frontend/preload.cjs → two levels up = PDF/ (project root).
// Exposed so App.jsx can build correct file:/// paths to testdata/images/.
const PROJECT_ROOT = path.join(__dirname, '../..').replace(/\\/g, '/');

contextBridge.exposeInMainWorld('api', {
  PROJECT_ROOT,
  askQuestion: (query) => ipcRenderer.send('ask-question', query),
  onPythonOutput: (callback) => ipcRenderer.on('python-output', (event, data) => callback(data)),
  onPythonDone: (callback) => ipcRenderer.on('python-done', (event, code) => callback(code)),
  removeListeners: () => {
    ipcRenderer.removeAllListeners('python-output');
    ipcRenderer.removeAllListeners('python-done');
  },
});
