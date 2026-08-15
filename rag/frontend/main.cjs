const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  const isDev = process.env.NODE_ENV === 'development';

  if (isDev) {
    // Only tries localhost if explicitly started in development mode (e.g. concurrently with Vite)
    mainWindow.loadURL('http://localhost:5173');
  } else {
    // Default behavior for npm run electron:dev (which builds first)
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// Ensure any spawned Python process is cleaned up when Electron exits —
// matches the lifecycle pattern from pdf-embeddings-app/frontend/main.cjs.
app.on('will-quit', () => {
  if (pythonProcess) {
    if (process.platform === 'win32') {
      spawn('taskkill', ['/pid', pythonProcess.pid, '/f', '/t']);
    } else {
      pythonProcess.kill('SIGINT');
    }
    pythonProcess = null;
  }
});

// IPC Handler — spawn answer_generator.py for each user question and stream
// stdout back to the renderer process via IPC events.
ipcMain.on('ask-question', (event, query) => {
  const scriptPath = path.join(__dirname, '../scripts/chat/answer_generator.py');

  pythonProcess = spawn('python', [scriptPath, query], {
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
  });

  pythonProcess.stdout.on('data', (data) => {
    event.sender.send('python-output', data.toString('utf8'));
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python stderr]: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    event.sender.send('python-done', code);
    pythonProcess = null;
  });
});
