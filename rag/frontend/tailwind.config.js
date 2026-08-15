/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bg-color': '#0f111a',
        'panel-bg': '#1e2130',
        'text-primary': '#e2e8f0',
        'text-secondary': '#94a3b8',
        'accent': '#6366f1',
        'accent-hover': '#4f46e5',
        'user-msg': '#312e81',
        'ai-msg': '#1e293b',
        'border-color': 'rgba(255, 255, 255, 0.1)'
      }
    },
  },
  plugins: [],
}
