import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '@fontsource/ibm-plex-sans/400.css';
import '@fontsource/ibm-plex-sans/500.css';
import '@fontsource/ibm-plex-sans/600.css';
import '@fontsource/ibm-plex-sans/700.css';
import 'leaflet/dist/leaflet.css';
import './index.css';
import App from './App.jsx';

// Apply theme immediately (before React hydrates) to avoid FOUT
;(function () {
  let theme = 'dark';
  try {
    const stored = localStorage.getItem('nwis-theme');
    if (stored === 'dark' || stored === 'light') {
      theme = stored;
    } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
      theme = 'light';
    }
  } catch (_) {}
  document.documentElement.setAttribute('data-theme', theme);
})();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
