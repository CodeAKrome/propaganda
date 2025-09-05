// src/App.js  (complete, back-link preserves source filter via query-string)
import React, { useState, useMemo } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useParams,
  Navigate,
  useSearchParams,
} from 'react-router-dom';
import { useQuery, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import CalendarHeatmap from 'react-calendar-heatmap';
import 'react-calendar-heatmap/dist/styles.css';
import axios from 'axios';
import './App.css';

const api = axios.create({ baseURL: 'http://localhost:4000/api' });
const queryClient = new QueryClient();

/* ---------- UTIL ---------- */
const toISODate = (d) => new Date(d).toISOString().slice(0, 10);

/* ---------- HOOKS ---------- */
function useArticles(params) {
  return useQuery({
    queryKey: ['articles', params],
    queryFn: () => api.get('/articles', { params }).then(r => r.data),
    placeholderData: (prev) => prev,
  });
}
function useSources() {
  return useQuery({
    queryKey: ['sources'],
    queryFn: () => api('/sources').then(r => r.data),
  });
}
function useGlobalDateRange() {
  return useQuery({
    queryKey: ['global-date-range'],
    queryFn: () => api.get('/articles/date-range').then(r => r.data),
    staleTime: 5 * 60 * 1000,
  });
}
function useDailyCounts() {
  return useQuery({
    queryKey: ['daily-counts'],
    queryFn: () => api.get('/articles/daily-counts').then(r => r.data),
    staleTime: 5 * 60 * 1000,
  });
}

/* ---------- FULL ARTICLE VIEW ---------- */
function ArticlePage() {
  const { id } = useParams();
  const [searchParams] = useSearchParams();          // ← read current filter
  const source = searchParams.get('source') || '';
  const { data: article, isFetching } = useQuery({
    queryKey: ['article', id],
    queryFn: () => api.get(`/article/${id}`).then(r => r.data),
  });

  if (isFetching) return <p>Loading…</p>;
  if (!article) return <p>Not found</p>;

  return (
    <div className="article">
      <h2>{article.title}</h2>
      <p style={{ marginBottom: 4 }}>
        <em>{article.source}</em> – {toISODate(article.published)}
      </p>

      {article.description && (
        <>
          <h4 style={{ margin: '16px 0 4px' }}>Description</h4>
          <p style={{ marginTop: 0 }}>{article.description}</p>
        </>
      )}

      <h4 style={{ margin: '24px 0 4px' }}>Article</h4>
      <div dangerouslySetInnerHTML={{ __html: article.article }} />

      <hr style={{ margin: '24px 0' }} />
      {/* source-aware back link */}
      <Link to={`/?source=${encodeURIComponent(source)}`}>← back to list</Link>
    </div>
  );
}

/* ---------- LIST VIEW ---------- */
function ListPage() {
  const [searchParams, setSearchParams] = useSearchParams(); // ← URL state
  const source = searchParams.get('source') || '';           // ← read filter
  const [page, setPage] = useState(0);
  const [size, setSize] = useState(50);
  const [query, setQuery] = useState('');
  const [failed, setFailed] = useState(false);
  const [from, setFrom] = useState('');
  const [to, setTo] = useState('');

  const params = useMemo(() => {
    const p = { page, size, q: query, source, failed };
    if (from) p.from = from;
    if (to) p.to = to;
    return p;
  }, [page, size, query, source, failed, from, to]);

  const { data, isFetching } = useArticles(params);
  const { data: sources } = useSources();
  const { data: globalRange } = useGlobalDateRange();
  const { data: daily } = useDailyCounts();

  const totalPages = data?.pages || 1;
  const dateRange = globalRange
    ? `${globalRange.min} – ${globalRange.max}`
    : null;

  // keep URL in sync when user changes source
  const handleSourceChange = (e) => {
    const newSrc = e.target.value;
    setSearchParams(prev => {
      prev.set('source', newSrc);
      return prev;
    });
    setPage(0);
  };

  return (
    <div className="App">
      <h1>RSS Reader</h1>

      {/* Calendar heat-map */}
      {daily && (
        <div style={{ marginBottom: 24 }}>
          <CalendarHeatmap
            startDate={new Date(new Date().setFullYear(new Date().getFullYear() - 1))}
            endDate={new Date()}
            values={daily}
            classForValue={(v) =>
              v ? `color-github-${Math.min(v.count, 4)}` : 'color-github-0'
            }
            titleForValue={(v) => (v ? `${v.date}: ${v.count} articles` : '')}
            showWeekdayLabels
          />
          <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>
            Articles per day (last 365 days)
          </div>
        </div>
      )}

      {dateRange && <h3 style={{ marginTop: 0 }}>Articles: {dateRange}</h3>}

      {/* Controls */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <input placeholder="Search…" value={query} onChange={(e) => { setQuery(e.target.value); setPage(0); }} />
        <select value={source} onChange={handleSourceChange}>
          <option value="">All sources</option>
          {sources?.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <label>
          <input type="checkbox" checked={failed} onChange={(e) => { setFailed(e.target.checked); setPage(0); }} />
          Failed only
        </label>
        <label>
          From:<input type="date" value={from} onChange={(e) => { setFrom(e.target.value); setPage(0); }} />
        </label>
        <label>
          To:<input type="date" value={to} onChange={(e) => { setTo(e.target.value); setPage(0); }} />
        </label>
        <label>
          Per page:<input type="number" min="1" max="100" value={size} onChange={(e) => {
            setSize(Math.max(1, Math.min(100, Number(e.target.value) || 50))); setPage(0);
          }} style={{ width: 50 }} />
        </label>
      </div>

      {isFetching && <p>Loading…</p>}

      {/* LEFT-ALIGNED titles */}
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {data?.rows.map((a) => (
          <li key={a._id} style={{ marginBottom: 8, textAlign: 'left' }}>
            <Link
              to={`/article/${a._id}?source=${encodeURIComponent(source)}`} // pass filter forward
              style={{ fontWeight: 'bold', color: '#1a0dab', textDecoration: 'none' }}
            >
              {toISODate(a.published)} – {a.title}
            </Link>
          </li>
        ))}
      </ul>

      {/* Pagination */}
      <div>
        {Array.from({ length: totalPages }, (_, i) => (
          <button key={i} disabled={i === page} onClick={() => setPage(i)}>
            {i + 1}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------- ROOT ---------- */
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          <Route path="/" element={<ListPage />} />
          <Route path="/article/:id" element={<ArticlePage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </QueryClientProvider>
  );
}

export default App;