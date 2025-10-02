// src/App.js  (pager window: first / last + up to 10 prev & next)
import React, { useState, useMemo, useEffect } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useParams,
  Navigate,
  useSearchParams,
  useLocation,
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
function useTags() {
  return useQuery({
    queryKey: ['tags'],
    queryFn: () => api.get('/tags').then(r => r.data),
  });
}

/* ---------- FULL ARTICLE VIEW ---------- */
function ArticlePage() {
  const { id } = useParams();
  const { search } = useLocation();
  const { data: article, isFetching } = useQuery({
    queryKey: ['article', id],
    queryFn: () => api.get(`/article/${id}`).then(r => r.data),
  });

  if (isFetching) return <p>Loading…</p>;
  if (!article) return <p>Not found</p>;

  return (
    <div className="article">
      <h2>{article.title}</h2>

      {/* NEW: link to the original story */}
      {article.link && (
        <p style={{ margin: '0 0 8px' }}>
          <a href={article.link} target="_blank" rel="noreferrer">
            🔗 Original story
          </a>
        </p>
      )}

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
      <Link to={`/${search}`}>← back to list</Link>
    </div>
  );
}

/* ---------- LIST VIEW ---------- */
function ListPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const [query, setQuery]       = useState(searchParams.get('q') || '');
  const [source, setSource]     = useState(searchParams.get('source') || '');
  const [failed, setFailed]     = useState(searchParams.get('failed') === 'true');
  const [from, setFrom]         = useState(searchParams.get('from') || '');
  const [to, setTo]             = useState(searchParams.get('to') || '');
  const [sort, setSort]         = useState(searchParams.get('sort') || 'published');
  const [page, setPage]         = useState(Number(searchParams.get('page') || 0));
  const [size, setSize]         = useState(Number(searchParams.get('size') || 50));
  const [selectedTag, setSelectedTag] = useState(searchParams.get('tag') || '');
  const [checkedIds, setCheckedIds] = useState([]);
  const [bulkTagInput, setBulkTagInput] = useState('');

  useEffect(() => {
    const next = new URLSearchParams();
    if (query)              next.set('q', query);
    if (source)             next.set('source', source);
    if (failed)             next.set('failed', 'true');
    if (from)               next.set('from', from);
    if (to)                 next.set('to', to);
    if (sort !== 'published') next.set('sort', sort);
    if (page > 0)           next.set('page', page);
    if (size !== 50)        next.set('size', size);
    if (selectedTag)        next.set('tag', selectedTag);
    setSearchParams(next, { replace: true });
  }, [query, source, failed, from, to, sort, page, size, selectedTag, setSearchParams]);

  const params = useMemo(() => {
    const p = { page, size, q: query, source, failed, sort };
    if (from) p.from = from;
    if (to) p.to = to;
    if (selectedTag) p.tags = selectedTag;
    return p;
  }, [page, size, query, source, failed, from, to, sort, selectedTag]);

  const { data, isFetching } = useArticles(params);
  const { data: sources } = useSources();
  const { data: globalRange } = useGlobalDateRange();
  const { data: daily } = useDailyCounts();
  const { data: allTags } = useTags();

  const totalPages = data?.pages || 1;
  const pagesAround = 5;
  const start = Math.max(0, page - pagesAround);
  const end   = Math.min(totalPages - 1, page + pagesAround);

  const dateRange = globalRange
    ? `${globalRange.min} – ${globalRange.max}`
    : null;

  const toTags = (str) => [...new Set(str.split(',').map(s => s.trim()).filter(Boolean))];

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
        <select value={source} onChange={(e) => { setSource(e.target.value); setPage(0); }}>
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
        <label>
          Sort:
          <select value={sort} onChange={(e) => { setSort(e.target.value); setPage(0); }} style={{ marginLeft: 4 }}>
            <option value="published">Newest first</option>
            <option value="title">Title A-Z</option>
          </select>
        </label>

        {/* Tag dropdown */}
        <select
          value={selectedTag}
          onChange={(e) => { setSelectedTag(e.target.value); setPage(0); }}
        >
          <option value="">All tags</option>
          {allTags?.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        {selectedTag && (
          <button
            onClick={async () => {
              if (!window.confirm(`Delete tag “${selectedTag}” globally?`)) return;
              await api.delete(`/tags/${selectedTag}`);
              queryClient.invalidateQueries(['tags']);
              queryClient.invalidateQueries(['articles']);
              setSelectedTag('');
            }}
            style={{ marginLeft: 8, color: '#d33' }}
          >
            Delete tag
          </button>
        )}
      </div>

      {/* Bulk-tag bar */}
      <div style={{ marginBottom: 16, display: 'flex', gap: 8 }}>
        <input
          placeholder="Tag checked articles (comma sep)"
          value={bulkTagInput}
          onChange={(e) => setBulkTagInput(e.target.value)}
          style={{ width: 220 }}
        />
        <button
          disabled={!checkedIds.length || !bulkTagInput}
          onClick={async () => {
            await api.post('/articles/bulk-tag', {
              ids: checkedIds,
              tags: toTags(bulkTagInput),
            });
            queryClient.invalidateQueries(['articles']);
            setCheckedIds([]);
            setBulkTagInput('');
          }}
        >
          Apply tags
        </button>
        <span style={{ marginLeft: 8, fontSize: 12, color: '#666' }}>
          {checkedIds.length} selected
        </span>
      </div>

      {isFetching && <p>Loading…</p>}

      {/* LEFT-ALIGNED titles */}
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {data?.rows.map((a) => (
          <li key={a._id} style={{ marginBottom: 8, textAlign: 'left' }}>
            <input
              type="checkbox"
              checked={checkedIds.includes(a._id)}
              onChange={(e) => {
                setCheckedIds(prev =>
                  e.target.checked
                    ? [...prev, a._id]
                    : prev.filter(id => id !== a._id)
                );
              }}
              style={{ marginRight: 6 }}
            />
            <Link
              to={`/article/${a._id}?${searchParams.toString()}`}
              style={{ fontWeight: 'bold', color: '#1a0dab', textDecoration: 'none' }}
            >
              {toISODate(a.published)} – {a.title}
            </Link>
            {a.tags?.length > 0 && (
              <span style={{ marginLeft: 8, fontSize: 12, color: '#555' }}>
                [{a.tags.join(', ')}]
              </span>
            )}
          </li>
        ))}
      </ul>

      {/* ---------- NEW PAGER ---------- */}
      <div style={{ marginTop: 16 }}>
        <button disabled={page === 0} onClick={() => setPage(0)}>First</button>

        {Array.from({ length: end - start + 1 }, (_, i) => start + i).map((n) => (
          <button
            key={n}
            disabled={n === page}
            onClick={() => setPage(n)}
            style={{ margin: '0 2px' }}
          >
            {n + 1}
          </button>
        ))}

        <button disabled={page === totalPages - 1} onClick={() => setPage(totalPages - 1)}>Last</button>
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