// src/App.js
import React, { useState } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Link,
  useParams,
  Navigate,
} from 'react-router-dom';
import { useQuery, QueryClient, QueryClientProvider } from '@tanstack/react-query';
import axios from 'axios';
import './App.css';

const api = axios.create({ baseURL: 'http://localhost:4000/api' });
const queryClient = new QueryClient();

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

/* ---------- FULL ARTICLE VIEW ---------- */
function ArticlePage() {
  const { id } = useParams();
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
        <em>{article.source}</em> – {new Date(article.published).toLocaleString()}
      </p>

      {/* NEW: description block */}
      {article.description && (
        <>
          <h4 style={{ margin: '16px 0 4px' }}>Description</h4>
          <p style={{ marginTop: 0 }}>{article.description}</p>
        </>
      )}

      <h4 style={{ margin: '24px 0 4px' }}>Article</h4>
      <div dangerouslySetInnerHTML={{ __html: article.article }} />

      <hr style={{ margin: '24px 0' }} />
      <Link to="/">← back to list</Link>
    </div>
  );
}

/* ---------- LIST VIEW ---------- */
function ListPage() {
  const [page, setPage] = useState(0);
  const [size, setSize] = useState(50);           // ← now editable, default 50
  const [query, setQuery] = useState('');
  const [source, setSource] = useState('');
  const [failed, setFailed] = useState(false);

  const params = { page, size, q: query, source, failed };
  const { data, isFetching } = useArticles(params);
  const { data: sources } = useSources();

  const totalPages = data?.pages || 1;

  return (
    <div className="App">
      <h1>RSS Reader</h1>

      {/* Filters & sizing */}
      <div style={{ marginBottom: 16 }}>
        <input
          placeholder="Search…"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setPage(0);
          }}
        />
        <select
          value={source}
          onChange={(e) => {
            setSource(e.target.value);
            setPage(0);
          }}
        >
          <option value="">All sources</option>
          {sources?.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <label>
          <input
            type="checkbox"
            checked={failed}
            onChange={(e) => {
              setFailed(e.target.checked);
              setPage(0);
            }}
          />
          Failed only
        </label>

        {/* items-per-page box */}
        <label style={{ marginLeft: 12 }}>
          Per page:
          <input
            type="number"
            min="1"
            max="100"
            value={size}
            onChange={(e) => {
              setSize(Math.max(1, Math.min(100, Number(e.target.value) || 50)));
              setPage(0);
            }}
            style={{ width: 50, marginLeft: 4 }}
          />
        </label>
      </div>

      {isFetching && <p>Loading…</p>}

      {/* left-aligned title links */}
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {data?.rows.map((a) => (
          <li key={a._id} style={{ marginBottom: 8 }}>
            <Link
              to={`/article/${a._id}`}
              style={{
                fontWeight: 'bold',
                color: '#1a0dab',
                textDecoration: 'none',
                textAlign: 'left',
                display: 'block',
              }}
            >
              {a.title}
            </Link>
          </li>
        ))}
      </ul>

      {/* Pagination */}
      <div>
        {Array.from({ length: totalPages }, (_, i) => (
          <button
            key={i}
            disabled={i === page}
            onClick={() => setPage(i)}
          >
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