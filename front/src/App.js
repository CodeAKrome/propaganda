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
      <p>
        <em>{article.source}</em> – {new Date(article.published).toLocaleString()}
      </p>
      <div dangerouslySetInnerHTML={{ __html: article.article }} />
      <hr />
      <Link to="/">← back to list</Link>
    </div>
  );
}

/* ---------- LIST VIEW (title only) ---------- */
function ListPage() {
  const [page, setPage] = useState(0);
  const [size] = useState(20);
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

      {/* Filters */}
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
      </div>

      {isFetching && <p>Loading…</p>}

      {/* Title-only list */}
      <ul>
        {data?.rows.map((a) => (
          <li key={a._id}>
            <Link
              to={`/article/${a._id}`}
              style={{ fontWeight: 'bold', color: '#1a0dab', textDecoration: 'none' }}
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