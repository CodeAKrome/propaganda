import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';
import './App.css';

const api = axios.create({ baseURL: 'http://localhost:4000/api' });

function useArticles(params) {
  return useQuery({
    queryKey: ['articles', params],
    queryFn:  () => api.get('/articles', { params }).then(r => r.data),
    placeholderData: (prev) => prev   // replaces keepPreviousData: true
  });
}

function App() {
  const [page, setPage] = useState(0);
  const [size] = useState(20);
  const [query, setQuery] = useState('');
  const [source, setSource] = useState('');
  const [failed, setFailed] = useState(false);

  const { data, isFetching } = useArticles({ page, size, q: query, source, failed });
  const { data: sources } = useQuery({
    queryKey: ['sources'],
    queryFn:  () => api('/sources').then(r => r.data)
  });

  const totalPages = data?.pages || 1;

  return (
    <div className="App">
      <h1>RSS Reader</h1>

      <div style={{ marginBottom: 16 }}>
        <input
          placeholder="Search…"
          value={query}
          onChange={e => { setQuery(e.target.value); setPage(0); }}
        />
        <select value={source} onChange={e => { setSource(e.target.value); setPage(0); }}>
          <option value="">All sources</option>
          {sources?.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
        <label>
          <input type="checkbox" checked={failed} onChange={e => { setFailed(e.target.checked); setPage(0); }} />
          Failed only
        </label>
      </div>

      {isFetching && <p>Loading…</p>}

      <ul>
        {data?.rows.map(a => (
          <li key={a._id}>
            <strong>{a.title}</strong>
            <div>{a.source} – {new Date(a.published).toLocaleString()}</div>
            <p>{a.description}</p>
            <a href={`/article/${a._id}`}>Read more…</a>
          </li>
        ))}
      </ul>

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

export default App;
