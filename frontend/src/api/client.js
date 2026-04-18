import axios from 'axios';

/**
 * Backend base URL. Override for a different host/port (LAN, Docker, cloud).
 * Create `frontend/.env.local` with e.g. VITE_API_BASE_URL=http://192.168.1.10:8000
 * @see https://vitejs.dev/guide/env-and-mode.html
 */
export const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8000';

let token = null;

const api = axios.create({ baseURL: API_BASE });

api.interceptors.request.use((config) => {
  if (token && !config.url.includes('/auth/')) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export async function login(username = 'admin', password = 'admin123') {
  const res = await api.post('/auth/token', { username, password });
  token = res.data.access_token;
  return token;
}

export async function getHealth() {
  const res = await api.get('/health');
  return res.data;
}

export async function getAtRiskCustomers(weekNumber, threshold = 0.40, limit = 50, riskLevel = null, search = null) {
  const params = { threshold, limit };
  if (weekNumber) params.week_number = weekNumber;
  if (riskLevel) params.risk_level = riskLevel;
  if (search && search.trim()) params.search = search.trim();
  const res = await api.get('/api/customers/at-risk', { params });
  return res.data;
}

export async function getCustomerDetail(customerId) {
  const res = await api.get(`/api/customers/${customerId}`);
  return res.data;
}

export async function getCustomerHistory(customerId) {
  const res = await api.get(`/api/customers/${customerId}/history`);
  return res.data;
}

export async function explainCustomer(customerId) {
  const res = await api.get(`/api/customers/${customerId}/explain`);
  return res.data;
}

export async function triggerIntervention(customerId, weekNumber = 52) {
  const res = await api.post('/api/interventions/trigger', {
    customer_id: customerId,
    week_number: weekNumber
  });
  return res.data;
}

export async function recordIntervention(recordData) {
  const res = await api.post('/api/interventions/record', recordData);
  return res.data;
}

export async function getCustomerTimeline(customerId) {
  const res = await api.get(`/api/customers/${customerId}/timeline`);
  return res.data;
}

export async function getAbilityWillingness(customerId) {
  const res = await api.get(`/api/customers/${customerId}/ability-willingness`);
  return res.data;
}

export async function getInterventionLog(page = 1, pageSize = 50) {
  const res = await api.get('/api/interventions/log', {
    params: { page, page_size: pageSize }
  });
  return res.data;
}

export async function getOverviewMetrics() {
  const res = await api.get('/api/metrics/overview');
  return res.data;
}

// ── Direct Model Prediction (No Auth Required) ──

// Removed


// ── Landing Page Metrics ──
export async function getLandingMetrics() {
  const res = await api.get('/api/metrics/landing');
  return res.data;
}

export async function getLatestStream() {
  const res = await api.get('/api/stream/latest');
  return res.data;
}

// ── Rules Engine ──
export async function getRulesImpact(rules) {
  const res = await api.post('/api/rules/impact', rules);
  return res.data;
}

export async function saveRulesConfig(rules) {
  const res = await api.post('/api/rules/save', rules);
  return res.data;
}

// ── Context Engine ──
export async function getCustomerContext(customerId) {
  const res = await api.get(`/api/customers/${customerId}/context`);
  return res.data;
}

export async function getPortfolioContext() {
  const res = await api.get('/api/context/portfolio-summary');
  return res.data;
}

export default api;
