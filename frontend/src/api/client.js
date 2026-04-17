import axios from 'axios';

const API_BASE = 'http://localhost:8000';
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

export async function getAtRiskCustomers(weekNumber, threshold = 0.40, limit = 50) {
  const params = { threshold, limit };
  if (weekNumber) params.week_number = weekNumber;
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

export async function predictRisk(features) {
  const res = await api.post('/api/predict', features);
  return res.data;
}

export async function getModelInfo() {
  const res = await api.get('/api/model-info');
  return res.data;
}

export async function predictBatch(loans) {
  const res = await api.post('/api/predict/batch', loans);
  return res.data;
}

export default api;
