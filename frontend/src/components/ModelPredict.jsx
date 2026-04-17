import React, { useState, useEffect, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { predictRisk, getModelInfo } from '../api/client';

const FEATURE_FIELDS = [
  { key: 'total_rec_late_fee', label: 'Late Fees Received', prefix: '$', default: 0, max: 500, step: 1, desc: 'Total late fees received to date' },
  { key: 'recoveries', label: 'Recoveries', prefix: '$', default: 0, max: 5000, step: 10, desc: 'Post-charge-off recovery amount' },
  { key: 'last_pymnt_amnt', label: 'Last Payment Amount', prefix: '$', default: 357.48, max: 5000, step: 10, desc: 'Last payment received' },
  { key: 'loan_amnt_div_instlmnt', label: 'Loan / Installment Ratio', prefix: '', default: 36.1, max: 100, step: 0.1, desc: 'Loan amount ÷ monthly installment' },
  { key: 'debt_settlement_flag', label: 'Debt Settlement Flag', prefix: '', default: 0, max: 1, step: 1, desc: '1 = debt settlement active' },
  { key: 'loan_age', label: 'Loan Age (months)', prefix: '', default: 36, max: 120, step: 1, desc: 'Months since loan origination' },
  { key: 'total_rec_int', label: 'Total Interest Received', prefix: '$', default: 2214.92, max: 20000, step: 10, desc: 'Total interest received to date' },
  { key: 'out_prncp', label: 'Outstanding Principal', prefix: '$', default: 0, max: 50000, step: 100, desc: 'Remaining principal balance' },
  { key: 'time_since_last_credit_pull', label: 'Months Since Credit Pull', prefix: '', default: 1, max: 60, step: 1, desc: 'Months since last credit check' },
  { key: 'time_since_last_payment', label: 'Months Since Last Payment', prefix: '', default: 1, max: 60, step: 1, desc: 'Months since last payment' },
  { key: 'int_rate_pct', label: 'Interest Rate (%)', prefix: '', default: 13.56, max: 35, step: 0.01, desc: 'Annual interest rate' },
  { key: 'total_rec_prncp', label: 'Total Principal Received', prefix: '$', default: 10000, max: 50000, step: 100, desc: 'Total principal received to date' },
];

// Map CSV headers (various formats) to our feature keys
const CSV_HEADER_MAP = {
  'total_rec_late_fee': 'total_rec_late_fee',
  'recoveries': 'recoveries',
  'last_pymnt_amnt': 'last_pymnt_amnt',
  'loan_amnt_div_instlmnt': 'loan_amnt_div_instlmnt',
  'debt_settlement_flag': 'debt_settlement_flag',
  'loan_age': 'loan_age',
  'total_rec_int': 'total_rec_int',
  'out_prncp': 'out_prncp',
  'time_since_last_credit_pull': 'time_since_last_credit_pull',
  'time_since_last_payment': 'time_since_last_payment',
  'int_rate%': 'int_rate_pct',
  'int_rate_pct': 'int_rate_pct',
  'int_rate': 'int_rate_pct',
  'total_rec_prncp': 'total_rec_prncp',
};

const PRESETS = {
  healthy: {
    name: '✅ Healthy Loan',
    desc: 'Fully performing, on-time payments, low risk',
    values: { total_rec_late_fee: 0, recoveries: 0, last_pymnt_amnt: 485.31, loan_amnt_div_instlmnt: 30.7, debt_settlement_flag: 0, loan_age: 36, total_rec_int: 4200, out_prncp: 0, time_since_last_credit_pull: 1, time_since_last_payment: 1, int_rate_pct: 10.5, total_rec_prncp: 15000 }
  },
  warning: {
    name: '⚠️ Warning Signs',
    desc: 'Missed payments, rising interest, stress signals',
    values: { total_rec_late_fee: 35, recoveries: 0, last_pymnt_amnt: 0, loan_amnt_div_instlmnt: 45, debt_settlement_flag: 0, loan_age: 18, total_rec_int: 1800, out_prncp: 8500, time_since_last_credit_pull: 3, time_since_last_payment: 4, int_rate_pct: 22.5, total_rec_prncp: 3200 }
  },
  critical: {
    name: '🚨 High Risk',
    desc: 'Charged-off, debt settlement, recovery in progress',
    values: { total_rec_late_fee: 120, recoveries: 2500, last_pymnt_amnt: 0, loan_amnt_div_instlmnt: 60, debt_settlement_flag: 1, loan_age: 48, total_rec_int: 3200, out_prncp: 12000, time_since_last_credit_pull: 8, time_since_last_payment: 12, int_rate_pct: 28.9, total_rec_prncp: 4000 }
  }
};

function RiskGauge({ value, label, color, size = 120 }) {
  const angle = value * 180;
  const r = size / 2 - 10;
  const cx = size / 2, cy = size / 2 + 10;
  const endAngle = (180 - angle) * Math.PI / 180;
  const ex = cx + r * Math.cos(endAngle);
  const ey = cy - r * Math.sin(endAngle);
  const largeArc = angle > 90 ? 1 : 0;
  const bgPath = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;
  const valPath = `M ${cx - r} ${cy} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey}`;

  return (
    <div className="gauge-container" style={{ textAlign: 'center' }}>
      <svg width={size} height={size / 2 + 30} viewBox={`0 0 ${size} ${size / 2 + 30}`}>
        <path d={bgPath} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="8" strokeLinecap="round" />
        <path d={valPath} fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
          style={{ transition: 'all 1s ease', filter: `drop-shadow(0 0 8px ${color}80)` }} />
        <text x={cx} y={cy - 8} textAnchor="middle" fill={color} fontFamily="'Syne', sans-serif" fontWeight="800" fontSize="24">
          {(value * 100).toFixed(0)}
        </text>
        <text x={cx} y={cy + 12} textAnchor="middle" fill="#5a5a7a" fontFamily="'DM Sans', sans-serif" fontSize="10">
          {label}
        </text>
      </svg>
    </div>
  );
}

function AnimatedNumber({ target, duration = 1200, decimals = 0, suffix = '' }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 4);
      setVal(ease * target);
      if (progress < 1) ref.current = requestAnimationFrame(step);
    };
    ref.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(ref.current);
  }, [target, duration]);
  return <span>{val.toFixed(decimals)}{suffix}</span>;
}

// ── CSV Parser ──
function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return null;
  const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
  const values = lines[1].split(',').map(v => v.trim().replace(/"/g, ''));

  const mapped = {};
  headers.forEach((header, i) => {
    const normalizedHeader = header.toLowerCase().trim();
    const matchedKey = CSV_HEADER_MAP[normalizedHeader] || CSV_HEADER_MAP[header];
    if (matchedKey && values[i] !== undefined) {
      mapped[matchedKey] = parseFloat(values[i]) || 0;
    }
  });
  return mapped;
}

// ── PDF Report Generator ──
async function generatePDFReport(result, features) {
  const { default: html2pdf } = await import('html2pdf.js');

  const riskColor = result.ensemble_prob >= 0.70 ? '#ff4757' : result.ensemble_prob >= 0.40 ? '#ff6b35' : '#06ffa5';
  const riskLevel = result.risk_level || 'LOW';
  const dateStr = new Date().toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' });
  const timeStr = new Date().toLocaleTimeString('en-IN', { hour12: false });

  const shapRows = (result.all_shap || []).map(s => {
    const dir = s.direction === 'INCREASES_RISK' ? '↑ RISK' : '↓ SAFE';
    const dirColor = s.direction === 'INCREASES_RISK' ? '#ff4757' : '#06ffa5';
    return `<tr>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;color:#c0c0d0;">${(s.feature || '').replace(/_/g, ' ')}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;font-family:monospace;color:${dirColor};text-align:right;">${s.contribution >= 0 ? '+' : ''}${(s.contribution || 0).toFixed(4)}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;color:${dirColor};text-align:right;font-weight:600;">${dir}</td>
    </tr>`;
  }).join('');

  const featureRows = FEATURE_FIELDS.map(f => {
    const val = features[f.key] ?? f.default;
    return `<tr>
      <td style="padding:6px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;color:#c0c0d0;">${f.label}</td>
      <td style="padding:6px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;font-family:monospace;color:#00d4ff;text-align:right;">${f.prefix}${Number(val).toLocaleString(undefined, { maximumFractionDigits: 2 })}</td>
    </tr>`;
  }).join('');

  const html = `
  <div style="font-family:'Segoe UI',Arial,sans-serif;background:#0a0a1a;color:#e0e0f0;padding:40px;min-height:100%;">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #1a1a3a;padding-bottom:20px;margin-bottom:24px;">
      <div>
        <h1 style="margin:0;font-size:28px;color:#00d4ff;letter-spacing:1px;">Praeventix</h1>
        <p style="margin:4px 0 0;font-size:12px;color:#6a6a8a;">AI-Powered Risk Intelligence Report</p>
      </div>
      <div style="text-align:right;">
        <div style="font-size:12px;color:#6a6a8a;">Report Generated</div>
        <div style="font-size:14px;color:#c0c0d0;font-family:monospace;">${dateStr} — ${timeStr}</div>
        <div style="font-size:11px;color:#6a6a8a;margin-top:2px;">CONFIDENTIAL — For Authorized Personnel Only</div>
      </div>
    </div>

    <!-- Risk Score Hero -->
    <div style="background:linear-gradient(135deg, #0f0f2a, #161640);border:1px solid ${riskColor}40;border-radius:12px;padding:24px 32px;margin-bottom:24px;text-align:center;">
      <div style="font-size:14px;color:#6a6a8a;margin-bottom:8px;text-transform:uppercase;letter-spacing:2px;">Ensemble Risk Score</div>
      <div style="font-size:72px;font-weight:800;color:${riskColor};line-height:1;">${(result.ensemble_prob * 100).toFixed(1)}%</div>
      <div style="display:inline-block;margin-top:12px;padding:6px 20px;border-radius:20px;background:${riskColor}20;color:${riskColor};font-weight:700;font-size:14px;letter-spacing:1px;border:1px solid ${riskColor}40;">
        ${riskLevel} RISK
      </div>
      ${result.anomaly_flag ? '<div style="margin-top:8px;color:#ff4757;font-size:12px;font-weight:600;">⚠ ANOMALY DETECTED BY ISOLATION FOREST</div>' : ''}
    </div>

    <!-- Model Breakdown -->
    <div style="display:flex;gap:16px;margin-bottom:24px;">
      <div style="flex:1;background:#0f0f2a;border:1px solid #1a1a3a;border-radius:10px;padding:20px;text-align:center;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">LightGBM</div>
        <div style="font-size:32px;font-weight:700;color:#00d4ff;margin:8px 0;">${(result.lgbm_prob * 100).toFixed(1)}%</div>
        <div style="font-size:10px;color:#4a4a6a;">Gradient Boosting</div>
      </div>
      <div style="flex:1;background:#0f0f2a;border:1px solid #1a1a3a;border-radius:10px;padding:20px;text-align:center;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">GRU Neural Net</div>
        <div style="font-size:32px;font-weight:700;color:#7c3aed;margin:8px 0;">${(result.gru_prob * 100).toFixed(1)}%</div>
        <div style="font-size:10px;color:#4a4a6a;">Temporal Patterns</div>
      </div>
      <div style="flex:1;background:#0f0f2a;border:1px solid #1a1a3a;border-radius:10px;padding:20px;text-align:center;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Ensemble</div>
        <div style="font-size:32px;font-weight:700;color:${riskColor};margin:8px 0;">${(result.ensemble_prob * 100).toFixed(1)}%</div>
        <div style="font-size:10px;color:#4a4a6a;">Meta-Learner</div>
      </div>
      <div style="flex:1;background:#0f0f2a;border:1px solid #1a1a3a;border-radius:10px;padding:20px;text-align:center;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Isolation Forest</div>
        <div style="font-size:32px;font-weight:700;color:${result.anomaly_flag ? '#ff4757' : '#06ffa5'};margin:8px 0;">${result.anomaly_flag ? 'ANOMALY' : 'NORMAL'}</div>
        <div style="font-size:10px;color:#4a4a6a;">Anomaly Detection</div>
      </div>
    </div>

    <!-- AI Explanation -->
    <div style="background:#0f0f2a;border-left:3px solid #00d4ff;border-radius:0 10px 10px 0;padding:16px 20px;margin-bottom:24px;">
      <div style="font-size:13px;color:#00d4ff;font-weight:600;margin-bottom:6px;">AI-Generated Explanation</div>
      <div style="font-size:13px;color:#c0c0d0;line-height:1.6;">${result.human_explanation || 'No explanation available.'}</div>
    </div>

    <!-- SHAP Table -->
    <div style="margin-bottom:24px;">
      <h3 style="font-size:16px;color:#e0e0f0;margin-bottom:12px;border-bottom:1px solid #1a1a3a;padding-bottom:8px;">SHAP Feature Attribution (All 12 Features)</h3>
      <table style="width:100%;border-collapse:collapse;background:#0c0c20;border-radius:8px;overflow:hidden;">
        <thead>
          <tr style="background:#12122a;">
            <th style="padding:10px 12px;text-align:left;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Feature</th>
            <th style="padding:10px 12px;text-align:right;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">SHAP Value</th>
            <th style="padding:10px 12px;text-align:right;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Direction</th>
          </tr>
        </thead>
        <tbody>${shapRows}</tbody>
      </table>
    </div>

    <!-- Input Features Table -->
    <div style="margin-bottom:24px;">
      <h3 style="font-size:16px;color:#e0e0f0;margin-bottom:12px;border-bottom:1px solid #1a1a3a;padding-bottom:8px;">Input Loan Features</h3>
      <table style="width:100%;border-collapse:collapse;background:#0c0c20;border-radius:8px;overflow:hidden;">
        <thead>
          <tr style="background:#12122a;">
            <th style="padding:10px 12px;text-align:left;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Feature</th>
            <th style="padding:10px 12px;text-align:right;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Value</th>
          </tr>
        </thead>
        <tbody>${featureRows}</tbody>
      </table>
    </div>

    <!-- Footer -->
    <div style="border-top:1px solid #1a1a3a;padding-top:16px;display:flex;justify-content:space-between;font-size:10px;color:#4a4a6a;">
      <span>Praeventix — Pre-Delinquency Intervention Engine</span>
      <span>Models: LightGBM · GRU · Ensemble · Isolation Forest | Training: Archive + Synthetic (2M+ loans)</span>
    </div>
  </div>`;

  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);

  const opt = {
    margin: 0,
    filename: `Praeventix_Risk_Report_${new Date().toISOString().slice(0,10)}.pdf`,
    image: { type: 'jpeg', quality: 0.98 },
    html2canvas: { scale: 2, backgroundColor: '#0a0a1a', useCORS: true },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' }
  };

  try {
    await html2pdf().set(opt).from(container).save();
  } finally {
    document.body.removeChild(container);
  }
}


export default function ModelPredict() {
  const [features, setFeatures] = useState(() => {
    const init = {};
    FEATURE_FIELDS.forEach(f => { init[f.key] = f.default; });
    return init;
  });

  const [result, setResult] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activePreset, setActivePreset] = useState(null);
  const [csvFileName, setCsvFileName] = useState(null);
  const [pdfLoading, setPdfLoading] = useState(false);
  const fileInputRef = useRef(null);

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const handlePreset = (key) => {
    setActivePreset(key);
    setCsvFileName(null);
    setFeatures({ ...PRESETS[key].values });
    setResult(null);
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const data = await predictRisk(features);
      setResult(data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setResult({ error: err.message || 'Prediction failed' });
    }
    setLoading(false);
  };

  const updateFeature = (key, val) => {
    setFeatures(f => ({ ...f, [key]: Number(val) }));
    setActivePreset(null);
    setCsvFileName(null);
  };

  // ── CSV Upload Handler ──
  const handleCSVUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target.result;
      const parsed = parseCSV(text);
      if (parsed && Object.keys(parsed).length > 0) {
        // Merge parsed values over defaults
        const newFeatures = {};
        FEATURE_FIELDS.forEach(f => {
          newFeatures[f.key] = parsed[f.key] !== undefined ? parsed[f.key] : f.default;
        });
        setFeatures(newFeatures);
        setCsvFileName(file.name);
        setActivePreset(null);
        setResult(null);
      } else {
        alert('Could not parse CSV. Please ensure headers match the expected feature names.');
      }
    };
    reader.readAsText(file);
    // Reset file input so re-uploading the same file triggers onChange
    e.target.value = '';
  };

  // ── PDF Export Handler ──
  const handleExportPDF = async () => {
    if (!result || result.error) return;
    setPdfLoading(true);
    try {
      await generatePDFReport(result, features);
    } catch (err) {
      console.error('PDF generation failed:', err);
      alert('PDF generation failed. Please try again.');
    }
    setPdfLoading(false);
  };

  const riskColor = (score) => score >= 0.70 ? '#ff4757' : score >= 0.40 ? '#ff6b35' : '#06ffa5';
  const riskGlow = (score) => score >= 0.70 ? 'rgba(255,71,87,0.3)' : score >= 0.40 ? 'rgba(255,107,53,0.3)' : 'rgba(6,255,165,0.3)';

  const shapDrivers = result?.all_shap || [];
  const maxShap = Math.max(...shapDrivers.map(s => Math.abs(s.contribution || 0)), 0.01);

  return (
    <div>
      {/* Hidden file input */}
      <input type="file" accept=".csv" ref={fileInputRef} style={{ display: 'none' }} onChange={handleCSVUpload} />

      {/* Page Header */}
      <div className="page-header" style={{ animation: 'fadeSlideDown 500ms ease' }}>
        <div>
          <h1>AI Risk Predictor</h1>
          <p className="subtitle">Run all 4 trained models — LightGBM · GRU · Ensemble · Isolation Forest</p>
        </div>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          {modelInfo && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className="live-badge" style={{ background: modelInfo.models_loaded ? 'rgba(6,255,165,0.1)' : 'rgba(255,71,87,0.1)', color: modelInfo.models_loaded ? '#06ffa5' : '#ff4757' }}>
                {modelInfo.models_loaded ? '● MODELS LOADED' : '✗ MODELS OFFLINE'}
              </span>
              <span style={{ fontFamily: 'DM Mono', fontSize: 11, color: '#5a5a7a' }}>
                {modelInfo.feature_columns?.length || 12} features
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Preset Selector + CSV Upload */}
      <div style={{ padding: '16px 0 0', display: 'flex', gap: 12, animation: 'fadeSlideUp 500ms 100ms ease both' }}>
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button key={key} className={`preset-btn ${activePreset === key ? 'active' : ''}`}
            onClick={() => handlePreset(key)}
            style={{
              flex: 1, padding: '16px 20px', borderRadius: 12,
              background: activePreset === key ? 'rgba(0,212,255,0.08)' : 'var(--bg-surface-1)',
              border: `1px solid ${activePreset === key ? 'rgba(0,212,255,0.3)' : 'var(--border-default)'}`,
              cursor: 'pointer', transition: 'all 200ms', textAlign: 'left',
            }}
          >
            <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 600, fontSize: 15, color: activePreset === key ? '#00d4ff' : 'var(--text-primary)', marginBottom: 4 }}>
              {preset.name}
            </div>
            <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: 'var(--text-secondary)' }}>
              {preset.desc}
            </div>
          </button>
        ))}

        {/* CSV Upload Button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            flex: 1, padding: '16px 20px', borderRadius: 12,
            background: csvFileName ? 'rgba(124,58,237,0.1)' : 'var(--bg-surface-1)',
            border: `1px solid ${csvFileName ? 'rgba(124,58,237,0.4)' : 'var(--border-default)'}`,
            cursor: 'pointer', transition: 'all 200ms', textAlign: 'left',
          }}
        >
          <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 600, fontSize: 15, color: csvFileName ? '#7c3aed' : 'var(--text-primary)', marginBottom: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill={csvFileName ? '#7c3aed' : '#6a6a8a'}>
              <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6zm-1 2l5 5h-5V4zM6 20V4h5v7h7v9H6z"/>
            </svg>
            {csvFileName ? `📄 ${csvFileName}` : '📤 Upload CSV'}
          </div>
          <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: 'var(--text-secondary)' }}>
            {csvFileName ? 'File loaded, ready to predict' : 'Load a single customer CSV file'}
          </div>
        </button>
      </div>

      <div className="predict-layout" style={{ display: 'grid', gridTemplateColumns: '420px 1fr', gap: 24, padding: '24px 0 24px' }}>
        {/* LEFT: Feature Input Panel */}
        <div className="card" style={{ animation: 'fadeSlideRight 500ms 150ms ease both', padding: 24, maxHeight: 'calc(100vh - 220px)', overflowY: 'auto' }}>
          <div className="card-header" style={{ marginBottom: 16 }}>
            <div>
              <span className="card-title" style={{ fontSize: 16 }}>Loan Features</span>
              <span className="live-badge" style={{ marginLeft: 10 }}>12 INPUTS</span>
            </div>
          </div>

          {FEATURE_FIELDS.map((field, i) => (
            <div key={field.key} className="feature-input-row" style={{
              marginBottom: 12, padding: '10px 14px', borderRadius: 10,
              background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)',
              transition: 'all 200ms', animation: `fadeSlideUp 300ms ${Math.min(i, 6) * 40}ms ease both`
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: 'var(--text-secondary)' }}>
                  {field.label}
                </span>
                <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 13, color: '#00d4ff', fontWeight: 500 }}>
                  {field.prefix}{Number(features[field.key]).toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
              </div>
              <input
                type="range"
                className="slider-input"
                min="0"
                max={field.max}
                step={field.step}
                value={features[field.key]}
                onChange={e => updateFeature(field.key, e.target.value)}
                style={{ width: '100%' }}
              />
              <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: '#3a3a5a', marginTop: 2 }}>
                {field.desc}
              </div>
            </div>
          ))}

          <button className="btn-primary" onClick={handlePredict} disabled={loading}
            style={{ width: '100%', marginTop: 16, fontSize: 15, padding: '14px 0', position: 'sticky', bottom: 0 }}>
            {loading ? (
              <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
                <span className="spinner"></span> Running 4 Models...
              </span>
            ) : (
              '⚡ Run Ensemble Prediction'
            )}
          </button>
        </div>

        {/* RIGHT: Results Panel */}
        <div style={{ animation: 'fadeSlideLeft 500ms 200ms ease both' }}>
          {!result && !loading && (
            <div className="card" style={{ padding: 60, textAlign: 'center', minHeight: 400, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ width: 80, height: 80, borderRadius: '50%', background: 'rgba(0,212,255,0.06)', border: '2px solid rgba(0,212,255,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 24 }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="rgba(0,212,255,0.5)">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                </svg>
              </div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 600, fontSize: 20, color: 'var(--text-primary)', marginBottom: 8 }}>
                Configure & Predict
              </div>
              <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, color: 'var(--text-secondary)', maxWidth: 400, lineHeight: 1.6 }}>
                Adjust loan features using the sliders, select a preset, or <strong style={{ color: '#7c3aed' }}>upload a CSV file</strong>. Then click <strong style={{ color: '#00d4ff' }}>Run Ensemble Prediction</strong> to score risk across all 4 AI models with SHAP explainability.
              </div>
            </div>
          )}

          {loading && (
            <div className="card" style={{ padding: 60, textAlign: 'center' }}>
              <div style={{ width: 64, height: 64, border: '3px solid rgba(0,212,255,0.15)', borderTopColor: '#00d4ff', borderRadius: '50%', animation: 'spinRing 800ms linear infinite', margin: '0 auto 24px' }}></div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, color: 'var(--text-primary)', marginBottom: 8 }}>Running AI Ensemble...</div>
              <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: 'var(--text-secondary)' }}>
                LightGBM → GRU → Isolation Forest → Meta-Learner
              </div>
            </div>
          )}

          {result && !result.error && !loading && (
            <>
              {/* Ensemble Risk Score Hero */}
              <div className="card" style={{
                padding: '32px 40px', marginBottom: 20, position: 'relative', overflow: 'hidden',
                borderColor: `${riskColor(result.ensemble_prob)}30`,
              }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2, background: `linear-gradient(90deg, transparent, ${riskColor(result.ensemble_prob)}, transparent)` }}></div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 40 }}>
                  {/* Big Score */}
                  <div style={{ textAlign: 'center', minWidth: 140 }}>
                    <div style={{
                      fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: 64,
                      color: riskColor(result.ensemble_prob),
                      textShadow: `0 0 40px ${riskGlow(result.ensemble_prob)}`,
                      lineHeight: 1,
                    }}>
                      <AnimatedNumber target={result.ensemble_prob * 100} decimals={1} />
                    </div>
                    <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: 'var(--text-secondary)', marginTop: 4 }}>Ensemble Score</div>
                    <div className={`risk-badge ${result.risk_level?.toLowerCase()}`} style={{ marginTop: 8, display: 'inline-flex' }}>
                      {result.risk_level === 'HIGH' && <span className="blink-dot"></span>}
                      {result.risk_level}
                    </div>
                  </div>

                  {/* Model Gauges */}
                  <div style={{ display: 'flex', gap: 16, flex: 1, justifyContent: 'center' }}>
                    <RiskGauge value={result.lgbm_prob} label="LightGBM" color="#00d4ff" size={110} />
                    <RiskGauge value={result.gru_prob} label="GRU" color="#7c3aed" size={110} />
                    <RiskGauge value={result.ensemble_prob} label="Ensemble" color={riskColor(result.ensemble_prob)} size={110} />
                  </div>

                  {/* Anomaly Flag */}
                  <div style={{ textAlign: 'center', minWidth: 100 }}>
                    <div style={{
                      width: 56, height: 56, borderRadius: '50%', margin: '0 auto 8px',
                      background: result.anomaly_flag ? 'rgba(255,71,87,0.12)' : 'rgba(6,255,165,0.08)',
                      border: `2px solid ${result.anomaly_flag ? 'rgba(255,71,87,0.3)' : 'rgba(6,255,165,0.2)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 24
                    }}>
                      {result.anomaly_flag ? '⚠' : '✓'}
                    </div>
                    <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 12, color: result.anomaly_flag ? '#ff4757' : '#06ffa5', fontWeight: 600 }}>
                      {result.anomaly_flag ? 'ANOMALY' : 'NORMAL'}
                    </div>
                    <div style={{ fontFamily: "'DM Mono', monospace", fontSize: 10, color: 'var(--text-muted)' }}>
                      Isolation Forest
                    </div>
                  </div>
                </div>

                {/* Risk Bar */}
                <div style={{ marginTop: 24 }}>
                  <div className="risk-bar-track" style={{ maxWidth: '100%', height: 6 }}>
                    <div className="risk-bar-fill" style={{ width: `${result.ensemble_prob * 100}%` }}></div>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontFamily: "'DM Mono', monospace", fontSize: 10, color: '#5a5a7a', marginTop: 4 }}>
                    <span>0 — Low</span><span>0.40 — Medium</span><span>0.70 — High</span><span>1.0</span>
                  </div>
                </div>

                {/* Export PDF Button */}
                <div style={{ marginTop: 20, display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    onClick={handleExportPDF}
                    disabled={pdfLoading}
                    style={{
                      padding: '10px 24px', borderRadius: 10,
                      background: 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(0,212,255,0.1))',
                      border: '1px solid rgba(124,58,237,0.35)',
                      color: '#c4b5fd', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif",
                      fontSize: 13, fontWeight: 600, display: 'flex', alignItems: 'center', gap: 8,
                      transition: 'all 200ms',
                    }}
                    onMouseEnter={e => { e.target.style.background = 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))'; e.target.style.borderColor = 'rgba(124,58,237,0.6)'; }}
                    onMouseLeave={e => { e.target.style.background = 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(0,212,255,0.1))'; e.target.style.borderColor = 'rgba(124,58,237,0.35)'; }}
                  >
                    {pdfLoading ? (
                      <><span className="spinner" style={{ width: 14, height: 14 }}></span> Generating PDF...</>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="#c4b5fd">
                          <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                        </svg>
                        Export Detailed PDF Report
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Two-column: SHAP + AI Explanation */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
                {/* SHAP Drivers */}
                <div className="card" style={{ padding: 24 }}>
                  <div className="card-header" style={{ marginBottom: 16 }}>
                    <span className="card-title" style={{ fontSize: 16 }}>Risk Factor Attribution</span>
                    <span className="live-badge">SHAP</span>
                  </div>

                  <div className="shap-bar-container">
                    {shapDrivers.slice(0, 12).map((s, i) => {
                      const val = s.contribution || 0;
                      const width = (Math.abs(val) / maxShap) * 100;
                      return (
                        <div key={i} className="shap-row" style={{ animationDelay: `${i * 60}ms` }}>
                          <span className="shap-label" style={{ minWidth: 140, fontSize: 11 }}>
                            {(s.feature || '').replace(/_/g, ' ')}
                          </span>
                          <div className="shap-bar-wrapper">
                            <div className={`shap-bar ${val >= 0 ? 'positive' : 'negative'}`}
                              style={{ width: `${Math.max(width, 3)}%`, animationDelay: `${i * 60}ms` }}></div>
                            <span className="shap-value" style={{ color: val >= 0 ? '#ff4757' : '#06ffa5' }}>
                              {val >= 0 ? '+' : ''}{val.toFixed(4)}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div style={{ display: 'flex', gap: 16, marginTop: 12 }}>
                    <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: '#ff4757' }}>▶ Increases Risk</span>
                    <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: '#06ffa5' }}>◀ Decreases Risk</span>
                  </div>
                </div>

                {/* AI Explanation + Model Details */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  {/* AI Explanation */}
                  <div className="ai-card" style={{ margin: 0 }}>
                    <div className="title">AI-Generated Explanation</div>
                    <div className="text">{result.human_explanation || 'No explanation available.'}</div>
                  </div>

                  {/* Model Comparison */}
                  <div className="card" style={{ padding: 24 }}>
                    <div className="card-header" style={{ marginBottom: 16 }}>
                      <span className="card-title" style={{ fontSize: 16 }}>Model Comparison</span>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                      {[
                        { name: 'LightGBM', prob: result.lgbm_prob, color: '#00d4ff', desc: 'Gradient Boosting' },
                        { name: 'GRU Neural Net', prob: result.gru_prob, color: '#7c3aed', desc: 'Temporal Patterns' },
                        { name: 'Ensemble', prob: result.ensemble_prob, color: riskColor(result.ensemble_prob), desc: 'Meta-Learner' },
                      ].map((m, i) => (
                        <div key={i} style={{
                          padding: '12px 16px', borderRadius: 10,
                          background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.04)',
                        }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                            <div>
                              <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>{m.name}</span>
                              <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>{m.desc}</span>
                            </div>
                            <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 16, fontWeight: 600, color: m.color }}>
                              {(m.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div style={{ height: 4, borderRadius: 100, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                            <div style={{
                              height: '100%', borderRadius: 100,
                              width: `${m.prob * 100}%`, background: m.color,
                              transition: 'width 1s ease',
                              boxShadow: `0 0 12px ${m.color}60`,
                            }}></div>
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Top 3 Signals */}
                    <div style={{ marginTop: 16 }}>
                      <div className="section-label">Top Risk Signals</div>
                      {(result.shap_top3 || []).map((s, i) => (
                        <div key={i} style={{
                          marginBottom: 6, padding: '8px 12px', borderRadius: 8,
                          borderLeft: `3px solid ${s.direction === 'INCREASES_RISK' ? '#ff4757' : '#06ffa5'}`,
                          background: 'rgba(255,255,255,0.02)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                        }}>
                          <span style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 13, color: 'var(--text-secondary)' }}>
                            {(s.feature || '').replace(/_/g, ' ')}
                          </span>
                          <span style={{
                            fontFamily: "'DM Mono', monospace", fontSize: 12,
                            color: s.direction === 'INCREASES_RISK' ? '#ff4757' : '#06ffa5'
                          }}>
                            {s.direction === 'INCREASES_RISK' ? '↑ Risk' : '↓ Risk'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {result?.error && (
            <div className="card" style={{ padding: 40, textAlign: 'center' }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>⚠</div>
              <div style={{ fontFamily: "'Syne', sans-serif", fontSize: 18, color: '#ff4757', marginBottom: 8 }}>Prediction Failed</div>
              <div style={{ fontFamily: "'DM Sans', sans-serif", fontSize: 14, color: 'var(--text-secondary)' }}>
                {result.error}. Make sure the backend is running with trained models.
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
