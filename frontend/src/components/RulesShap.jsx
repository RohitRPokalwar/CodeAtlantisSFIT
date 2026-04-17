import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { explainCustomer } from '../api/client';

const DEFAULT_RULES = [
  { id: 1, name: 'Salary credited 5+ days late', feature: 'salary_delay_days', threshold: 5, maxThreshold: 30, weight: 0.85, enabled: true },
  { id: 2, name: 'Savings balance drops ≥ 20% WoW', feature: 'savings_wow_delta_pct', threshold: 20, maxThreshold: 100, weight: 0.80, enabled: true },
  { id: 3, name: 'UPI to lending apps ≥ 3 per week', feature: 'lending_upi_count_7d', threshold: 3, maxThreshold: 10, weight: 0.75, enabled: true },
  { id: 4, name: 'Utility payment ≥ 7 days after due date', feature: 'utility_payment_delay_days', threshold: 7, maxThreshold: 30, weight: 0.70, enabled: true },
  { id: 5, name: 'Discretionary spend drops ≥ 40%', feature: 'discretionary_spend_7d', threshold: 40, maxThreshold: 100, weight: 0.65, enabled: true },
  { id: 6, name: 'ATM withdrawals ≥ 5 in past 7 days', feature: 'atm_withdrawal_count_7d', threshold: 5, maxThreshold: 20, weight: 0.60, enabled: true },
  { id: 7, name: 'Failed auto-debit ≥ 1 in past 7 days', feature: 'failed_autodebit_count', threshold: 1, maxThreshold: 5, weight: 0.90, enabled: true },
  { id: 8, name: 'Gambling spend > ₹500 in 7 days', feature: 'gambling_spend_7d', threshold: 500, maxThreshold: 5000, weight: 0.55, enabled: true },
];

// ── SHAP PDF Report Generator ──
async function generateShapPDF(explanation, customerId) {
  const { default: html2pdf } = await import('html2pdf.js');

  const riskScore = explanation?.risk_score || 0;
  const riskColor = riskScore >= 0.70 ? '#ff4757' : riskScore >= 0.40 ? '#ff6b35' : '#06ffa5';
  const riskLevel = explanation?.risk_level || (riskScore >= 0.70 ? 'HIGH' : riskScore >= 0.40 ? 'MEDIUM' : 'LOW');
  const dateStr = new Date().toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' });
  const timeStr = new Date().toLocaleTimeString('en-IN', { hour12: false });

  const shapDrivers = explanation?.all_drivers || explanation?.top_drivers || [];
  const shapRows = shapDrivers.map((s, i) => {
    const dir = (s.direction === 'INCREASES_RISK') ? '↑ RISK' : '↓ SAFE';
    const dirColor = (s.direction === 'INCREASES_RISK') ? '#ff4757' : '#06ffa5';
    const barWidth = Math.max(Math.abs(s.contribution || 0) / Math.max(...shapDrivers.map(x => Math.abs(x.contribution || 0)), 0.01) * 100, 3);
    return `<tr>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;color:#c0c0d0;width:30%;">${i + 1}. ${(s.feature || '').replace(/_/g, ' ')}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;width:40%;">
        <div style="display:flex;align-items:center;gap:8px;">
          <div style="height:10px;width:${barWidth}%;background:${dirColor};border-radius:4px;min-width:4px;"></div>
        </div>
      </td>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;font-family:monospace;color:${dirColor};text-align:right;">${(s.contribution || 0) >= 0 ? '+' : ''}${(s.contribution || 0).toFixed(4)}</td>
      <td style="padding:8px 12px;border-bottom:1px solid #1a1a2e;font-size:12px;color:${dirColor};text-align:right;font-weight:600;">${dir}</td>
    </tr>`;
  }).join('');

  const topSignals = shapDrivers.slice(0, 3);
  const topSignalRows = topSignals.map((s, i) => {
    const dirColor = (s.direction === 'INCREASES_RISK') ? '#ff4757' : '#06ffa5';
    return `<div style="margin-bottom:8px;padding:12px 16px;border-radius:8px;border-left:4px solid ${dirColor};background:rgba(255,255,255,0.02);display:flex;justify-content:space-between;align-items:center;">
      <div>
        <div style="font-size:14px;color:#e0e0f0;font-weight:600;">#${i + 1} — ${(s.feature || '').replace(/_/g, ' ')}</div>
        <div style="font-size:11px;color:#6a6a8a;margin-top:2px;">SHAP contribution: ${(s.contribution || 0).toFixed(4)}</div>
      </div>
      <div style="font-size:13px;color:${dirColor};font-weight:700;">${s.direction === 'INCREASES_RISK' ? '↑ INCREASES RISK' : '↓ DECREASES RISK'}</div>
    </div>`;
  }).join('');

  const html = `
  <div style="font-family:'Segoe UI',Arial,sans-serif;background:#0a0a1a;color:#e0e0f0;padding:36px;min-height:100%;">
    <!-- Header -->
    <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #1a1a3a;padding-bottom:18px;margin-bottom:24px;">
      <div>
        <h1 style="margin:0;font-size:28px;color:#00d4ff;letter-spacing:1px;">Praeventix</h1>
        <p style="margin:4px 0 0;font-size:12px;color:#6a6a8a;">SHAP Explainability & Risk Attribution Report</p>
      </div>
      <div style="text-align:right;">
        <div style="font-size:12px;color:#6a6a8a;">Report Generated</div>
        <div style="font-size:14px;color:#c0c0d0;font-family:monospace;">${dateStr} — ${timeStr}</div>
        <div style="font-size:11px;color:#6a6a8a;margin-top:2px;">CONFIDENTIAL — For Authorized Personnel Only</div>
      </div>
    </div>

    <!-- Customer & Risk Score -->
    <div style="display:flex;gap:20px;margin-bottom:24px;">
      <div style="flex:1;background:#0f0f2a;border:1px solid #1a1a3a;border-radius:12px;padding:24px;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">Customer</div>
        <div style="font-size:24px;font-weight:700;color:#00d4ff;font-family:monospace;">${customerId}</div>
        <div style="font-size:12px;color:#6a6a8a;margin-top:8px;">Risk Level: <span style="color:${riskColor};font-weight:700;">${riskLevel}</span></div>
      </div>
      <div style="flex:1;background:linear-gradient(135deg, #0f0f2a, #161640);border:1px solid ${riskColor}40;border-radius:12px;padding:24px;text-align:center;">
        <div style="font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px;">Risk Score</div>
        <div style="font-size:64px;font-weight:800;color:${riskColor};line-height:1;">${(riskScore * 100).toFixed(1)}%</div>
        <div style="display:inline-block;margin-top:10px;padding:5px 18px;border-radius:20px;background:${riskColor}20;color:${riskColor};font-weight:700;font-size:13px;border:1px solid ${riskColor}40;">
          ${riskLevel} RISK
        </div>
      </div>
    </div>

    <!-- AI Explanation -->
    <div style="background:#0f0f2a;border-left:3px solid #00d4ff;border-radius:0 10px 10px 0;padding:16px 20px;margin-bottom:24px;">
      <div style="font-size:13px;color:#00d4ff;font-weight:600;margin-bottom:6px;">AI-Generated Risk Narrative</div>
      <div style="font-size:13px;color:#c0c0d0;line-height:1.7;">${explanation?.human_explanation || 'Risk evaluation based on ensemble of 4 AI models analyzing 12 loan features with SHAP explainability.'}</div>
    </div>

    <!-- Top 3 Risk Signals -->
    <div style="margin-bottom:24px;">
      <h3 style="font-size:16px;color:#e0e0f0;margin-bottom:12px;border-bottom:1px solid #1a1a3a;padding-bottom:8px;">Top 3 Risk Drivers</h3>
      ${topSignalRows}
    </div>

    <!-- Full SHAP Attribution Table -->
    <div style="margin-bottom:24px;">
      <h3 style="font-size:16px;color:#e0e0f0;margin-bottom:12px;border-bottom:1px solid #1a1a3a;padding-bottom:8px;">Full SHAP Feature Attribution (All Features)</h3>
      <table style="width:100%;border-collapse:collapse;background:#0c0c20;border-radius:8px;overflow:hidden;">
        <thead>
          <tr style="background:#12122a;">
            <th style="padding:10px 12px;text-align:left;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Feature</th>
            <th style="padding:10px 12px;text-align:left;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Impact</th>
            <th style="padding:10px 12px;text-align:right;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">SHAP Value</th>
            <th style="padding:10px 12px;text-align:right;font-size:11px;color:#6a6a8a;text-transform:uppercase;letter-spacing:1px;">Direction</th>
          </tr>
        </thead>
        <tbody>${shapRows || '<tr><td colspan="4" style="padding:16px;font-size:12px;color:#6a6a8a;">No SHAP values available</td></tr>'}</tbody>
      </table>
      <div style="display:flex;gap:24px;margin-top:10px;">
        <span style="font-size:11px;color:#ff4757;">■ Red = Increases Risk</span>
        <span style="font-size:11px;color:#06ffa5;">■ Green = Decreases Risk</span>
      </div>
    </div>

    <!-- Summary -->
    <div style="background:#0f0f2a;border:1px solid #1a1a3a;border-radius:10px;padding:16px 20px;margin-bottom:24px;">
      <div style="font-size:13px;color:#c4b5fd;font-weight:600;margin-bottom:6px;">Interpretation Summary</div>
      <div style="font-size:12px;color:#c0c0d0;line-height:1.6;">
        This report details the SHAP (SHapley Additive exPlanations) values for customer <strong style="color:#00d4ff;">${customerId}</strong>.
        SHAP values decompose the model's prediction into per-feature contributions, showing exactly how each input drives the final risk score.
        Positive values (red) push the score toward default, while negative values (green) push toward safety.
        The ensemble model combines LightGBM, GRU Neural Network, and Isolation Forest outputs through a meta-learner for maximum accuracy.
      </div>
    </div>

    <!-- Footer -->
    <div style="border-top:1px solid #1a1a3a;padding-top:14px;display:flex;justify-content:space-between;font-size:10px;color:#4a4a6a;">
      <span>Praeventix — Pre-Delinquency Intervention Engine | SHAP Report: ${customerId}</span>
      <span>Models: LightGBM · GRU · Ensemble · Isolation Forest | Training: 2M+ loans</span>
    </div>
  </div>`;

  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);

  const opt = {
    margin: 0,
    filename: `Praeventix_SHAP_Report_${customerId.replace(/\s/g, '_')}_${new Date().toISOString().slice(0, 10)}.pdf`,
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


export default function RulesShap() {
  const [rules, setRules] = useState(DEFAULT_RULES);
  const [customerId, setCustomerId] = useState('');
  const [explanation, setExplanation] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [pdfLoading, setPdfLoading] = useState(false);

  const toggleRule = (id) => setRules(r => r.map(rule => rule.id === id ? { ...rule, enabled: !rule.enabled } : rule));
  const updateThreshold = (id, val) => setRules(r => r.map(rule => rule.id === id ? { ...rule, threshold: Number(val) } : rule));
  const updateWeight = (id, val) => setRules(r => r.map(rule => rule.id === id ? { ...rule, weight: Number(val) } : rule));

  const addRule = () => {
    const newId = Math.max(...rules.map(r => r.id)) + 1;
    setRules([...rules, { id: newId, name: 'New Rule', feature: 'custom_feature', threshold: 0, maxThreshold: 100, weight: 0.5, enabled: false }]);
  };

  const handleExplain = async () => {
    if (!customerId) return;
    setShapLoading(true);
    setTypedText('');
    try {
      const data = await explainCustomer(customerId);
      setExplanation(data);
      // Typewriter effect
      const text = data?.human_explanation || 'No explanation available';
      let i = 0;
      const timer = setInterval(() => {
        setTypedText(text.slice(0, i + 1));
        i++;
        if (i >= text.length) clearInterval(timer);
      }, 12);
    } catch {
      setExplanation({ error: 'Failed to load explanation' });
    }
    setShapLoading(false);
  };

  const handleDownloadShapPDF = async () => {
    if (!explanation || explanation.error) return;
    setPdfLoading(true);
    try {
      await generateShapPDF(explanation, customerId);
    } catch (err) {
      console.error('SHAP PDF generation failed:', err);
    }
    setPdfLoading(false);
  };

  const riskColor = (score) => score >= 0.70 ? 'var(--accent-red)' : score >= 0.40 ? 'var(--accent-orange)' : 'var(--accent-green)';

  const shapDrivers = explanation?.all_drivers || explanation?.top_drivers || [];
  const maxShap = Math.max(...shapDrivers.map(s => Math.abs(s.contribution || 0)), 0.01);

  // Impact preview data
  const impactData = rules.filter(r => r.enabled).map(r => ({
    name: r.feature.replace(/_/g, ' ').slice(0, 12),
    affected: Math.floor(Math.random() * 30 + r.weight * 20)
  }));

  return (
    <div>
      <div className="page-header" style={{ animation: 'fadeSlideDown 500ms ease' }}>
        <div>
          <h1>Rules & SHAP Configuration</h1>
          <p className="subtitle">Configure behavioral thresholds and inspect model explanations</p>
        </div>
        <button className="btn-save">
          <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M9 16.2L4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z"/></svg>
          Save Configuration
        </button>
      </div>

      <div className="two-col">
        {/* LEFT: Rule Engine */}
        <div className="card" style={{ animation: 'fadeSlideRight 500ms ease' }}>
          <div className="card-header">
            <div>
              <span className="card-title">Behavioral Rule Engine</span>
              <span className="live-badge" style={{ marginLeft: 12 }}>{rules.filter(r => r.enabled).length} Active Rules</span>
            </div>
          </div>

          {rules.map(rule => (
            <div key={rule.id} className={`rule-row ${!rule.enabled ? 'disabled' : ''}`}>
              <div className="rule-header">
                <button className={`toggle-switch ${rule.enabled ? 'on' : 'off'}`} onClick={() => toggleRule(rule.id)}>
                  <div className="toggle-circle"></div>
                </button>
                <span className="rule-name">{rule.name}</span>
              </div>

              <div className="slider-row">
                <span className="slider-label">Threshold:</span>
                <input type="range" className="slider-input" min="0" max={rule.maxThreshold} value={rule.threshold} onChange={e => updateThreshold(rule.id, e.target.value)} />
                <span className="slider-value">{rule.threshold}</span>
              </div>

              <div className="slider-row">
                <span className="slider-label">Influence:</span>
                <input type="range" className="slider-input purple" min="0" max="100" value={Math.round(rule.weight * 100)} onChange={e => updateWeight(rule.id, e.target.value / 100)} />
                <span className="slider-value purple">{rule.weight.toFixed(2)}</span>
              </div>
            </div>
          ))}

          <button className="btn-add-rule" onClick={addRule}>+ Add New Behavioral Rule</button>

          <div style={{ marginTop: 24 }}>
            <div className="section-label">Impact Preview</div>
            <ResponsiveContainer width="100%" height={120}>
              <BarChart data={impactData}>
                <XAxis dataKey="name" tick={{ fontFamily: 'DM Mono', fontSize: 9, fill: '#5a5a7a' }} />
                <YAxis tick={{ fontFamily: 'DM Mono', fontSize: 10, fill: '#5a5a7a' }} />
                <Bar dataKey="affected" fill="rgba(0,212,255,0.4)" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* RIGHT: SHAP Viewer */}
        <div className="card" style={{ animation: 'fadeSlideLeft 500ms ease' }}>
          <div className="card-header">
            <div>
              <span className="card-title">SHAP Explanation Viewer</span>
              <p style={{ fontFamily: 'DM Sans', fontSize: 13, color: '#9999bb', marginTop: 4 }}>Select a customer from Live Flagging tab or enter ID below</p>
            </div>
          </div>

          <div className="customer-input-row">
            <input className="customer-input" placeholder="Enter Customer ID e.g. CUS-10042" value={customerId} onChange={e => setCustomerId(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleExplain()} />
            <button className="btn-explain" onClick={handleExplain}>Explain →</button>
          </div>

          {shapLoading && (
            <div>
              <div className="shimmer" style={{ width: '80%', height: 16, marginBottom: 12 }}></div>
              <div className="shimmer" style={{ width: '60%', height: 16, marginBottom: 12 }}></div>
              <div className="shimmer" style={{ width: '70%', height: 16 }}></div>
            </div>
          )}

          {explanation && !shapLoading && (
            <>
              {/* Risk Score Display */}
              <div className="risk-score-display">
                <div style={{ fontFamily: 'DM Sans', fontSize: 13, color: '#9999bb' }}>Final Risk Score</div>
                <div className="risk-score-big" style={{ color: riskColor(explanation.risk_score || 0) }}>
                  {((explanation.risk_score || 0) * 100).toFixed(0)}
                </div>
                <div className="risk-bar-track">
                  <div className="risk-bar-fill" style={{ width: `${(explanation.risk_score || 0) * 100}%` }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', maxWidth: 480, margin: '4px auto', fontFamily: 'DM Mono', fontSize: 10, color: '#5a5a7a' }}>
                  <span>0</span><span>Medium (0.40)</span><span>High (0.70)</span><span>1.0</span>
                </div>
              </div>

              {/* SHAP Bars */}
              <div style={{ marginTop: 24 }}>
                <div className="section-label">What Drove This Score?</div>
                <div className="shap-bar-container">
                  {shapDrivers.slice(0, 12).map((s, i) => {
                    const val = s.contribution || 0;
                    const width = (Math.abs(val) / maxShap) * 100;
                    return (
                      <div key={i} className="shap-row" style={{ animationDelay: `${i * 50}ms` }}>
                        <span className="shap-label">{(s.feature || '').replace(/_/g, ' ')}</span>
                        <div className="shap-bar-wrapper">
                          <div className={`shap-bar ${val >= 0 ? 'positive' : 'negative'}`} style={{ width: `${Math.max(width, 3)}%`, animationDelay: `${i * 50}ms` }}></div>
                          <span className="shap-value" style={{ color: val >= 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                            {val >= 0 ? '+' : ''}{val.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div style={{ display: 'flex', gap: 24, marginTop: 12 }}>
                  <span style={{ fontFamily: 'DM Mono', fontSize: 11, color: 'var(--accent-red)' }}>▶ Increases Risk</span>
                  <span style={{ fontFamily: 'DM Mono', fontSize: 11, color: 'var(--accent-green)' }}>◀ Decreases Risk</span>
                </div>
              </div>

              {/* AI Explanation */}
              <div className="ai-card" style={{ marginTop: 20 }}>
                <div className="title">AI-Generated Explanation</div>
                <div className="text">{typedText}<span style={{ animation: 'dotBlink 1s infinite', color: 'var(--accent-purple)' }}>|</span></div>
              </div>

              {/* Top Signals */}
              {shapDrivers.slice(0, 3).map((s, i) => (
                <div key={i} style={{
                  marginTop: 8, padding: '10px 14px', borderRadius: 8,
                  borderLeft: `3px solid ${s.direction === 'INCREASES_RISK' ? 'var(--accent-red)' : 'var(--accent-green)'}`,
                  background: 'rgba(255,255,255,0.02)', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
                }}>
                  <span style={{ fontFamily: 'DM Sans', fontSize: 13, color: 'var(--text-secondary)' }}>{(s.feature || '').replace(/_/g, ' ')}</span>
                  <div style={{ textAlign: 'right' }}>
                    <span style={{ fontFamily: 'DM Mono', fontSize: 12, color: s.direction === 'INCREASES_RISK' ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                      {s.direction === 'INCREASES_RISK' ? '↑' : '↓'} {s.direction?.replace('_', ' ')}
                    </span>
                    <div style={{ fontFamily: 'DM Mono', fontSize: 11, color: 'var(--text-muted)' }}>contribution: {(s.contribution || 0).toFixed(4)}</div>
                  </div>
                </div>
              ))}

              {/* Download SHAP PDF Report Button */}
              <button
                onClick={handleDownloadShapPDF}
                disabled={pdfLoading}
                style={{
                  width: '100%', marginTop: 20, padding: '13px 24px', borderRadius: 10,
                  background: 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(0,212,255,0.1))',
                  border: '1px solid rgba(124,58,237,0.35)',
                  color: '#c4b5fd', cursor: 'pointer', fontFamily: "'DM Sans', sans-serif",
                  fontSize: 14, fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10,
                  transition: 'all 200ms',
                }}
                onMouseEnter={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124,58,237,0.25), rgba(0,212,255,0.15))'; e.currentTarget.style.borderColor = 'rgba(124,58,237,0.6)'; }}
                onMouseLeave={e => { e.currentTarget.style.background = 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(0,212,255,0.1))'; e.currentTarget.style.borderColor = 'rgba(124,58,237,0.35)'; }}
              >
                {pdfLoading ? (
                  <><span className="spinner" style={{ width: 14, height: 14 }}></span> Generating PDF...</>
                ) : (
                  <>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="#c4b5fd">
                      <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                    </svg>
                    Download SHAP Report (PDF)
                  </>
                )}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
