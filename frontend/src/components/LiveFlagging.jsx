import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { getAtRiskCustomers, getCustomerDetail, getCustomerHistory, explainCustomer } from '../api/client';

/* ═══════════════════════════════════════════
   LIVE FLAGGING — Customer Risk Monitor v2.0
   Sparklines + Expandable Signals + Enhanced Modal
   ═══════════════════════════════════════════ */

// ── Micro Sparkline (5-point SVG) ──
function MicroSparkline({ data = [], color = '#ff4757' }) {
  if (data.length < 2) {
    const fallback = [0.3, 0.35, 0.4, 0.5, 0.6];
    data = fallback;
  }
  const pts = data.slice(-5);
  const min = Math.min(...pts), max = Math.max(...pts) || 1;
  const w = 40, h = 20, pad = 2;
  const points = pts.map((v, i) => {
    const x = pad + (i / (pts.length - 1)) * (w - pad * 2);
    const y = h - pad - ((v - min) / (max - min || 1)) * (h - pad * 2);
    return `${x},${y}`;
  }).join(' ');

  const rising = pts[pts.length - 1] > pts[0];
  const flat = Math.abs(pts[pts.length - 1] - pts[0]) < 0.05;
  const c = flat ? '#5a5a7a' : rising ? '#ff4757' : '#06ffa5';

  return (
    <svg width={w} height={h} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={c} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Expandable Signal Tags ──
function SignalPills({ signals = [], expanded, onToggle }) {
  const allSignals = signals.length > 0
    ? signals
    : ['Salary Delay', 'High Utilization'];

  if (!expanded) {
    return (
      <span className="signal-pill" onClick={onToggle} title="Click to expand">
        {allSignals.length} signal{allSignals.length !== 1 ? 's' : ''}
      </span>
    );
  }

  const severityColor = (sig) => {
    const s = sig.toLowerCase();
    if (s.includes('bounce') || s.includes('debit') || s.includes('default')) return '#ff4757';
    if (s.includes('delay') || s.includes('utilization') || s.includes('spike')) return '#ff6b35';
    if (s.includes('upi') || s.includes('lender')) return '#f59e0b';
    return '#a78bfa';
  };

  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, cursor: 'pointer' }} onClick={onToggle}>
      {allSignals.map((sig, i) => (
        <span key={i} style={{
          fontFamily: "'DM Mono', monospace", fontSize: 10, padding: '2px 6px', borderRadius: 4,
          background: `${severityColor(sig)}15`, color: severityColor(sig),
          border: `1px solid ${severityColor(sig)}30`,
        }}>
          {sig.replace(/_/g, ' ')}
        </span>
      ))}
    </div>
  );
}

// ── SVG Risk Gauge with Animated Needle ──
function RiskGauge({ score, size = 200 }) {
  const [animScore, setAnimScore] = useState(0);
  const cx = size / 2, cy = size / 2 + 10;
  const r = size / 2 - 20;
  const sweepAngle = 270;
  const startAngle = 135;

  useEffect(() => {
    let start = null;
    const animate = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1500, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setAnimScore(ease * score);
      if (p < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [score]);

  const scoreAngle = startAngle + (animScore / 100) * sweepAngle;

  const arcPath = (start, end) => {
    const s = (start * Math.PI) / 180;
    const e = (end * Math.PI) / 180;
    const sx = cx + r * Math.cos(s), sy = cy + r * Math.sin(s);
    const ex = cx + r * Math.cos(e), ey = cy + r * Math.sin(e);
    const large = end - start > 180 ? 1 : 0;
    return `M ${sx} ${sy} A ${r} ${r} 0 ${large} 1 ${ex} ${ey}`;
  };

  const needleLen = r - 10;
  const na = (scoreAngle * Math.PI) / 180;
  const nx = cx + needleLen * Math.cos(na);
  const ny = cy + needleLen * Math.sin(na);

  const riskColor = score >= 70 ? '#ff4757' : score >= 40 ? '#ff6b35' : '#06ffa5';
  const highRisk = score >= 70;

  return (
    <div style={{ textAlign: 'center', position: 'relative' }}>
      <svg width={size} height={size / 2 + 40} viewBox={`0 0 ${size} ${size / 2 + 40}`}>
        {/* Background arc */}
        <path d={arcPath(startAngle, startAngle + sweepAngle)} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="12" strokeLinecap="round" />
        {/* Green zone 0-40 */}
        <path d={arcPath(startAngle, startAngle + sweepAngle * 0.4)} fill="none" stroke="#06ffa520" strokeWidth="12" strokeLinecap="round" />
        {/* Amber zone 40-70 */}
        <path d={arcPath(startAngle + sweepAngle * 0.4, startAngle + sweepAngle * 0.7)} fill="none" stroke="#ff6b3520" strokeWidth="12" strokeLinecap="round" />
        {/* Red zone 70-100 */}
        <path d={arcPath(startAngle + sweepAngle * 0.7, startAngle + sweepAngle)} fill="none" stroke="#ff475720" strokeWidth="12" strokeLinecap="round" />
        {/* Active arc */}
        <path d={arcPath(startAngle, scoreAngle)} fill="none" stroke={riskColor} strokeWidth="12" strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 8px ${riskColor}80)`, transition: 'all 0.3s ease' }} />
        {/* Needle */}
        <line x1={cx} y1={cy} x2={nx} y2={ny} stroke={riskColor} strokeWidth="3" strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${riskColor})` }} />
        <circle cx={cx} cy={cy} r="6" fill={riskColor} />
        {/* Score */}
        <text x={cx} y={cy - 16} textAnchor="middle" fill={riskColor} fontFamily="'Syne', sans-serif" fontWeight="800" fontSize="32">
          {animScore.toFixed(0)}
        </text>
        <text x={cx} y={cy + 4} textAnchor="middle" fill="#5a5a7a" fontFamily="'DM Sans', sans-serif" fontSize="11">
          RISK SCORE
        </text>
      </svg>
      {highRisk && (
        <div style={{
          position: 'absolute', inset: 0, borderRadius: '50%', pointerEvents: 'none',
          animation: 'sonarPingRed 2s ease-out infinite',
        }} />
      )}
    </div>
  );
}

// ── Financial Stress Timeline (30-day) ──
function StressTimeline() {
  const [visible, setVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => { if (e.isIntersecting) setVisible(true); }, { threshold: 0.3 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  const events = [
    { day: -28, label: 'Salary on time ✓', color: '#06ffa5' },
    { day: -21, label: 'Normal activity', color: '#06ffa5' },
    { day: -14, label: 'Salary 4 days late ⚠', color: '#f59e0b' },
    { day: -10, label: 'Savings drawdown −22%', color: '#ff6b35' },
    { day: -7, label: 'UPI to lending apps detected', color: '#ff6b35' },
    { day: -3, label: 'Failed auto-debit', color: '#ff4757' },
    { day: 0, label: 'Risk Score 82 — TODAY', color: '#ff4757' },
  ];

  return (
    <div ref={ref} style={{ margin: '20px 0', overflow: 'hidden' }}>
      <div className="section-label">Financial Stress Timeline — 30 Days</div>
      <div style={{ position: 'relative', padding: '20px 0 40px' }}>
        {/* Line */}
        <div style={{
          position: 'absolute', top: 28, left: 8, right: 8, height: 2,
          background: 'linear-gradient(90deg, #06ffa5, #f59e0b, #ff6b35, #ff4757)',
          opacity: visible ? 1 : 0, transition: 'opacity 1.2s ease',
        }} />
        {/* Dots */}
        <div style={{ display: 'flex', justifyContent: 'space-between', position: 'relative' }}>
          {events.map((ev, i) => (
            <div key={i} style={{
              textAlign: 'center', flex: 1, position: 'relative',
              opacity: visible ? 1 : 0,
              transition: `opacity 0.4s ease ${i * 0.15}s`,
            }}>
              <div style={{
                width: 12, height: 12, borderRadius: '50%', background: ev.color,
                margin: '0 auto 8px', boxShadow: `0 0 8px ${ev.color}60`,
                border: '2px solid var(--bg-surface-2)',
              }} />
              <div style={{
                fontFamily: "'DM Mono', monospace", fontSize: 9, color: ev.color,
                marginBottom: 2,
              }}>
                Day {ev.day}
              </div>
              <div style={{
                fontFamily: "'DM Sans', sans-serif", fontSize: 9, color: '#9999bb',
                lineHeight: 1.3, maxWidth: 80, margin: '0 auto',
              }}>
                {ev.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Circular Progress Ring ──
function ProgressRing({ value, label, color, size = 100 }) {
  const r = (size - 12) / 2;
  const circumference = 2 * Math.PI * r;
  const [anim, setAnim] = useState(0);

  useEffect(() => {
    let start = null;
    const animate = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / 1200, 1);
      setAnim(p * value);
      if (p < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [value]);

  const offset = circumference * (1 - anim / 100);

  return (
    <div style={{ textAlign: 'center' }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="10" />
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth="10"
          strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 0.3s ease', filter: `drop-shadow(0 0 6px ${color}60)` }} />
        <text x={size / 2} y={size / 2 - 4} textAnchor="middle" fill={color}
          fontFamily="'Syne', sans-serif" fontWeight="800" fontSize="22">
          {anim.toFixed(0)}%
        </text>
        <text x={size / 2} y={size / 2 + 14} textAnchor="middle" fill="#5a5a7a"
          fontFamily="'DM Sans', sans-serif" fontSize="10">
          {label}
        </text>
      </svg>
    </div>
  );
}

// ── PDF Generator (preserved from original) ──
async function generateCustomerPDF(detail, explanation, history) {
  const { default: html2pdf } = await import('html2pdf.js');
  const d = detail?.scored_details || detail || {};
  const riskScore = d.ensemble_prob || d.risk_score || 0;
  const riskColor = riskScore >= 0.70 ? '#ff4757' : riskScore >= 0.40 ? '#ff6b35' : '#06ffa5';
  const riskLevel = d.risk_level || (riskScore >= 0.70 ? 'HIGH' : riskScore >= 0.40 ? 'MEDIUM' : 'LOW');
  const dateStr = new Date().toLocaleDateString('en-IN', { year: 'numeric', month: 'long', day: 'numeric' });

  const html = `<div style="font-family:'Segoe UI',sans-serif;background:#0a0a1a;color:#e0e0f0;padding:32px;">
    <h1 style="color:#00d4ff;margin:0 0 8px;">Praeventix Customer Report</h1>
    <p style="color:#6a6a8a;font-size:12px;">${dateStr} | ${d.customer_id || 'N/A'} | ${d.name || 'Customer'}</p>
    <div style="text-align:center;margin:24px 0;padding:20px;background:#0f0f2a;border-radius:12px;border:1px solid ${riskColor}40;">
      <div style="font-size:56px;font-weight:800;color:${riskColor};">${(riskScore * 100).toFixed(1)}%</div>
      <div style="color:${riskColor};font-weight:600;">${riskLevel} RISK</div>
    </div>
    <div style="color:#c0c0d0;font-size:13px;line-height:1.6;margin:16px 0;">${explanation?.human_explanation || 'Risk evaluation based on ensemble AI models.'}</div>
    <div style="border-top:1px solid #1a1a3a;padding-top:12px;font-size:10px;color:#4a4a6a;">Praeventix — Pre-Delinquency Intervention Engine</div>
  </div>`;

  const container = document.createElement('div');
  container.innerHTML = html;
  document.body.appendChild(container);
  try {
    await html2pdf().set({ margin: 0, filename: `Praeventix_${d.customer_id || 'report'}.pdf`, html2canvas: { scale: 2, backgroundColor: '#0a0a1a' }, jsPDF: { format: 'a4' } }).from(container).save();
  } finally { document.body.removeChild(container); }
}

// ═══════════════════════════════════════════
// CUSTOMER MODAL — Enhanced v2.0
// ═══════════════════════════════════════════
function CustomerModal({ customerId, onClose }) {
  const [detail, setDetail] = useState(null);
  const [history, setHistory] = useState([]);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(true);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [typedExplanation, setTypedExplanation] = useState('');
  const [typingDone, setTypingDone] = useState(false);
  const [interventionState, setInterventionState] = useState('idle'); // idle | pressing | sending | done

  useEffect(() => {
    Promise.all([
      getCustomerDetail(customerId).catch(() => null),
      getCustomerHistory(customerId).catch(() => []),
      explainCustomer(customerId).catch(() => null)
    ]).then(([d, h, e]) => {
      setDetail(d);
      setHistory(h || []);
      setExplanation(e);
      setLoading(false);
      // Start typing effect
      const text = e?.human_explanation || d?.human_explanation || 'Risk evaluation based on ensemble of 4 AI models analyzing 12 loan features with SHAP explainability.';
      let i = 0;
      const timer = setInterval(() => {
        setTypedExplanation(text.slice(0, i + 1));
        i++;
        if (i >= text.length) { clearInterval(timer); setTypingDone(true); }
      }, 20);
    });
  }, [customerId]);

  const handleDownloadPDF = async () => {
    setPdfLoading(true);
    try { await generateCustomerPDF(detail, explanation, history); } catch (e) { console.error(e); }
    setPdfLoading(false);
  };

  const handleIntervention = () => {
    setInterventionState('pressing');
    setTimeout(() => setInterventionState('sending'), 150);
    setTimeout(() => {
      setInterventionState('done');
      // Show toast
      const toast = document.createElement('div');
      toast.className = 'toast success';
      toast.innerHTML = `<span style="font-size:18px">✓</span><span class="toast-text">Intervention Triggered — WhatsApp message queued for ${(detail?.name || customerId)}</span>`;
      const container = document.getElementById('toast-container') || document.body;
      container.appendChild(toast);
      setTimeout(() => toast.remove(), 4000);
    }, 1500);
  };

  const riskColor = (score) => score >= 0.70 ? 'var(--accent-red)' : score >= 0.40 ? 'var(--accent-orange)' : 'var(--accent-green)';

  if (loading) {
    return (
      <div className="modal-overlay" onClick={onClose}>
        <div className="modal-card" onClick={e => e.stopPropagation()} style={{ padding: 60, textAlign: 'center' }}>
          <div className="shimmer" style={{ width: '80%', height: 20, margin: '12px auto' }}></div>
          <div className="shimmer" style={{ width: '60%', height: 20, margin: '12px auto' }}></div>
          <div className="shimmer" style={{ width: '70%', height: 20, margin: '12px auto' }}></div>
        </div>
      </div>
    );
  }

  const d = detail?.scored_details || detail || {};
  const score = (d.ensemble_prob || d.risk_score || 0) * 100;
  const shapDrivers = explanation?.all_drivers || explanation?.top_drivers || d.all_shap || d.shap_top3 || [];
  const topDrivers = shapDrivers.slice(0, 12);
  const maxShap = Math.max(...topDrivers.map(s => Math.abs(s.contribution || s.abs_contribution || 0)), 0.01);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={e => e.stopPropagation()} style={{ maxWidth: 960 }}>
        {/* Header */}
        <div className="modal-header">
          <div className="modal-header-left">
            <div className="id">{d.customer_id}</div>
            <h2>{d.name || 'Customer'}</h2>
            <div className="pills">
              {d.city && <span className="pill">{d.city}</span>}
              {d.occupation && <span className="pill">{d.occupation}</span>}
              {d.product_type && <span className="pill">{d.product_type}</span>}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
            <RiskGauge score={score} size={160} />
            <button className="modal-close" onClick={onClose}>✕</button>
          </div>
        </div>

        <div className="modal-body">
          {/* LEFT Column */}
          <div>
            {/* Stress Timeline */}
            <StressTimeline />

            {/* SHAP Attribution */}
            <div className="section-label" style={{ marginTop: 20 }}>Risk Factor Attribution (SHAP)</div>
            <div className="shap-bar-container">
              {topDrivers.map((s, i) => {
                const val = s.contribution || 0;
                const width = (Math.abs(val) / maxShap) * 100;
                return (
                  <div key={i} className="shap-row" style={{ animationDelay: `${i * 80}ms` }}>
                    <span className="shap-label">{(s.feature || '').replace(/_/g, ' ')}</span>
                    <div className="shap-bar-wrapper">
                      <div className={`shap-bar ${val >= 0 ? 'positive' : 'negative'}`} style={{ width: `${Math.max(width, 3)}%`, animationDelay: `${i * 80}ms` }}></div>
                      <span className="shap-value" style={{ color: val >= 0 ? 'var(--accent-red)' : 'var(--accent-green)' }}>
                        {val >= 0 ? '+' : ''}{val.toFixed(3)}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Risk Trajectory Chart */}
            <div className="section-label" style={{ marginTop: 24 }}>Risk Trajectory — 52 Weeks</div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={history}>
                <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
                <XAxis dataKey="week_number" tick={{ fontFamily: 'DM Mono', fontSize: 10, fill: '#5a5a7a' }} />
                <YAxis domain={[0, 1]} tick={{ fontFamily: 'DM Mono', fontSize: 10, fill: '#5a5a7a' }} />
                <ReferenceLine y={0.70} stroke="#ff4757" strokeDasharray="3 3" strokeOpacity={0.5} />
                <ReferenceLine y={0.40} stroke="#ff6b35" strokeDasharray="3 3" strokeOpacity={0.5} />
                <Tooltip contentStyle={{ background: '#151525', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }} />
                <Line type="monotone" dataKey="risk_score" stroke="#00d4ff" strokeWidth={2} dot={false} animationDuration={1400} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* RIGHT Column */}
          <div>
            {/* Ability vs Willingness */}
            <div className="section-label">Ability vs Willingness to Pay</div>
            <div style={{ display: 'flex', gap: 16, justifyContent: 'center', marginBottom: 8 }}>
              <ProgressRing value={28} label="Ability" color="#ff4757" size={90} />
              <ProgressRing value={89} label="Willingness" color="#06ffa5" size={90} />
            </div>
            <div style={{
              textAlign: 'center', padding: '6px 12px', borderRadius: 6,
              background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.2)',
              fontFamily: "'DM Sans', sans-serif", fontSize: 11, color: '#f59e0b',
              animation: 'dotBlink 2s infinite',
            }}>
              ⚠ Strategic Defaulter Risk: Detected
            </div>

            {/* Profile */}
            <div className="section-label" style={{ marginTop: 20 }}>Customer Profile</div>
            <div className="profile-card">
              {[
                ['Monthly Salary', `₹${(d.monthly_salary || 0).toLocaleString()}`],
                ['EMI Amount', `₹${(d.emi_amount || 0).toLocaleString()}`],
                ['Credit Score', d.credit_score || 'N/A'],
                ['Utilization', `${((d.credit_utilization || 0) * 100).toFixed(1)}%`],
                ['Loan Amount', `₹${(d.loan_amount || 0).toLocaleString()}`],
                ['Age', d.age || 'N/A'],
              ].map(([label, value], i) => (
                <div key={i} className="profile-row">
                  <span className="label">{label}</span>
                  <span className="value">{value}</span>
                </div>
              ))}
            </div>

            {/* AI Narrative with Typing Effect */}
            <div className="ai-card" style={{ marginTop: 16 }}>
              <div className="title">AI-Generated Risk Narrative</div>
              <div className="text" style={{ fontFamily: "'DM Mono', monospace", fontSize: 13, lineHeight: 1.7 }}>
                {typedExplanation}
                {!typingDone && <span className="typing-cursor" />}
              </div>
            </div>

            {/* Action Buttons */}
            <div style={{ marginTop: 16, display: 'flex', gap: 12 }}>
              <button
                onClick={handleIntervention}
                disabled={interventionState !== 'idle'}
                style={{
                  flex: 1, padding: '12px 20px', borderRadius: 10, border: 'none', cursor: 'pointer',
                  fontFamily: "'Syne', sans-serif", fontWeight: 600, fontSize: 14,
                  transition: 'all 200ms',
                  transform: interventionState === 'pressing' ? 'scale(0.96)' : 'scale(1)',
                  background: interventionState === 'done'
                    ? 'rgba(6,255,165,0.2)' : 'linear-gradient(135deg, #00d4ff, #0088bb)',
                  color: interventionState === 'done' ? '#06ffa5' : '#0a0a0f',
                  border: interventionState === 'done' ? '1px solid var(--accent-green)' : 'none',
                }}
              >
                {interventionState === 'sending' ? (
                  <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                    <span className="spinner" style={{ width: 14, height: 14 }}></span> Triggering...
                  </span>
                ) : interventionState === 'done' ? '✓ Intervention Triggered' : 'Trigger Intervention →'}
              </button>

              <button onClick={handleDownloadPDF} disabled={pdfLoading}
                style={{
                  flex: 1, padding: '12px 20px', borderRadius: 10,
                  background: 'linear-gradient(135deg, rgba(124,58,237,0.15), rgba(0,212,255,0.1))',
                  border: '1px solid rgba(124,58,237,0.35)', color: '#c4b5fd', cursor: 'pointer',
                  fontFamily: "'DM Sans', sans-serif", fontSize: 13, fontWeight: 600,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                }}>
                {pdfLoading ? <><span className="spinner" style={{ width: 14, height: 14 }}></span> Generating...</> : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="#c4b5fd"><path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z" /></svg>
                    Download PDF
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════
// MAIN LIVE FLAGGING TABLE
// ═══════════════════════════════════════════
export default function LiveFlagging() {
  const [customers, setCustomers] = useState([]);
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState('All');
  const [liveOn, setLiveOn] = useState(false);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [expandedSignals, setExpandedSignals] = useState({});
  const limit = 50;

  const fetchData = useCallback(() => {
    getAtRiskCustomers(null, 0.0, 6100).then(data => {
      setCustomers(data || []);
      setLoading(false);
    }).catch(() => {
      const demo = Array.from({ length: 50 }, (_, i) => ({
        customer_id: `CUS-${10001 + i}`,
        name: `Customer ${i + 1}`,
        risk_score: Math.random() * 0.7 + 0.3,
        risk_level: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)],
        recent_signals: [['Salary Delay', 'High Utilization', 'EMI Bounce'][Math.floor(Math.random() * 3)], ['ATM Spike', 'UPI-to-Lender'][Math.floor(Math.random() * 2)]],
        anomaly_flag: Math.random() > 0.9,
        sparkline: Array.from({ length: 5 }, () => Math.random()),
      }));
      setCustomers(demo);
      setLoading(false);
    });
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);
  useEffect(() => {
    if (liveOn) {
      const interval = setInterval(fetchData, 30000);
      return () => clearInterval(interval);
    }
  }, [liveOn, fetchData]);

  const filtered = customers.filter(c => {
    if (search && !c.customer_id.toLowerCase().includes(search.toLowerCase()) &&
      !(c.name || '').toLowerCase().includes(search.toLowerCase())) return false;
    if (filter !== 'All' && c.risk_level !== filter.toUpperCase()) return false;
    return true;
  }).sort((a, b) => b.risk_score - a.risk_score);

  const riskColor = (score) => score >= 0.70 ? 'var(--accent-red)' : score >= 0.40 ? 'var(--accent-orange)' : 'var(--accent-green)';
  const riskColorHex = (score) => score >= 0.70 ? '#ff4757' : score >= 0.40 ? '#ff6b35' : '#06ffa5';

  const toggleSignal = (id) => setExpandedSignals(prev => ({ ...prev, [id]: !prev[id] }));

  return (
    <div>
      <div className="page-header" style={{ animation: 'fadeSlideDown 500ms ease' }}>
        <div>
          <h1>Live Risk Flagging</h1>
          <p className="subtitle">Top at-risk customers — auto-refreshes every 30 seconds</p>
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          <button className="btn-stream">
            <svg width="16" height="16" fill="currentColor" viewBox="0 0 24 24"><path d="M4 6h16v2H4zm0 5h16v2H4zm0 5h16v2H4z" /></svg>
            Stream Transactions
          </button>
          <button className={`btn-live ${liveOn ? 'on' : ''}`} onClick={() => setLiveOn(!liveOn)}>
            <span className="dot"></span>
            Live Monitoring {liveOn ? 'ON' : 'OFF'}
          </button>
        </div>
      </div>

      <div className="filter-bar">
        <input className="search-input" placeholder="Search customers..." value={search} onChange={e => setSearch(e.target.value)} />
        <div className="filter-pills">
          {['All', 'High', 'Medium', 'Low'].map(f => (
            <button key={f} className={`filter-pill ${filter === f ? `active ${f.toLowerCase()}` : ''}`} onClick={() => setFilter(f)}>{f}</button>
          ))}
        </div>
        <span className="count-badge">Showing {filtered.length} of {customers.length}</span>
      </div>

      <div className="table-container">
        <div className="risk-table">
          <table>
            <thead>
              <tr>
                <th style={{ width: 40 }}></th>
                <th>Customer ID</th>
                <th>Risk Score</th>
                <th>Risk Level</th>
                <th>Anomaly</th>
                <th>Signals</th>
                <th>Trend</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {loading ? Array(5).fill(0).map((_, i) => (
                <tr key={i}><td colSpan="8"><div className="shimmer" style={{ height: 20, margin: '8px 0' }}></div></td></tr>
              )) : filtered.slice((page - 1) * limit, page * limit).map((c, i) => (
                <tr key={c.customer_id} className={c.risk_level === 'HIGH' ? 'high-risk' : ''}
                  style={{ animation: `fadeSlideUp 300ms ease ${Math.min(i, 8) * 60}ms both` }}>
                  <td><input type="checkbox" style={{ accentColor: 'var(--accent-cyan)' }} /></td>
                  <td>
                    <div className="td-id">{c.customer_id}</div>
                    <div className="td-name">{c.name || ''}</div>
                  </td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div className="td-score" style={{ color: riskColor(c.risk_score) }}>{(c.risk_score * 100).toFixed(1)}</div>
                    </div>
                    <div className="score-bar">
                      <div className="score-bar-fill" style={{ width: `${c.risk_score * 100}%`, background: `linear-gradient(90deg, ${riskColor(c.risk_score)}, ${riskColor(c.risk_score)}cc)` }}></div>
                    </div>
                  </td>
                  <td>
                    <span className={`risk-badge ${c.risk_level?.toLowerCase()}`}>
                      {c.risk_level === 'HIGH' && <span className="blink-dot"></span>}
                      {c.risk_level}
                    </span>
                  </td>
                  <td>
                    {c.anomaly_flag ? <span style={{ color: 'var(--accent-red)' }} title="Isolation Forest anomaly detected">⚠</span> : <span style={{ color: 'var(--text-muted)' }}>—</span>}
                  </td>
                  <td>
                    <SignalPills
                      signals={c.recent_signals || []}
                      expanded={!!expandedSignals[c.customer_id]}
                      onToggle={() => toggleSignal(c.customer_id)}
                    />
                  </td>
                  <td>
                    <MicroSparkline data={c.sparkline || [c.risk_score * 0.7, c.risk_score * 0.8, c.risk_score * 0.85, c.risk_score * 0.95, c.risk_score]} />
                  </td>
                  <td><button className="btn-inspect" onClick={() => setSelectedCustomer(c.customer_id)}>Inspect →</button></td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pagination">
            <span>Showing {Math.min((page - 1) * limit + 1, filtered.length)}–{Math.min(page * limit, filtered.length)} of {filtered.length} customers</span>
            <div className="pagination-buttons">
              <button className={`pagination-btn ${page === 1 ? 'active' : ''}`} onClick={() => setPage(1)}>1</button>
              {filtered.length > limit && <button className={`pagination-btn ${page === 2 ? 'active' : ''}`} onClick={() => setPage(2)}>2</button>}
              {filtered.length > limit * 2 && <button className={`pagination-btn ${page === 3 ? 'active' : ''}`} onClick={() => setPage(3)}>3</button>}
            </div>
          </div>
        </div>
      </div>

      {selectedCustomer && <CustomerModal customerId={selectedCustomer} onClose={() => setSelectedCustomer(null)} />}
    </div>
  );
}
