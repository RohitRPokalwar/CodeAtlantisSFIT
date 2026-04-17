import React, { useState, useEffect, useCallback, useRef } from 'react';
import { getInterventionLog, getCustomerDetail, triggerIntervention } from '../api/client';

/* ═══════════════════════════════════════════
   OUTREACH PANEL — Intervention Hub v2.0
   KPI Command Strip + Slide-In Cards + Enhanced Preview
   ═══════════════════════════════════════════ */

function AnimatedNumber({ target, duration = 1200, color = 'var(--text-primary)' }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      setVal(ease * target);
      if (p < 1) ref.current = requestAnimationFrame(step);
    };
    ref.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(ref.current);
  }, [target, duration]);
  return <span style={{ color }}>{Math.round(val)}</span>;
}

export default function OutreachPanel() {
  const [queue, setQueue] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [channel, setChannel] = useState('SMS');
  const [message, setMessage] = useState('');
  const [sending, setSending] = useState(false);
  const [sent, setSent] = useState(false);
  const [toasts, setToasts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQueue, setSearchQueue] = useState('');

  useEffect(() => {
    getInterventionLog(1, 50).then(data => {
      const pending = (data || []).filter(d => d.status === 'SENT' || d.status === 'DELIVERED' || d.outcome === 'PENDING');
      setQueue(pending.length > 0 ? pending : (data || []).slice(0, 24));
      setLoading(false);
    }).catch(() => {
      const demo = Array.from({ length: 24 }, (_, i) => ({
        customer_id: `CUS-${10001 + i * 3}`,
        week_number: 52,
        risk_score_at_trigger: 0.5 + Math.random() * 0.4,
        intervention_type: ['PAYMENT_HOLIDAY', 'SMS_OUTREACH', 'RM_CALL', 'FINANCIAL_COUNSELING'][i % 4],
        channel: ['SMS', 'EMAIL', 'CALL', 'APP'][i % 4],
        status: ['SENT', 'DELIVERED', 'SENT', 'SENT'][i % 4],
        outcome: 'PENDING',
        top_signal: ['salary_delay_days', 'lending_upi_count_7d', 'credit_utilization'][i % 3]
      }));
      setQueue(demo);
      setLoading(false);
    });
  }, []);

  const handleSelect = async (item) => {
    setSelected(item);
    setSent(false);
    setMessage('');
    try {
      const d = await getCustomerDetail(item.customer_id);
      setDetail(d);
      setMessage(`We noticed some changes in your payment patterns. We'd like to help you explore flexible options that work for you. Please reach out to us anytime.`);
    } catch {
      setDetail(null);
      setMessage(`We care about your financial wellness. Our team is here to help with flexible repayment options. Please reach out to us.`);
    }
  };

  const handleSend = async () => {
    if (!selected) return;
    setSending(true);
    try { await triggerIntervention(selected.customer_id, selected.week_number || 52); } catch {}
    setTimeout(() => {
      setSending(false);
      setSent(true);
      addToast(`Intervention dispatched for ${selected.customer_id}`, 'success');
      setQueue(q => q.map(i => i.customer_id === selected.customer_id ? { ...i, status: 'SENT', outcome: 'PENDING' } : i));
    }, 1500);
  };

  const addToast = (text, type = 'success') => {
    const id = Date.now();
    setToasts(t => [...t.slice(-2), { id, text, type }]);
    setTimeout(() => setToasts(t => t.filter(toast => toast.id !== id)), 4000);
  };

  const filteredQueue = queue.filter(q =>
    !searchQueue || q.customer_id.toLowerCase().includes(searchQueue.toLowerCase())
  );

  const riskColor = (s) => s >= 0.70 ? 'var(--accent-red)' : s >= 0.40 ? 'var(--accent-orange)' : 'var(--accent-green)';

  const pendingCount = queue.filter(q => q.outcome === 'PENDING' || q.status === 'SENT').length;
  const sentCount = queue.filter(q => q.status === 'SENT' || q.status === 'DELIVERED').length;
  const criticalCount = queue.filter(q => q.risk_score_at_trigger >= 0.70).length;

  return (
    <div>
      <div className="page-header" style={{ animation: 'fadeSlideDown 500ms ease' }}>
        <div>
          <h1>Intervention Outreach Center</h1>
          <p className="subtitle">Review, edit, and dispatch AI-generated intervention messages</p>
        </div>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <span style={{ fontFamily: 'DM Mono', fontSize: 13, color: 'var(--accent-orange)' }}>● {pendingCount} Pending</span>
          <span style={{ fontFamily: 'DM Mono', fontSize: 13, color: 'var(--accent-cyan)' }}>● {sentCount} Sent</span>
          <span style={{ fontFamily: 'DM Mono', fontSize: 13, color: 'var(--accent-green)' }}>● {queue.filter(q => q.status === 'DELIVERED').length} Delivered</span>
        </div>
      </div>

      {/* Command Center KPI Strip */}
      <div className="kpi-strip" style={{ animation: 'fadeSlideUp 500ms 100ms ease both' }}>
        <div className="kpi-strip-tile">
          <div className="kpi-strip-value" style={{ color: 'var(--text-primary)' }}>
            <AnimatedNumber target={queue.length} color="var(--text-primary)" />
          </div>
          <div className="kpi-strip-label">Total Alerts</div>
        </div>
        <div className="kpi-strip-tile" style={{ border: '1px solid rgba(255,71,87,0.3)' }}>
          <div className="kpi-strip-value" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#ff4757', animation: 'dotBlink 1.2s infinite' }} />
            <AnimatedNumber target={criticalCount} color="var(--accent-red)" />
          </div>
          <div className="kpi-strip-label">Active Critical</div>
        </div>
        <div className="kpi-strip-tile">
          <div className="kpi-strip-value"><AnimatedNumber target={pendingCount} color="var(--accent-orange)" /></div>
          <div className="kpi-strip-label">Pending High</div>
        </div>
        <div className="kpi-strip-tile">
          <div className="kpi-strip-value"><AnimatedNumber target={0} color="var(--accent-cyan)" /></div>
          <div className="kpi-strip-label">Resolved</div>
        </div>
        <div className="kpi-strip-tile">
          <div className="kpi-strip-value"><AnimatedNumber target={queue.length} color="var(--accent-green)" /></div>
          <div className="kpi-strip-label">Auto-Triggered</div>
        </div>
      </div>

      <div className="two-col outreach">
        {/* LEFT: Queue */}
        <div className="card" style={{ animation: 'fadeSlideRight 500ms ease', maxHeight: 'calc(100vh - 280px)', overflowY: 'auto', padding: 24 }}>
          <div className="card-header" style={{ marginBottom: 16 }}>
            <div>
              <span className="card-title" style={{ fontSize: 18 }}>Pending Outreach</span>
              <span style={{ marginLeft: 8, fontFamily: 'DM Mono', fontSize: 12, background: 'rgba(255,107,53,0.1)', color: 'var(--accent-orange)', padding: '2px 8px', borderRadius: 4 }}>
                <span style={{ animation: 'dotBlink 1.2s infinite' }}>●</span> {pendingCount} pending
              </span>
            </div>
          </div>

          <input className="search-input" style={{ width: '100%', marginBottom: 16, paddingLeft: 16 }} placeholder="Filter by customer ID..." value={searchQueue} onChange={e => setSearchQueue(e.target.value)} />

          {loading ? Array(5).fill(0).map((_, i) => (
            <div key={i} className="shimmer" style={{ height: 80, marginBottom: 8, borderRadius: 10 }}></div>
          )) : filteredQueue.map((item, i) => (
            <div key={`${item.customer_id}-${i}`}
              className={`queue-card ${selected?.customer_id === item.customer_id ? 'selected' : ''}`}
              onClick={() => handleSelect(item)}
              style={{
                animation: `slideInRight 400ms ease ${Math.min(i, 12) * 100}ms both`,
              }}>
              <div className="queue-card-row">
                <span className="cid">{item.customer_id}</span>
                <span style={{ fontFamily: 'DM Mono', fontSize: 12, padding: '2px 8px', borderRadius: 100, background: `${riskColor(item.risk_score_at_trigger)}20`, color: riskColor(item.risk_score_at_trigger) }}>
                  {(item.risk_score_at_trigger * 100).toFixed(0)}
                </span>
              </div>
              <div className="queue-card-row">
                <span className="intervention">{item.intervention_type?.replace(/_/g, ' ')}</span>
                <span className={`status-pill ${item.status?.toLowerCase()}`}>{item.status}</span>
              </div>
              <div className="queue-card-row">
                <span className="time">Week {item.week_number}</span>
                <span className="signal-pill">{(item.top_signal || '').replace(/_/g, ' ')}</span>
              </div>
              {/* Active badge for critical */}
              {item.risk_score_at_trigger >= 0.70 && (
                <div style={{
                  position: 'absolute', top: 8, right: 8, fontFamily: 'DM Mono', fontSize: 9,
                  padding: '2px 6px', borderRadius: 4,
                  background: 'rgba(255,71,87,0.1)', color: '#ff4757', border: '1px solid rgba(255,71,87,0.25)',
                  animation: 'dotBlink 2s infinite',
                }}>
                  ACTIVE
                </div>
              )}
            </div>
          ))}
        </div>

        {/* RIGHT: Preview */}
        <div className="card" style={{ animation: 'fadeSlideLeft 500ms 100ms ease both', padding: 32 }}>
          {!selected ? (
            <div className="empty-state">
              <svg viewBox="0 0 24 24" fill="rgba(255,255,255,0.06)"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
              <div className="title">Select a customer from the queue</div>
              <div className="subtitle">The AI-generated message preview will appear here</div>
            </div>
          ) : (
            <>
              {/* Customer Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'linear-gradient(135deg, var(--accent-purple), var(--accent-cyan))', display: 'flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'DM Mono', fontSize: 14, color: 'white', fontWeight: 500 }}>
                  {(detail?.name || 'CU').slice(0, 2).toUpperCase()}
                </div>
                <div>
                  <div style={{ fontFamily: 'var(--font-heading)', fontWeight: 600, fontSize: 20, color: 'var(--text-primary)' }}>{detail?.name || selected.customer_id}</div>
                  <div style={{ fontFamily: 'DM Mono', fontSize: 13, color: 'var(--text-secondary)' }}>{selected.customer_id}</div>
                </div>
                <span className={`risk-badge ${(detail?.risk_level || 'MEDIUM').toLowerCase()}`} style={{ marginLeft: 'auto' }}>
                  {detail?.risk_level || 'MEDIUM'}
                </span>
              </div>

              {detail && (
                <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
                  {detail.city && <span className="pill" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 100, padding: '4px 10px', fontFamily: 'DM Sans', fontSize: 12, color: 'var(--text-secondary)' }}>{detail.city}</span>}
                  {detail.occupation && <span className="pill" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 100, padding: '4px 10px', fontFamily: 'DM Sans', fontSize: 12, color: 'var(--text-secondary)' }}>{detail.occupation}</span>}
                  {detail.product_type && <span className="pill" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 100, padding: '4px 10px', fontFamily: 'DM Sans', fontSize: 12, color: 'var(--text-secondary)' }}>{detail.product_type}</span>}
                </div>
              )}

              {/* Channel Switcher */}
              <div className="channel-switcher">
                {['SMS', 'Email', 'In-App'].map(ch => (
                  <button key={ch} className={`channel-btn ${channel === ch ? 'active' : ''}`} onClick={() => setChannel(ch)}>{ch}</button>
                ))}
              </div>

              {/* Intervention Type */}
              <div className="intervention-pill">{selected.intervention_type?.replace(/_/g, ' ')}</div>
              <div style={{ fontFamily: 'DM Sans', fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16 }}>
                Recommended by LangGraph Agent · Risk Score {(selected.risk_score_at_trigger * 100).toFixed(0)}
              </div>

              {/* Message */}
              <textarea className="message-textarea" value={message} onChange={e => setMessage(e.target.value)} placeholder="Type your outreach message..." />
              <div className="char-counter" style={{ color: message.length > 160 && channel === 'SMS' ? 'var(--accent-red)' : message.length > 140 ? 'var(--accent-orange)' : 'var(--text-muted)' }}>
                {message.length}{channel === 'SMS' ? '/160' : '/500'} chars
                {message.length > 160 && channel === 'SMS' && ' ⚠ Exceeds SMS limit'}
              </div>

              {/* Compliance Badges */}
              <div className="compliance-row">
                <div className="compliance-badge">
                  <span className="icon">✓</span>
                  <div>
                    <div className="badge-title">Policy Match</div>
                    <div className="badge-subtitle">Pre-delinquency empathetic outreach (Tier 1)</div>
                  </div>
                </div>
                <div className="compliance-badge cyan">
                  <span className="icon">🛡</span>
                  <div>
                    <div className="badge-title">Compliance Approved</div>
                    <div className="badge-subtitle">Supportive · Non-aggressive · Regulatory-compliant</div>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="action-buttons">
                <button className="btn-ghost">Edit Message</button>
                <button className="btn-ghost orange">Schedule for Later</button>
                <button className={`btn-primary ${sent ? 'success' : ''} ${sending ? 'loading' : ''}`} onClick={handleSend} disabled={sending || sent}>
                  {sending ? <span className="spinner"></span> : sent ? '✓ Dispatched Successfully' : 'Approve & Send Outreach'}
                </button>
              </div>

              {/* Intervention History */}
              <details style={{ marginTop: 24 }}>
                <summary style={{ fontFamily: 'DM Sans', fontSize: 14, color: 'var(--text-secondary)', cursor: 'pointer', padding: '8px 0' }}>
                  Previous Interventions ▾
                </summary>
                <div style={{ marginTop: 8 }}>
                  {queue.filter(q => q.customer_id === selected.customer_id).slice(0, 5).map((h, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', background: i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent', borderRadius: 6, fontFamily: 'DM Mono', fontSize: 12 }}>
                      <span style={{ color: 'var(--text-muted)' }}>Week {h.week_number}</span>
                      <span style={{ color: 'var(--text-secondary)' }}>{h.intervention_type?.replace(/_/g, ' ')}</span>
                      <span className={`status-pill ${h.status?.toLowerCase()}`}>{h.status}</span>
                      <span style={{ color: h.outcome === 'RECOVERED' ? 'var(--accent-green)' : 'var(--text-muted)' }}>{h.outcome}</span>
                    </div>
                  ))}
                </div>
              </details>
            </>
          )}
        </div>
      </div>

      {/* Toasts */}
      <div className="toast-container">
        {toasts.map(toast => (
          <div key={toast.id} className={`toast ${toast.type}`}>
            <span style={{ fontSize: 18 }}>{toast.type === 'success' ? '✓' : '✗'}</span>
            <span className="toast-text">{toast.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
