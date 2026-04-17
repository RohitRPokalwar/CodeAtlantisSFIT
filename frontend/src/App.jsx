import React, { useState, useEffect } from 'react';
import LandingHero from './components/LandingHero';
import Overview from './components/Overview';
import LiveFlagging from './components/LiveFlagging';
import RulesShap from './components/RulesShap';
import OutreachPanel from './components/OutreachPanel';
import ModelPredict from './components/ModelPredict';
import { login } from './api/client';

export default function App() {
  const [activeTab, setActiveTab] = useState('landing');
  const [scrolled, setScrolled] = useState(false);
  const [loggedIn, setLoggedIn] = useState(false);
  const [clock, setClock] = useState('');
  const [fadeKey, setFadeKey] = useState(0);
  const [badgeFlash, setBadgeFlash] = useState(false);

  useEffect(() => {
    login().then(() => setLoggedIn(true)).catch(() => setLoggedIn(true));
    const timer = setInterval(() => {
      const now = new Date();
      setClock(now.toLocaleTimeString('en-US', { hour12: false }));
    }, 1000);
    const handleScroll = () => setScrolled(window.scrollY > 80);
    window.addEventListener('scroll', handleScroll);

    // Badge flash every 30s
    const flashTimer = setInterval(() => {
      setBadgeFlash(true);
      setTimeout(() => setBadgeFlash(false), 1500);
    }, 30000);

    return () => {
      clearInterval(timer);
      clearInterval(flashTimer);
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleTabChange = (tab) => {
    setFadeKey(k => k + 1);
    setActiveTab(tab);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'predict', label: 'AI Predict' },
    { id: 'live', label: 'Live Flagging', liveDot: true, badge: '33', badgeColor: 'rgba(255,71,87,0.15)', badgeText: '#ff4757' },
    { id: 'rules', label: 'Rules & SHAP' },
    { id: 'outreach', label: 'Outreach', badge: '134', badgeColor: 'rgba(255,107,53,0.15)', badgeText: '#ff6b35' },
  ];

  // Landing page = no navbar
  if (activeTab === 'landing') {
    return (
      <LandingHero onEnterDashboard={() => handleTabChange('overview')} />
    );
  }

  return (
    <>
      {/* NAVBAR */}
      <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
        <div className="navbar-logo" onClick={() => handleTabChange('landing')}>
          <div className="dot"></div>
          <span>Praeven<span className="cyan">tix</span></span>
        </div>

        <div className="nav-tabs">
          {tabs.map(t => (
            <button key={t.id}
              className={`nav-tab ${activeTab === t.id ? 'active' : ''}`}
              onClick={() => handleTabChange(t.id)}>
              {t.label}
              {t.liveDot && <span className="live-dot" />}
              {t.badge && (
                <span
                  className="nav-badge"
                  style={{
                    background: badgeFlash ? 'rgba(245,158,11,0.25)' : t.badgeColor,
                    color: t.badgeText,
                    transition: 'background 300ms ease',
                  }}
                >
                  {t.badge}
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="nav-right">
          <div className="status-badge">
            <div className="dot"></div>
            <span>System Live</span>
          </div>
          <div className="nav-bell">
            <svg viewBox="0 0 24 24"><path d="M12 22c1.1 0 2-.9 2-2h-4a2 2 0 002 2zm6-6v-5c0-3.07-1.63-5.64-4.5-6.32V4c0-.83-.67-1.5-1.5-1.5s-1.5.67-1.5 1.5v.68C7.64 5.36 6 7.92 6 11v5l-2 2v1h16v-1l-2-2zm-2 1H8v-6c0-2.48 1.51-4.5 4-4.5s4 2.02 4 4.5v6z" /></svg>
          </div>
          <div className="nav-avatar">AD</div>
        </div>
      </nav>

      {/* PAGE CONTENT */}
      <div className="page-content" key={fadeKey}>
        {activeTab === 'overview' && <Overview clock={clock} />}
        {activeTab === 'predict' && <ModelPredict />}
        {activeTab === 'live' && <LiveFlagging />}
        {activeTab === 'rules' && <RulesShap />}
        {activeTab === 'outreach' && <OutreachPanel />}
      </div>

      {/* TOAST CONTAINER */}
      <div id="toast-container" className="toast-container"></div>
    </>
  );
}
