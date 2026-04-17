import React, { useEffect, useRef } from 'react';
import { TextRevealCardPreview } from './ui/TextRevealCard';

/* ═══════════════════════════════════════════
   LANDING HERO — TextRevealCard + Neural BG
   Canvas 2D neural network (smooth, no lag)
   ═══════════════════════════════════════════ */

// ── Lightweight Canvas 2D Neural Network ──
function NeuralBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animId;
    let w, h;

    const PARTICLE_COUNT = 90;
    const MAX_DIST = 140;
    const particles = [];

    const resize = () => {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    // Init particles
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: 1.5 + Math.random() * 1.5,
        color: i % 2 === 0 ? 'rgba(6,182,212,' : 'rgba(99,102,241,',
      });
    }

    let frame = 0;

    const draw = () => {
      animId = requestAnimationFrame(draw);
      frame++;
      ctx.clearRect(0, 0, w, h);

      // Move particles
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0) p.x = w;
        if (p.x > w) p.x = 0;
        if (p.y < 0) p.y = h;
        if (p.y > h) p.y = 0;
      }

      // Draw connections (every 2nd frame for perf)
      if (frame % 2 === 0) {
        for (let i = 0; i < PARTICLE_COUNT; i++) {
          for (let j = i + 1; j < PARTICLE_COUNT; j++) {
            const dx = particles[i].x - particles[j].x;
            const dy = particles[i].y - particles[j].y;
            const dist = dx * dx + dy * dy;
            if (dist < MAX_DIST * MAX_DIST) {
              const alpha = 1 - Math.sqrt(dist) / MAX_DIST;
              ctx.beginPath();
              ctx.strokeStyle = `rgba(6,182,212,${alpha * 0.12})`;
              ctx.lineWidth = 0.5;
              ctx.moveTo(particles[i].x, particles[i].y);
              ctx.lineTo(particles[j].x, particles[j].y);
              ctx.stroke();
            }
          }
        }
      }

      // Draw particles
      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = p.color + '0.6)';
        ctx.fill();

        // Glow
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r * 3, 0, Math.PI * 2);
        ctx.fillStyle = p.color + '0.04)';
        ctx.fill();
      }
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed', inset: 0, zIndex: 0,
        pointerEvents: 'none',
      }}
    />
  );
}

export default function LandingHero({ onEnterDashboard }) {
  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '40px 24px',
      background: '#0a0a0f',
      position: 'relative',
    }}>
      <NeuralBackground />

      {/* Card wrapper — relative so it sits above canvas */}
      <div style={{ position: 'relative', zIndex: 1, width: '100%', maxWidth: 800 }}>
        <TextRevealCardPreview />
      </div>

      {/* CTA Button */}
      <button
        onClick={onEnterDashboard}
        style={{
          position: 'relative', zIndex: 1,
          marginTop: 40, padding: '14px 40px', borderRadius: 12,
          background: 'linear-gradient(135deg, #00d4ff, #6366F1)',
          border: 'none', color: '#0a0a0f', fontFamily: "'Syne', sans-serif",
          fontWeight: 700, fontSize: 16, cursor: 'pointer',
          transition: 'all 200ms', letterSpacing: '0.5px',
        }}
        onMouseEnter={(e) => { e.target.style.transform = 'translateY(-3px) scale(1.03)'; e.target.style.boxShadow = '0 12px 40px rgba(0,212,255,0.4)'; }}
        onMouseLeave={(e) => { e.target.style.transform = ''; e.target.style.boxShadow = ''; }}
      >
        Enter Risk Operations Center →
      </button>
    </div>
  );
}
