import React, { useRef, useState, useEffect, useCallback } from 'react';

/* ═══════════════════════════════════════════════════════════════
   TextRevealCard — Aceternity-style hover reveal component
   Mask sweeps left→right following cursor X position,
   revealing the hidden "foresight" text underneath.
   ═══════════════════════════════════════════════════════════════ */

export function TextRevealCard({
  text,
  revealText,
  children,
  className = '',
  style = {},
}) {
  const cardRef = useRef(null);
  const [maskWidth, setMaskWidth] = useState(0);
  const [isHovered, setIsHovered] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 0, left: 0 });

  useEffect(() => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect();
      setDimensions({ width: rect.width, left: rect.left });
    }
    const handleResize = () => {
      if (cardRef.current) {
        const rect = cardRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, left: rect.left });
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleMouseMove = useCallback((e) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(x / rect.width, 1));
    setMaskWidth(pct * 100);
  }, []);

  const handleMouseEnter = useCallback(() => {
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    setIsHovered(false);
    setMaskWidth(0);
  }, []);

  return (
    <div
      ref={cardRef}
      className={`trc-card ${className}`}
      style={{
        background: '#111827',
        border: '1px solid rgba(99,102,241,0.2)',
        borderRadius: 16,
        padding: '40px 40px',
        position: 'relative',
        overflow: 'hidden',
        cursor: 'pointer',
        width: '100%',
        transition: 'border-color 300ms ease, box-shadow 300ms ease',
        boxShadow: isHovered
          ? '0 0 60px rgba(6,182,212,0.08), 0 20px 60px rgba(0,0,0,0.5)'
          : '0 12px 40px rgba(0,0,0,0.4)',
        borderColor: isHovered ? 'rgba(99,102,241,0.4)' : 'rgba(99,102,241,0.2)',
        ...style,
      }}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Children (title + description) */}
      <div style={{ position: 'relative', zIndex: 10, marginBottom: 32 }}>
        {children}
      </div>

      {/* Text layers */}
      <div style={{ position: 'relative', height: 70 }}>
        {/* Static text — muted (bank's blind spot) */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            fontFamily: "'Syne', sans-serif",
            fontWeight: 800,
            fontSize: 'clamp(18px, 2vw, 24px)',
            color: '#6b7280',
            letterSpacing: '-0.5px',
            lineHeight: 1.2,
            userSelect: 'none',
            whiteSpace: 'nowrap',
          }}
        >
          {text}
        </div>

        {/* Revealed text — bright with cyan glow (Praeventix foresight) */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            fontFamily: "'Syne', sans-serif",
            fontWeight: 900,
            fontSize: 'clamp(18px, 2vw, 28px)',
            color: '#ffffff',
            letterSpacing: '-0.5px',
            lineHeight: 1.2,
            userSelect: 'none',
            whiteSpace: 'nowrap',
            textShadow: '0 0 30px rgba(6,182,212,0.6), 0 0 60px rgba(6,182,212,0.3)',
            clipPath: `inset(0 ${100 - maskWidth}% 0 0)`,
            transition: isHovered ? 'none' : 'clip-path 400ms ease-out',
          }}
        >
          {revealText}
        </div>

        {/* Gradient reveal edge line */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            bottom: 0,
            left: `${maskWidth}%`,
            width: 3,
            background: 'linear-gradient(180deg, #06B6D4, #6366F1)',
            borderRadius: 4,
            opacity: isHovered ? 1 : 0,
            transition: isHovered ? 'opacity 150ms ease' : 'opacity 400ms ease, left 400ms ease',
            boxShadow: '0 0 20px rgba(6,182,212,0.6), 0 0 40px rgba(99,102,241,0.4)',
            pointerEvents: 'none',
          }}
        />

        {/* Gradient mask glow region behind the revealed text */}
        <div
          style={{
            position: 'absolute',
            top: -20,
            bottom: -20,
            left: 0,
            width: `${maskWidth}%`,
            background: 'linear-gradient(90deg, rgba(6,182,212,0.03), rgba(99,102,241,0.06))',
            pointerEvents: 'none',
            opacity: isHovered ? 1 : 0,
            transition: isHovered ? 'none' : 'opacity 400ms ease',
          }}
        />
      </div>

      {/* Corner shimmer decorations */}
      <div style={{
        position: 'absolute', top: 0, right: 0,
        width: 100, height: 100,
        background: 'radial-gradient(circle at top right, rgba(99,102,241,0.08), transparent 70%)',
        pointerEvents: 'none',
      }} />
      <div style={{
        position: 'absolute', bottom: 0, left: 0,
        width: 100, height: 100,
        background: 'radial-gradient(circle at bottom left, rgba(6,182,212,0.06), transparent 70%)',
        pointerEvents: 'none',
      }} />
    </div>
  );
}

export function TextRevealCardTitle({ children, style = {} }) {
  return (
    <h3 style={{
      fontFamily: "'Syne', sans-serif",
      fontWeight: 700,
      fontSize: 18,
      color: '#ffffff',
      margin: '0 0 8px',
      letterSpacing: '-0.3px',
      ...style,
    }}>
      {children}
    </h3>
  );
}

export function TextRevealCardDescription({ children, style = {} }) {
  return (
    <p style={{
      fontFamily: "'DM Sans', sans-serif",
      fontSize: 14,
      color: '#9ca3af',
      lineHeight: 1.7,
      margin: 0,
      maxWidth: 480,
      ...style,
    }}>
      {children}
    </p>
  );
}

/* ═══════════════════════════════════════════════════════════════
   TextRevealCardPreview — Complete Praeventix section
   Full dark section with card + stat pills below
   ═══════════════════════════════════════════════════════════════ */
export function TextRevealCardPreview() {
  const [visible, setVisible] = useState(false);
  const sectionRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.3 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  const pills = [
    {
      text: '₹16.5 Cr losses prevented',
      bg: 'rgba(16,185,129,0.1)',
      border: '1px solid rgba(16,185,129,0.2)',
      color: '#34d399',
    },
    {
      text: '500 customers monitored',
      bg: 'rgba(99,102,241,0.1)',
      border: '1px solid rgba(99,102,241,0.2)',
      color: '#818cf8',
    },
    {
      text: '74% ensemble accuracy',
      bg: 'rgba(6,182,212,0.1)',
      border: '1px solid rgba(6,182,212,0.2)',
      color: '#22d3ee',
    },
  ];

  return (
    <div
      ref={sectionRef}
      style={{
        background: 'transparent',
        borderRadius: 16,
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '20px 0',
        position: 'relative',
      }}
    >
      {/* Subtle grid background */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
      }} />

      {/* Card */}
      <div style={{
        position: 'relative', zIndex: 1, width: '100%',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(20px)',
        transition: 'opacity 800ms ease, transform 800ms ease',
      }}>
        <TextRevealCard
          text="Most banks react after default."
          revealText="We intervene 15 days before."
        >
          <TextRevealCardTitle>
            Sometimes, you just need to see it coming.
          </TextRevealCardTitle>
          <TextRevealCardDescription>
            Praeventix EWS-360 detects salary delays, savings drawdown,
            and EMI bounce risk — before a single payment is missed.
          </TextRevealCardDescription>
        </TextRevealCard>
      </div>

      {/* Stat Pills */}
      <div style={{
        display: 'flex',
        gap: 12,
        marginTop: 40,
        justifyContent: 'center',
        flexWrap: 'wrap',
        position: 'relative',
        zIndex: 1,
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(16px)',
        transition: 'opacity 800ms ease 300ms, transform 800ms ease 300ms',
      }}>
        {pills.map((pill, i) => (
          <span
            key={i}
            style={{
              fontFamily: "'DM Sans', sans-serif",
              fontSize: 12,
              fontWeight: 600,
              padding: '8px 16px',
              borderRadius: 100,
              background: pill.bg,
              border: pill.border,
              color: pill.color,
              letterSpacing: '0.2px',
              transition: 'transform 200ms ease, box-shadow 200ms ease',
              cursor: 'default',
            }}
            onMouseEnter={(e) => {
              e.target.style.transform = 'translateY(-2px)';
              e.target.style.boxShadow = `0 8px 24px ${pill.bg}`;
            }}
            onMouseLeave={(e) => {
              e.target.style.transform = '';
              e.target.style.boxShadow = '';
            }}
          >
            {pill.text}
          </span>
        ))}
      </div>
    </div>
  );
}

export default TextRevealCard;
