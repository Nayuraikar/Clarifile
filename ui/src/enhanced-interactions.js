// ===== ULTRA-ADVANCED PREMIUM INTERACTIONS =====
// Professional-grade interaction system for premium websites

// Prevent duplicate loading
if (typeof window.ClarifileInteractions !== 'undefined') {
  console.log('ClarifileInteractions already loaded, skipping...');
} else {
  window.ClarifileInteractions = {};

class UltraAdvancedInteractions {
  constructor() {
    this.mouseX = 0;
    this.mouseY = 0;
    this.particles = [];
    this.magneticElements = [];
    this.init();
  }

  init() {
    this.setupAdvancedCursor();
    this.setupMagneticElements();
    this.setupAdvancedParticles();
    this.setupCinematicScrollEffects();
    this.setupPremiumHoverEffects();
    this.setupAdvancedRipples();
    this.setupMorphingShapes();
  }

  setupAdvancedCursor() {
    // Create ultra-premium cursor system
    const cursor = document.createElement('div');
    cursor.className = 'ultra-cursor';
    cursor.innerHTML = `
      <div class="cursor-core"></div>
      <div class="cursor-trail"></div>
      <div class="cursor-glow"></div>
    `;
    document.body.appendChild(cursor);

    // Advanced cursor tracking with smooth interpolation
    let targetX = 0, targetY = 0;
    let currentX = 0, currentY = 0;

    document.addEventListener('mousemove', (e) => {
      targetX = e.clientX;
      targetY = e.clientY;
      this.mouseX = e.clientX;
      this.mouseY = e.clientY;
    });

    const updateCursor = () => {
      currentX += (targetX - currentX) * 0.15;
      currentY += (targetY - currentY) * 0.15;
      
      cursor.style.transform = `translate(${currentX}px, ${currentY}px)`;
      requestAnimationFrame(updateCursor);
    };
    updateCursor();

    // Enhanced hover states
    document.addEventListener('mouseover', (e) => {
      if (e.target.matches('button, a, .interactive, .professional-interactive')) {
        cursor.classList.add('cursor-hover');
      }
    });

    document.addEventListener('mouseout', (e) => {
      if (e.target.matches('button, a, .interactive, .professional-interactive')) {
        cursor.classList.remove('cursor-hover');
      }
    });
  }

  setupMagneticElements() {
    // Magnetic attraction effect for interactive elements
    const magneticElements = document.querySelectorAll('.professional-btn, .professional-card, .professional-interactive');
    
    magneticElements.forEach(element => {
      element.addEventListener('mousemove', (e) => {
        const rect = element.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const deltaX = (e.clientX - centerX) * 0.15;
        const deltaY = (e.clientY - centerY) * 0.15;
        
        element.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(1.05)`;
      });

      element.addEventListener('mouseleave', () => {
        element.style.transform = 'translate(0px, 0px) scale(1)';
      });
    });
  }

  setupAdvancedParticles() {
    // Create advanced particle system container
    const particleContainer = document.createElement('div');
    particleContainer.className = 'ultra-particle-system';
    document.body.appendChild(particleContainer);

    // Generate premium particles
    for (let i = 0; i < 30; i++) {
      this.createAdvancedParticle(particleContainer);
    }

    // Mouse-following particle trail
    document.addEventListener('mousemove', (e) => {
      if (Math.random() < 0.1) { // 10% chance to create particle
        this.createMouseParticle(e.clientX, e.clientY);
      }
    });
  }

  createAdvancedParticle(container) {
    const particle = document.createElement('div');
    particle.className = 'ultra-particle';
    
    // Random positioning and animation
    particle.style.left = Math.random() * 100 + '%';
    particle.style.animationDelay = Math.random() * 15 + 's';
    particle.style.animationDuration = (10 + Math.random() * 10) + 's';
    
    container.appendChild(particle);
    
    // Remove and recreate particle after animation
    setTimeout(() => {
      if (particle.parentNode) {
        particle.parentNode.removeChild(particle);
        this.createAdvancedParticle(container);
      }
    }, (10 + Math.random() * 10) * 1000);
  }

  createMouseParticle(x, y) {
    const particle = document.createElement('div');
    particle.className = 'mouse-particle';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    
    document.body.appendChild(particle);
    
    // Animate and remove
    setTimeout(() => {
      particle.style.opacity = '0';
      particle.style.transform = 'scale(0) translateY(-50px)';
    }, 50);
    
    setTimeout(() => {
      if (particle.parentNode) {
        particle.parentNode.removeChild(particle);
      }
    }, 1000);
  }

  setupCinematicScrollEffects() {
    // Advanced intersection observer for scroll animations
    const observerOptions = {
      threshold: [0, 0.25, 0.5, 0.75, 1],
      rootMargin: '-50px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const element = entry.target;
        const ratio = entry.intersectionRatio;
        
        if (entry.isIntersecting) {
          element.classList.add('ultra-scroll-reveal');
          
          // Parallax effect based on scroll position
          const translateY = (1 - ratio) * 50;
          element.style.transform = `translateY(${translateY}px)`;
        }
      });
    }, observerOptions);

    // Observe all cards and interactive elements
    document.querySelectorAll('.professional-card, .fade-in').forEach(el => {
      observer.observe(el);
    });
  }

  setupPremiumHoverEffects() {
    // Advanced hover effects for cards
    document.querySelectorAll('.professional-card').forEach(card => {
      card.addEventListener('mouseenter', (e) => {
        this.createHoverGlow(card);
        card.classList.add('ultra-hover-effect');
      });

      card.addEventListener('mouseleave', (e) => {
        card.classList.remove('ultra-hover-effect');
      });

      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        
        card.style.setProperty('--mouse-x', x + '%');
        card.style.setProperty('--mouse-y', y + '%');
      });
    });
  }

  createHoverGlow(element) {
    const glow = document.createElement('div');
    glow.className = 'hover-glow';
    element.appendChild(glow);
    
    setTimeout(() => {
      if (glow.parentNode) {
        glow.parentNode.removeChild(glow);
      }
    }, 600);
  }

  setupAdvancedRipples() {
    // Premium ripple effect on click
    document.addEventListener('click', (e) => {
      if (e.target.closest('.professional-btn, .professional-interactive')) {
        this.createAdvancedRipple(e.clientX, e.clientY, e.target.closest('.professional-btn, .professional-interactive'));
      }
    });
  }

  createAdvancedRipple(x, y, element) {
    const ripple = document.createElement('div');
    ripple.className = 'ultra-ripple-effect';
    
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height) * 2;
    
    ripple.style.width = size + 'px';
    ripple.style.height = size + 'px';
    ripple.style.left = (x - rect.left - size / 2) + 'px';
    ripple.style.top = (y - rect.top - size / 2) + 'px';
    
    element.appendChild(ripple);
    
    setTimeout(() => {
      ripple.style.transform = 'scale(1)';
      ripple.style.opacity = '0';
    }, 50);
    
    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple);
      }
    }, 600);
  }

  setupMorphingShapes() {
    // Create morphing background shapes
    const morphContainer = document.createElement('div');
    morphContainer.className = 'ultra-morph-container';
    document.body.appendChild(morphContainer);

    for (let i = 0; i < 4; i++) {
      const shape = document.createElement('div');
      shape.className = 'ultra-morph-shape';
      morphContainer.appendChild(shape);
    }
  }
}

// Enhanced styles for ultra-advanced interactions
const ultraStyles = document.createElement('style');
ultraStyles.textContent = `
  /* Ultra-Advanced Cursor */
  .ultra-cursor {
    position: fixed;
    width: 40px;
    height: 40px;
    pointer-events: none;
    z-index: 9999;
    transform: translate(-50%, -50%);
    transition: all 0.3s cubic-bezier(0.23, 1, 0.32, 1);
  }

  .cursor-core {
    width: 8px;
    height: 8px;
    background: rgba(193, 154, 107, 0.8);
    border-radius: 50%;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .cursor-trail {
    width: 100%;
    height: 100%;
    border: 2px solid rgba(193, 154, 107, 0.3);
    border-radius: 50%;
    position: absolute;
    animation: cursorPulse 2s ease-in-out infinite;
  }

  .cursor-glow {
    width: 60px;
    height: 60px;
    background: radial-gradient(circle, rgba(193, 154, 107, 0.1) 0%, transparent 70%);
    border-radius: 50%;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .ultra-cursor.cursor-hover {
    transform: translate(-50%, -50%) scale(1.5);
  }

  .ultra-cursor.cursor-hover .cursor-core {
    background: rgba(212, 165, 116, 1);
    transform: translate(-50%, -50%) scale(1.5);
  }

  @keyframes cursorPulse {
    0%, 100% { transform: scale(1); opacity: 0.3; }
    50% { transform: scale(1.2); opacity: 0.6; }
  }

  /* Mouse Particles */
  .mouse-particle {
    position: fixed;
    width: 4px;
    height: 4px;
    background: rgba(193, 154, 107, 0.6);
    border-radius: 50%;
    pointer-events: none;
    z-index: 9998;
    transform: translate(-50%, -50%);
    transition: all 1s ease-out;
    box-shadow: 0 0 10px rgba(193, 154, 107, 0.4);
  }

  /* Ultra Ripple Effect */
  .ultra-ripple-effect {
    position: absolute;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(193, 154, 107, 0.3) 0%, transparent 70%);
    transform: scale(0);
    opacity: 1;
    transition: all 0.6s ease-out;
    pointer-events: none;
  }

  /* Hover Glow */
  .hover-glow {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at var(--mouse-x, 50%) var(--mouse-y, 50%), 
      rgba(193, 154, 107, 0.1) 0%, transparent 50%);
    border-radius: inherit;
    pointer-events: none;
    animation: glowFade 0.6s ease-out;
  }

  @keyframes glowFade {
    0% { opacity: 0; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1); }
    100% { opacity: 0; transform: scale(1.2); }
  }

  /* Ultra Scroll Reveal */
  .ultra-scroll-reveal {
    animation: ultraReveal 1s cubic-bezier(0.23, 1, 0.32, 1) forwards;
  }

  @keyframes ultraReveal {
    0% {
      opacity: 0;
      transform: translateY(50px) scale(0.9);
      filter: blur(5px);
    }
    100% {
      opacity: 1;
      transform: translateY(0) scale(1);
      filter: blur(0px);
    }
  }
`;

document.head.appendChild(ultraStyles);

// Initialize ultra-advanced interactions
document.addEventListener('DOMContentLoaded', () => {
  new UltraAdvancedInteractions();
});

// Smooth page transitions
window.addEventListener('beforeunload', () => {
  document.body.style.opacity = '0';
  document.body.style.transform = 'scale(0.95)';
});

// Enhanced loading state
window.addEventListener('load', () => {
  document.body.style.opacity = '0';
  document.body.style.transform = 'scale(0.95)';
  document.body.style.transition = 'all 0.6s cubic-bezier(0.23, 1, 0.32, 1)';
  
  setTimeout(() => {
    document.body.style.opacity = '1';
    document.body.style.transform = 'scale(1)';
  }, 100);
});

// Add smooth magnetic effect
class SmoothMagneticEffect {
  constructor() {
    this.init();
  }

  init() {
    const cards = document.querySelectorAll('.professional-card, .aesthetic-card');
    
    cards.forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const deltaX = (e.clientX - centerX) * 0.08; // Reduced for smoothness
        const deltaY = (e.clientY - centerY) * 0.08;
        
        card.style.transform = `translate(${deltaX}px, ${deltaY}px) scale(1.02)`;
        card.style.transition = 'transform 0.3s cubic-bezier(0.165, 0.84, 0.44, 1)';
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'translate(0px, 0px) scale(1)';
        card.style.transition = 'transform 0.8s cubic-bezier(0.165, 0.84, 0.44, 1)';
      });
    });
  }
}

// Smooth scroll reveal
class SmoothScrollReveal {
  constructor() {
    this.init();
  }

  init() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0) scale(1)';
          entry.target.style.filter = 'blur(0)';
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.professional-card, .fade-in').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(30px) scale(0.95)';
      el.style.filter = 'blur(2px)';
      el.style.transition = 'all 1.2s cubic-bezier(0.165, 0.84, 0.44, 1)';
      observer.observe(el);
    });
  }
}

// Initialize smooth effects
document.addEventListener('DOMContentLoaded', () => {
  new SmoothMagneticEffect();
  new SmoothScrollReveal();
});

// ===== COMPLEX UI/UX INTERACTIONS =====

// Advanced 3D Tilt Effect
class Complex3DTilt {
  constructor() {
    this.init();
  }

  init() {
    const tiltCards = document.querySelectorAll('.complex-tilt-card, .professional-card');
    
    tiltCards.forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        const rotateX = (e.clientY - centerY) / 10;
        const rotateY = (centerX - e.clientX) / 10;
        
        card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(50px)`;
      });

      card.addEventListener('mouseleave', () => {
        card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) translateZ(0px)';
      });
    });
  }
}

// Floating Action Button Menu
class ComplexFAB {
  constructor() {
    this.createFAB();
  }

  createFAB() {
    const fab = document.createElement('div');
    fab.className = 'complex-fab';
    fab.innerHTML = `
      <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
        <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
      </svg>
      <div class="complex-fab-menu">
        <div class="complex-fab-item" data-tooltip="Settings">⚙️</div>
        <div class="complex-fab-item" data-tooltip="Help">❓</div>
        <div class="complex-fab-item" data-tooltip="Share">📤</div>
        <div class="complex-fab-item" data-tooltip="Download">⬇️</div>
      </div>
    `;
    
    document.body.appendChild(fab);
    
    fab.addEventListener('click', () => {
      fab.classList.toggle('active');
    });
  }
}

// Advanced Progress Rings
class ComplexProgressRing {
  constructor() {
    this.createProgressRings();
  }

  createProgressRings() {
    const rings = document.querySelectorAll('.complex-progress-ring');
    
    rings.forEach(ring => {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.innerHTML = `
        <defs>
          <linearGradient id="progressGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style="stop-color:#c19a6b"/>
            <stop offset="100%" style="stop-color:#d4a574"/>
          </linearGradient>
        </defs>
        <circle class="bg-circle" cx="60" cy="60" r="45"/>
        <circle class="progress-circle" cx="60" cy="60" r="45"/>
      `;
      
      ring.appendChild(svg);
      
      // Animate on scroll
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            ring.classList.add('animate');
          }
        });
      });
      
      observer.observe(ring);
    });
  }
}

// Interactive Background Patterns
class ComplexBackgroundPattern {
  constructor() {
    this.createPattern();
  }

  createPattern() {
    const pattern = document.createElement('div');
    pattern.className = 'complex-bg-pattern';
    document.body.appendChild(pattern);
    
    // Mouse-following effect
    document.addEventListener('mousemove', (e) => {
      const x = (e.clientX / window.innerWidth) * 100;
      const y = (e.clientY / window.innerHeight) * 100;
      
      pattern.style.background = `
        radial-gradient(circle at ${x}% ${y}%, rgba(193, 154, 107, 0.15) 0%, transparent 50%),
        radial-gradient(circle at ${100-x}% ${100-y}%, rgba(212, 165, 116, 0.1) 0%, transparent 50%)
      `;
    });
  }
}

// Advanced Parallax Effect
class ComplexParallax {
  constructor() {
    this.init();
  }

  init() {
    const parallaxElements = document.querySelectorAll('.complex-parallax-layer');
    
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      
      parallaxElements.forEach((element, index) => {
        const speed = (index + 1) * 0.5;
        const yPos = -(scrolled * speed);
        element.style.transform = `translateY(${yPos}px)`;
      });
    });
  }
}

// Advanced Typewriter Effect
class ComplexTypewriter {
  constructor(element, text, speed = 100) {
    this.element = element;
    this.text = text;
    this.speed = speed;
    this.index = 0;
    this.type();
  }

  type() {
    if (this.index < this.text.length) {
      this.element.textContent += this.text.charAt(this.index);
      this.index++;
      setTimeout(() => this.type(), this.speed);
    }
  }
}

// Interactive Data Visualization
class ComplexDataViz {
  constructor() {
    this.createCharts();
  }

  createCharts() {
    // Animated counter
    const counters = document.querySelectorAll('[data-count]');
    
    counters.forEach(counter => {
      const target = parseInt(counter.getAttribute('data-count'));
      let current = 0;
      const increment = target / 100;
      
      const updateCounter = () => {
        if (current < target) {
          current += increment;
          counter.textContent = Math.ceil(current);
          requestAnimationFrame(updateCounter);
        } else {
          counter.textContent = target;
        }
      };
      
      // Start animation when visible
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            updateCounter();
            observer.unobserve(counter);
          }
        });
      });
      
      observer.observe(counter);
    });
  }
}

// Advanced Gesture Recognition
class ComplexGestures {
  constructor() {
    this.init();
  }

  init() {
    let startX, startY, endX, endY;
    
    document.addEventListener('touchstart', (e) => {
      startX = e.touches[0].clientX;
      startY = e.touches[0].clientY;
    });
    
    document.addEventListener('touchend', (e) => {
      endX = e.changedTouches[0].clientX;
      endY = e.changedTouches[0].clientY;
      
      const deltaX = endX - startX;
      const deltaY = endY - startY;
      
      if (Math.abs(deltaX) > Math.abs(deltaY)) {
        if (deltaX > 50) {
          this.handleSwipeRight();
        } else if (deltaX < -50) {
          this.handleSwipeLeft();
        }
      } else {
        if (deltaY > 50) {
          this.handleSwipeDown();
        } else if (deltaY < -50) {
          this.handleSwipeUp();
        }
      }
    });
  }

  handleSwipeRight() {
    document.body.style.transform = 'translateX(10px)';
    setTimeout(() => document.body.style.transform = 'translateX(0)', 200);
  }

  handleSwipeLeft() {
    document.body.style.transform = 'translateX(-10px)';
    setTimeout(() => document.body.style.transform = 'translateX(0)', 200);
  }

  handleSwipeUp() {
    window.scrollBy({ top: -100, behavior: 'smooth' });
  }

  handleSwipeDown() {
    window.scrollBy({ top: 100, behavior: 'smooth' });
  }
}

// ===== INSANE PREMIUM UI/UX INTERACTIONS =====

// Mind-Blowing 3D Scene
class Insane3DScene {
  constructor() {
    this.mouseX = 0;
    this.mouseY = 0;
    this.init();
  }

  init() {
    // Create 3D scene
    const scene = document.createElement('div');
    scene.className = 'insane-3d-scene';
    
    for (let i = 0; i < 3; i++) {
      const layer = document.createElement('div');
      layer.className = 'insane-3d-layer';
      scene.appendChild(layer);
    }
    
    document.body.appendChild(scene);
    
    // Mouse tracking for 3D effect
    document.addEventListener('mousemove', (e) => {
      this.mouseX = (e.clientX / window.innerWidth) * 100;
      this.mouseY = (e.clientY / window.innerHeight) * 100;
      
      scene.style.setProperty('--mouse-x', this.mouseX + '%');
      scene.style.setProperty('--mouse-y', this.mouseY + '%');
      
      const rotateX = (this.mouseY - 50) * 0.1;
      const rotateY = (this.mouseX - 50) * 0.1;
      
      scene.style.transform = `perspective(2000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    });
  }
}

// Particle Explosion System
class InsaneParticleExplosion {
  constructor() {
    this.init();
  }

  init() {
    document.addEventListener('click', (e) => {
      this.createExplosion(e.clientX, e.clientY);
    });
  }

  createExplosion(x, y) {
    const explosion = document.createElement('div');
    explosion.className = 'insane-particle-explosion';
    explosion.style.left = x + 'px';
    explosion.style.top = y + 'px';
    
    for (let i = 0; i < 20; i++) {
      const particle = document.createElement('div');
      particle.className = 'insane-explosion-particle';
      
      const angle = (i / 20) * Math.PI * 2;
      const distance = 100 + Math.random() * 100;
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance;
      
      particle.style.setProperty('--dx', dx + 'px');
      particle.style.setProperty('--dy', dy + 'px');
      particle.style.animationDelay = Math.random() * 0.5 + 's';
      
      explosion.appendChild(particle);
    }
    
    document.body.appendChild(explosion);
    
    setTimeout(() => {
      if (explosion.parentNode) {
        explosion.parentNode.removeChild(explosion);
      }
    }, 2000);
  }
}

// Insane Scroll Progress
class InsaneScrollProgress {
  constructor() {
    this.createProgressBar();
  }

  createProgressBar() {
    const progressContainer = document.createElement('div');
    progressContainer.className = 'insane-scroll-progress';
    
    const progressBar = document.createElement('div');
    progressBar.className = 'insane-progress-bar';
    progressBar.style.transform = 'scaleX(0)';
    
    progressContainer.appendChild(progressBar);
    document.body.appendChild(progressContainer);
    
    window.addEventListener('scroll', () => {
      const scrolled = window.pageYOffset;
      const maxHeight = document.documentElement.scrollHeight - window.innerHeight;
      const progress = (scrolled / maxHeight) * 100;
      
      progressBar.style.transform = `scaleX(${progress / 100})`;
    });
  }
}

// Floating Orbs System
class InsaneFloatingOrbs {
  constructor() {
    this.createOrbs();
  }

  createOrbs() {
    setInterval(() => {
      if (Math.random() < 0.3) { // 30% chance every interval
        this.createOrb();
      }
    }, 2000);
  }

  createOrb() {
    const orb = document.createElement('div');
    orb.className = 'insane-floating-orb';
    orb.style.left = '-50px';
    orb.style.top = Math.random() * window.innerHeight + 'px';
    orb.style.animationDuration = (10 + Math.random() * 10) + 's';
    
    document.body.appendChild(orb);
    
    setTimeout(() => {
      if (orb.parentNode) {
        orb.parentNode.removeChild(orb);
      }
    }, 20000);
  }
}

// Insane Text Effects
class InsaneTextEffects {
  constructor() {
    this.init();
  }

  init() {
    const textElements = document.querySelectorAll('.insane-text-explosion');
    
    textElements.forEach(element => {
      element.setAttribute('data-text', element.textContent);
      
      element.addEventListener('mouseenter', () => {
        this.createTextExplosion(element);
      });
    });
  }

  createTextExplosion(element) {
    const rect = element.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    
    for (let i = 0; i < 10; i++) {
      const particle = document.createElement('div');
      particle.textContent = element.textContent[Math.floor(Math.random() * element.textContent.length)];
      particle.style.position = 'fixed';
      particle.style.left = centerX + 'px';
      particle.style.top = centerY + 'px';
      particle.style.color = `hsl(${Math.random() * 360}, 70%, 60%)`;
      particle.style.fontSize = '20px';
      particle.style.fontWeight = 'bold';
      particle.style.pointerEvents = 'none';
      particle.style.zIndex = '9999';
      particle.style.transition = 'all 1s ease-out';
      
      document.body.appendChild(particle);
      
      setTimeout(() => {
        const angle = Math.random() * Math.PI * 2;
        const distance = 50 + Math.random() * 100;
        particle.style.transform = `translate(${Math.cos(angle) * distance}px, ${Math.sin(angle) * distance}px) scale(0)`;
        particle.style.opacity = '0';
      }, 50);
      
      setTimeout(() => {
        if (particle.parentNode) {
          particle.parentNode.removeChild(particle);
        }
      }, 1000);
    }
  }
}

// ===== BEAUTIFUL AESTHETIC INTERACTIONS =====

// Beautiful Animated Background
class BeautifulAnimatedBackground {
  constructor() {
    this.mouseX = 50;
    this.mouseY = 50;
    this.init();
  }

  init() {
    // Create beautiful animated background
    const bg = document.createElement('div');
    bg.className = 'beautiful-animated-bg';
    document.body.appendChild(bg);
    
    // Mouse tracking for background effects
    document.addEventListener('mousemove', (e) => {
      this.mouseX = (e.clientX / window.innerWidth) * 100;
      this.mouseY = (e.clientY / window.innerHeight) * 100;
      
      bg.style.setProperty('--mouse-x', this.mouseX + '%');
      bg.style.setProperty('--mouse-y', this.mouseY + '%');
    });
  }
}

// Beautiful Hover Text Effects
class BeautifulHoverEffects {
  constructor() {
    this.init();
  }

  init() {
    // Add beautiful hover effects to all text elements
    const textElements = document.querySelectorAll('h1, h2, h3, h4, h5, h6, p, span, div');
    
    textElements.forEach(element => {
      if (element.textContent.trim().length > 0 && !element.querySelector('*')) {
        element.classList.add('beautiful-hover-text');
      }
    });

    // Add beautiful effects to specific elements
    const headings = document.querySelectorAll('.hero-heading, .section-heading, .card-heading');
    headings.forEach(heading => {
      heading.classList.add('beautiful-text-glow', 'beautiful-smooth-scale');
    });
  }
}

// Beautiful Button Zoom Effects
class BeautifulButtonEffects {
  constructor() {
    this.init();
  }

  init() {
    // Add zoom effects to all buttons
    const buttons = document.querySelectorAll('button, .professional-btn, [role="button"]');
    
    buttons.forEach(button => {
      button.classList.add('beautiful-zoom-button', 'beautiful-shimmer');
      
      // Add click ripple effect
      button.addEventListener('click', (e) => {
        this.createRipple(e, button);
      });
    });
  }

  createRipple(event, element) {
    const ripple = document.createElement('div');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.style.position = 'absolute';
    ripple.style.borderRadius = '50%';
    ripple.style.background = 'rgba(255, 255, 255, 0.6)';
    ripple.style.transform = 'scale(0)';
    ripple.style.animation = 'ripple 0.6s linear';
    ripple.style.pointerEvents = 'none';
    
    element.style.position = 'relative';
    element.appendChild(ripple);
    
    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple);
      }
    }, 600);
  }
}

// Beautiful Card Breathing Effects
class BeautifulCardEffects {
  constructor() {
    this.init();
  }

  init() {
    // Add breathing animation to all cards
    const cards = document.querySelectorAll('.professional-card');
    
    cards.forEach((card, index) => {
      card.classList.add('beautiful-breathing-card', 'beautiful-gradient-border', 'beautiful-float');
      
      // Stagger the breathing animation
      card.style.animationDelay = (index * 0.5) + 's';
      
      // Add hover glow effect
      card.addEventListener('mouseenter', () => {
        this.createHoverGlow(card);
      });
    });
  }

  createHoverGlow(element) {
    const glow = document.createElement('div');
    glow.style.position = 'absolute';
    glow.style.top = '-10px';
    glow.style.left = '-10px';
    glow.style.right = '-10px';
    glow.style.bottom = '-10px';
    glow.style.background = 'radial-gradient(circle, rgba(193, 154, 107, 0.3) 0%, transparent 70%)';
    glow.style.borderRadius = 'inherit';
    glow.style.pointerEvents = 'none';
    glow.style.zIndex = '-1';
    glow.style.opacity = '0';
    glow.style.transition = 'opacity 0.6s ease';
    glow.className = 'hover-glow-effect';
    
    element.style.position = 'relative';
    element.appendChild(glow);
    
    setTimeout(() => {
      glow.style.opacity = '1';
    }, 50);
    
    element.addEventListener('mouseleave', () => {
      glow.style.opacity = '0';
      setTimeout(() => {
        if (glow.parentNode) {
          glow.parentNode.removeChild(glow);
        }
      }, 600);
    }, { once: true });
  }
}

// Beautiful Navigation Effects
class BeautifulNavEffects {
  constructor() {
    this.init();
  }

  init() {
    // Add beautiful effects to navigation items
    const navItems = document.querySelectorAll('.professional-nav-item');
    
    navItems.forEach(item => {
      item.classList.add('beautiful-nav-item', 'beautiful-pulse');
    });
  }
}

// Beautiful Scroll Animations
class BeautifulScrollAnimations {
  constructor() {
    this.init();
  }

  init() {
    // Enhanced scroll observer with beautiful effects
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('beautiful-smooth-scale');
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0) scale(1)';
          entry.target.style.filter = 'blur(0)';
          
          // Add staggered shimmer effect
          setTimeout(() => {
            entry.target.classList.add('beautiful-shimmer');
          }, Math.random() * 500);
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });

    // Observe all elements with fade-in class
    document.querySelectorAll('.fade-in').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(30px) scale(0.95)';
      el.style.filter = 'blur(2px)';
      el.style.transition = 'all 1.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
      observer.observe(el);
    });
  }
}

// Add ripple animation to CSS
const rippleCSS = document.createElement('style');
rippleCSS.textContent = `
  @keyframes ripple {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`;
document.head.appendChild(rippleCSS);

// ===== INSANE DYNAMIC PERSONALITY SYSTEM =====

// EXPLOSIVE Dynamic Entrance Manager
class DynamicExplosiveEntrance {
  constructor() {
    this.init();
  }

  init() {
    // Make EVERYTHING explode in dynamically!
    const allElements = document.querySelectorAll('.professional-card, .hero-heading, .section-heading, .professional-btn');
    
    allElements.forEach((element, index) => {
      element.classList.add('dynamic-explode-in');
      element.style.animationDelay = (index * 0.2) + 's';
    });

    // Add random dynamic effects to elements
    setTimeout(() => {
      this.addRandomDynamicEffects();
    }, 2000);
  }

  addRandomDynamicEffects() {
    const cards = document.querySelectorAll('.professional-card');
    const effects = [
      'dynamic-bounce-crazy',
      'dynamic-pulse-energy', 
      'dynamic-wiggle',
      'dynamic-dance',
      'dynamic-float-bubble',
      'dynamic-heartbeat',
      'dynamic-wave'
    ];

    cards.forEach((card, index) => {
      const randomEffect = effects[index % effects.length];
      card.classList.add(randomEffect);
      
      // Add rotating background to some cards
      if (index % 2 === 0) {
        card.classList.add('dynamic-rotating-bg');
      }
    });
  }
}

// CRAZY Button Personality
class DynamicButtonPersonality {
  constructor() {
    this.init();
  }

  init() {
    const buttons = document.querySelectorAll('button, .professional-btn');
    
    buttons.forEach((button, index) => {
      // Add different personalities to buttons
      const personalities = [
        'dynamic-pulse-energy',
        'dynamic-glow-pulse', 
        'dynamic-elastic',
        'dynamic-zoom-explode',
        'dynamic-jello'
      ];
      
      const personality = personalities[index % personalities.length];
      button.classList.add(personality);
      
      // Make buttons shake when hovered
      button.addEventListener('mouseenter', () => {
        button.classList.add('dynamic-shake');
        this.createButtonExplosion(button);
      });
      
      button.addEventListener('mouseleave', () => {
        button.classList.remove('dynamic-shake');
      });
      
      // Rainbow effect on click
      button.addEventListener('click', () => {
        button.classList.add('dynamic-rainbow');
        setTimeout(() => {
          button.classList.remove('dynamic-rainbow');
        }, 2000);
      });
    });
  }

  createButtonExplosion(button) {
    // Create mini explosion around button
    for (let i = 0; i < 8; i++) {
      const spark = document.createElement('div');
      spark.style.position = 'absolute';
      spark.style.width = '4px';
      spark.style.height = '4px';
      spark.style.background = '#d4a574';
      spark.style.borderRadius = '50%';
      spark.style.pointerEvents = 'none';
      spark.style.zIndex = '9999';
      
      const rect = button.getBoundingClientRect();
      spark.style.left = (rect.left + rect.width/2) + 'px';
      spark.style.top = (rect.top + rect.height/2) + 'px';
      
      const angle = (i / 8) * Math.PI * 2;
      const distance = 30;
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance;
      
      spark.style.transition = 'all 0.6s ease-out';
      document.body.appendChild(spark);
      
      setTimeout(() => {
        spark.style.transform = `translate(${dx}px, ${dy}px) scale(0)`;
        spark.style.opacity = '0';
      }, 50);
      
      setTimeout(() => {
        if (spark.parentNode) {
          spark.parentNode.removeChild(spark);
        }
      }, 600);
    }
  }
}

// DANCING Text System
class DynamicDancingText {
  constructor() {
    this.init();
  }

  init() {
    // Make all headings dance!
    const headings = document.querySelectorAll('h1, h2, h3, .hero-heading, .section-heading, .card-heading');
    
    headings.forEach((heading, index) => {
      // Split text into spans for individual letter animation
      const text = heading.textContent;
      heading.innerHTML = '';
      heading.classList.add('dynamic-bouncing-text');
      
      [...text].forEach((char, charIndex) => {
        const span = document.createElement('span');
        span.textContent = char === ' ' ? '\u00A0' : char;
        span.style.animationDelay = (charIndex * 0.1) + 's';
        heading.appendChild(span);
      });
      
      // Add different dance styles
      const danceStyles = [
        'dynamic-dance',
        'dynamic-wiggle', 
        'dynamic-wave',
        'dynamic-float-bubble'
      ];
      
      heading.classList.add(danceStyles[index % danceStyles.length]);
    });
  }
}

// CRAZY Card Personalities
class DynamicCardPersonalities {
  constructor() {
    this.init();
  }

  init() {
    const cards = document.querySelectorAll('.professional-card');
    
    cards.forEach((card, index) => {
      // Give each card a unique personality
      const personalities = [
        { main: 'dynamic-jello', hover: 'dynamic-flip' },
        { main: 'dynamic-float-bubble', hover: 'dynamic-elastic' },
        { main: 'dynamic-wave', hover: 'dynamic-zoom-explode' },
        { main: 'dynamic-heartbeat', hover: 'dynamic-spin-dance' }
      ];
      
      const personality = personalities[index % personalities.length];
      card.classList.add(personality.main);
      
      // Add hover personality change
      card.addEventListener('mouseenter', () => {
        card.classList.remove(personality.main);
        card.classList.add(personality.hover);
        this.createCardAura(card);
      });
      
      card.addEventListener('mouseleave', () => {
        card.classList.remove(personality.hover);
        card.classList.add(personality.main);
      });
      
      // Random color flashes
      setInterval(() => {
        if (Math.random() < 0.1) { // 10% chance
          card.classList.add('dynamic-rainbow');
          setTimeout(() => {
            card.classList.remove('dynamic-rainbow');
          }, 1000);
        }
      }, 3000);
    });
  }

  createCardAura(card) {
    const aura = document.createElement('div');
    aura.style.position = 'absolute';
    aura.style.top = '-20px';
    aura.style.left = '-20px';
    aura.style.right = '-20px';
    aura.style.bottom = '-20px';
    aura.style.background = 'radial-gradient(circle, rgba(193, 154, 107, 0.4) 0%, transparent 70%)';
    aura.style.borderRadius = 'inherit';
    aura.style.pointerEvents = 'none';
    aura.style.zIndex = '-1';
    aura.style.animation = 'dynamicPulseEnergy 1s ease-in-out';
    
    card.style.position = 'relative';
    card.appendChild(aura);
    
    setTimeout(() => {
      if (aura.parentNode) {
        aura.parentNode.removeChild(aura);
      }
    }, 1000);
  }
}

// INSANE Navigation Dance
class DynamicNavDance {
  constructor() {
    this.init();
  }

  init() {
    const navItems = document.querySelectorAll('.professional-nav-item');
    
    navItems.forEach((item, index) => {
      // Make nav items dance in sequence
      item.classList.add('dynamic-bounce-crazy');
      item.style.animationDelay = (index * 0.2) + 's';
      
      // Add click explosion
      item.addEventListener('click', () => {
        this.createNavExplosion(item);
        item.classList.add('dynamic-flip');
        setTimeout(() => {
          item.classList.remove('dynamic-flip');
        }, 3000);
      });
    });
  }

  createNavExplosion(navItem) {
    const rect = navItem.getBoundingClientRect();
    
    for (let i = 0; i < 15; i++) {
      const particle = document.createElement('div');
      particle.style.position = 'fixed';
      particle.style.width = '6px';
      particle.style.height = '6px';
      particle.style.background = `hsl(${Math.random() * 60 + 30}, 70%, 60%)`;
      particle.style.borderRadius = '50%';
      particle.style.pointerEvents = 'none';
      particle.style.zIndex = '9999';
      particle.style.left = (rect.left + rect.width/2) + 'px';
      particle.style.top = (rect.top + rect.height/2) + 'px';
      
      const angle = Math.random() * Math.PI * 2;
      const distance = 50 + Math.random() * 100;
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance;
      
      particle.style.transition = 'all 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94)';
      document.body.appendChild(particle);
      
      setTimeout(() => {
        particle.style.transform = `translate(${dx}px, ${dy}px) rotate(720deg) scale(0)`;
        particle.style.opacity = '0';
      }, 50);
      
      setTimeout(() => {
        if (particle.parentNode) {
          particle.parentNode.removeChild(particle);
        }
      }, 1500);
    }
  }
}

// CONTINUOUS Animation Manager
class DynamicContinuousAnimations {
  constructor() {
    this.init();
  }

  init() {
    // Add continuous subtle movements to everything
    setInterval(() => {
      this.addRandomMovements();
    }, 5000);
    
    // Add breathing to the whole page
    document.body.style.animation = 'dynamicPageBreathe 8s ease-in-out infinite';
    
    // Add random sparkles
    setInterval(() => {
      this.createRandomSparkle();
    }, 2000);
  }

  addRandomMovements() {
    const elements = document.querySelectorAll('.professional-card, .professional-btn');
    
    elements.forEach(element => {
      if (Math.random() < 0.3) { // 30% chance
        const movements = ['dynamic-wiggle', 'dynamic-pulse-energy', 'dynamic-heartbeat'];
        const movement = movements[Math.floor(Math.random() * movements.length)];
        
        element.classList.add(movement);
        setTimeout(() => {
          element.classList.remove(movement);
        }, 2000);
      }
    });
  }

  createRandomSparkle() {
    const sparkle = document.createElement('div');
    sparkle.style.position = 'fixed';
    sparkle.style.width = '8px';
    sparkle.style.height = '8px';
    sparkle.style.background = 'radial-gradient(circle, #d4a574, transparent)';
    sparkle.style.borderRadius = '50%';
    sparkle.style.pointerEvents = 'none';
    sparkle.style.zIndex = '9999';
    sparkle.style.left = Math.random() * window.innerWidth + 'px';
    sparkle.style.top = Math.random() * window.innerHeight + 'px';
    sparkle.style.animation = 'dynamicSparkle 2s ease-out forwards';
    
    document.body.appendChild(sparkle);
    
    setTimeout(() => {
      if (sparkle.parentNode) {
        sparkle.parentNode.removeChild(sparkle);
      }
    }, 2000);
  }
}

// Add dynamic CSS animations
const dynamicCSS = document.createElement('style');
dynamicCSS.textContent = `
  @keyframes dynamicPageBreathe {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.002); }
  }
  
  @keyframes dynamicSparkle {
    0% { 
      transform: scale(0) rotate(0deg);
      opacity: 0;
    }
    50% { 
      transform: scale(1) rotate(180deg);
      opacity: 1;
    }
    100% { 
      transform: scale(0) rotate(360deg);
      opacity: 0;
    }
  }
`;
document.head.appendChild(dynamicCSS);

// ===== FRAMER-MOTION STYLE PROFESSIONAL INTERACTIONS =====

// Professional Visual Feedback System
class FramerVisualFeedback {
  constructor() {
    this.init();
  }

  init() {
    // Add feedback to all interactive elements
    const buttons = document.querySelectorAll('button, .professional-btn');
    const cards = document.querySelectorAll('.professional-card');
    
    buttons.forEach(button => {
      this.addButtonFeedback(button);
    });
    
    cards.forEach(card => {
      this.addCardFeedback(card);
    });
  }

  addButtonFeedback(button) {
    button.classList.add('framer-bounce');
    
    button.addEventListener('click', (e) => {
      // Success feedback animation
      button.classList.add('feedback-success');
      this.createCheckmark(button);
      
      setTimeout(() => {
        button.classList.remove('feedback-success');
      }, 600);
    });
    
    // Processing state simulation
    button.addEventListener('mousedown', () => {
      button.classList.add('feedback-processing');
    });
    
    button.addEventListener('mouseup', () => {
      setTimeout(() => {
        button.classList.remove('feedback-processing');
      }, 200);
    });
  }

  addCardFeedback(card) {
    card.classList.add('framer-smooth', 'glass-card', 'elevation-2');
    
    card.addEventListener('mouseenter', () => {
      card.classList.remove('elevation-2');
      card.classList.add('elevation-4');
      this.createProgressRing(card);
    });
    
    card.addEventListener('mouseleave', () => {
      card.classList.remove('elevation-4');
      card.classList.add('elevation-2');
    });
    
    card.addEventListener('click', () => {
      this.createSuccessFeedback(card);
    });
  }

  createCheckmark(element) {
    const checkmark = document.createElement('div');
    checkmark.innerHTML = `
      <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
        <circle class="checkmark-circle" cx="26" cy="26" r="25" fill="none"/>
        <path class="checkmark-check" fill="none" d="m14.1 27.2l7.1 7.2 16.7-16.8"/>
      </svg>
    `;
    checkmark.style.position = 'absolute';
    checkmark.style.top = '50%';
    checkmark.style.left = '50%';
    checkmark.style.transform = 'translate(-50%, -50%)';
    checkmark.style.pointerEvents = 'none';
    checkmark.style.zIndex = '1000';
    
    element.style.position = 'relative';
    element.appendChild(checkmark);
    
    setTimeout(() => {
      if (checkmark.parentNode) {
        checkmark.parentNode.removeChild(checkmark);
      }
    }, 1200);
  }

  createProgressRing(element) {
    const progressRing = document.createElement('div');
    progressRing.innerHTML = `
      <svg class="progress-ring" viewBox="0 0 60 60">
        <circle class="progress-ring-circle"></circle>
        <circle class="progress-ring-progress"></circle>
      </svg>
    `;
    progressRing.style.position = 'absolute';
    progressRing.style.top = '10px';
    progressRing.style.right = '10px';
    progressRing.style.width = '30px';
    progressRing.style.height = '30px';
    progressRing.style.pointerEvents = 'none';
    progressRing.style.zIndex = '10';
    
    element.style.position = 'relative';
    element.appendChild(progressRing);
    
    setTimeout(() => {
      if (progressRing.parentNode) {
        progressRing.parentNode.removeChild(progressRing);
      }
    }, 2000);
  }

  createSuccessFeedback(element) {
    element.classList.add('feedback-success');
    
    // Create badge
    const badge = document.createElement('div');
    badge.className = 'badge badge-success';
    badge.innerHTML = '✓ Selected';
    badge.style.position = 'absolute';
    badge.style.top = '10px';
    badge.style.left = '10px';
    badge.style.zIndex = '10';
    
    element.appendChild(badge);
    
    setTimeout(() => {
      element.classList.remove('feedback-success');
      if (badge.parentNode) {
        badge.parentNode.removeChild(badge);
      }
    }, 2000);
  }
}

// Professional Glassmorphism Manager
class FramerGlassmorphism {
  constructor() {
    this.init();
  }

  init() {
    // Apply glassmorphism to cards
    const cards = document.querySelectorAll('.professional-card');
    const buttons = document.querySelectorAll('button');
    
    cards.forEach(card => {
      card.classList.add('glass-card', 'feature-glow');
    });
    
    buttons.forEach(button => {
      button.classList.add('glass-button');
    });
    
    // Create floating glass elements
    this.createFloatingGlassElements();
  }

  createFloatingGlassElements() {
    for (let i = 0; i < 5; i++) {
      const glassElement = document.createElement('div');
      glassElement.style.position = 'fixed';
      glassElement.style.width = '100px';
      glassElement.style.height = '100px';
      glassElement.style.background = 'rgba(255, 255, 255, 0.05)';
      glassElement.style.backdropFilter = 'blur(10px)';
      glassElement.style.borderRadius = '50%';
      glassElement.style.border = '1px solid rgba(255, 255, 255, 0.1)';
      glassElement.style.pointerEvents = 'none';
      glassElement.style.zIndex = '-1';
      glassElement.style.left = Math.random() * window.innerWidth + 'px';
      glassElement.style.top = Math.random() * window.innerHeight + 'px';
      glassElement.style.animation = `dynamicFloatBubble ${6 + Math.random() * 4}s ease-in-out infinite`;
      glassElement.style.animationDelay = Math.random() * 2 + 's';
      
      document.body.appendChild(glassElement);
    }
  }
}

// Professional Badge & Gamification System
class FramerGamification {
  constructor() {
    this.init();
  }

  init() {
    this.addBadgesToCards();
    this.addProgressIndicators();
    this.addAnimatedIcons();
  }

  addBadgesToCards() {
    const cards = document.querySelectorAll('.professional-card');
    const badgeTypes = ['premium', 'success', 'info', 'warning'];
    const badgeTexts = ['Premium', 'Active', 'New', 'Popular'];
    
    cards.forEach((card, index) => {
      const badgeType = badgeTypes[index % badgeTypes.length];
      const badgeText = badgeTexts[index % badgeTexts.length];
      
      const badge = document.createElement('div');
      badge.className = `badge badge-${badgeType}`;
      badge.textContent = badgeText;
      badge.style.position = 'absolute';
      badge.style.top = '15px';
      badge.style.right = '15px';
      badge.style.zIndex = '10';
      
      card.style.position = 'relative';
      card.appendChild(badge);
    });
  }

  addProgressIndicators() {
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach((button, index) => {
      if (index % 2 === 0) { // Add to every other button
        button.addEventListener('click', () => {
          this.showProgressIndicator(button);
        });
      }
    });
  }

  showProgressIndicator(button) {
    const loadingDots = document.createElement('div');
    loadingDots.className = 'loading-dots';
    loadingDots.innerHTML = '<span></span><span></span><span></span>';
    
    const originalText = button.textContent;
    button.textContent = '';
    button.appendChild(loadingDots);
    button.disabled = true;
    
    setTimeout(() => {
      button.removeChild(loadingDots);
      button.textContent = originalText;
      button.disabled = false;
      
      // Show success state
      button.classList.add('feedback-success');
      setTimeout(() => {
        button.classList.remove('feedback-success');
      }, 600);
    }, 2000);
  }

  addAnimatedIcons() {
    // Add animated icons to headings
    const headings = document.querySelectorAll('.hero-heading, .section-heading, .card-heading');
    const icons = ['*', '^', '!', '+', '*', '>>'];
    
    headings.forEach((heading, index) => {
      const icon = document.createElement('span');
      icon.textContent = icons[index % icons.length];
      icon.className = 'icon-bounce';
      icon.style.marginRight = '10px';
      icon.style.display = 'inline-block';
      
      heading.insertBefore(icon, heading.firstChild);
    });
  }
}

// Professional Loading States Manager
class FramerLoadingStates {
  constructor() {
    this.init();
  }

  init() {
    this.addSkeletonLoaders();
    this.addPageTransitions();
  }

  addSkeletonLoaders() {
    // Create skeleton loaders for cards during loading
    const cards = document.querySelectorAll('.professional-card');
    
    cards.forEach(card => {
      const skeleton = this.createSkeleton();
      card.appendChild(skeleton);
      
      // Remove skeleton after content loads
      setTimeout(() => {
        skeleton.style.opacity = '0';
        setTimeout(() => {
          if (skeleton.parentNode) {
            skeleton.parentNode.removeChild(skeleton);
          }
        }, 300);
      }, 1000 + Math.random() * 1000);
    });
  }

  createSkeleton() {
    const skeleton = document.createElement('div');
    skeleton.style.position = 'absolute';
    skeleton.style.top = '0';
    skeleton.style.left = '0';
    skeleton.style.right = '0';
    skeleton.style.bottom = '0';
    skeleton.style.background = 'rgba(255, 255, 255, 0.1)';
    skeleton.style.borderRadius = 'inherit';
    skeleton.style.zIndex = '5';
    skeleton.style.transition = 'opacity 0.3s ease';
    
    // Add skeleton content
    const skeletonContent = document.createElement('div');
    skeletonContent.innerHTML = `
      <div class="loading-skeleton" style="height: 20px; margin: 20px; margin-bottom: 10px;"></div>
      <div class="loading-skeleton" style="height: 15px; margin: 20px; width: 70%;"></div>
      <div class="loading-skeleton" style="height: 15px; margin: 20px; width: 50%;"></div>
    `;
    
    skeleton.appendChild(skeletonContent);
    return skeleton;
  }

  addPageTransitions() {
    // Add smooth page entrance
    document.body.style.opacity = '0';
    document.body.style.transform = 'translateY(20px)';
    document.body.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
    
    setTimeout(() => {
      document.body.style.opacity = '1';
      document.body.style.transform = 'translateY(0)';
    }, 100);
  }
}

// Professional Microinteractions Manager
class FramerMicrointeractions {
  constructor() {
    this.init();
  }

  init() {
    this.addHoverMicrointeractions();
    this.addClickMicrointeractions();
    this.addScrollMicrointeractions();
  }

  addHoverMicrointeractions() {
    const interactiveElements = document.querySelectorAll('button, .professional-card, .professional-nav-item');
    
    interactiveElements.forEach(element => {
      element.addEventListener('mouseenter', () => {
        this.createHoverEffect(element);
      });
    });
  }

  createHoverEffect(element) {
    // Create subtle glow effect
    const glow = document.createElement('div');
    glow.style.position = 'absolute';
    glow.style.top = '-5px';
    glow.style.left = '-5px';
    glow.style.right = '-5px';
    glow.style.bottom = '-5px';
    glow.style.background = 'radial-gradient(circle, rgba(212, 165, 116, 0.2) 0%, transparent 70%)';
    glow.style.borderRadius = 'inherit';
    glow.style.pointerEvents = 'none';
    glow.style.zIndex = '-1';
    glow.style.opacity = '0';
    glow.style.transition = 'opacity 0.3s ease';
    glow.className = 'hover-glow-microinteraction';
    
    element.style.position = 'relative';
    element.appendChild(glow);
    
    setTimeout(() => {
      glow.style.opacity = '1';
    }, 50);
    
    element.addEventListener('mouseleave', () => {
      glow.style.opacity = '0';
      setTimeout(() => {
        if (glow.parentNode) {
          glow.parentNode.removeChild(glow);
        }
      }, 300);
    }, { once: true });
  }

  addClickMicrointeractions() {
    const clickableElements = document.querySelectorAll('button, .professional-card');
    
    clickableElements.forEach(element => {
      element.addEventListener('click', (e) => {
        this.createClickRipple(e, element);
      });
    });
  }

  createClickRipple(event, element) {
    const ripple = document.createElement('div');
    const rect = element.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height) * 2;
    const x = event.clientX - rect.left - size / 2;
    const y = event.clientY - rect.top - size / 2;
    
    ripple.style.width = ripple.style.height = size + 'px';
    ripple.style.left = x + 'px';
    ripple.style.top = y + 'px';
    ripple.style.position = 'absolute';
    ripple.style.borderRadius = '50%';
    ripple.style.background = 'rgba(212, 165, 116, 0.3)';
    ripple.style.transform = 'scale(0)';
    ripple.style.animation = 'ripple 0.6s linear';
    ripple.style.pointerEvents = 'none';
    ripple.style.zIndex = '1';
    
    element.style.position = 'relative';
    element.style.overflow = 'hidden';
    element.appendChild(ripple);
    
    setTimeout(() => {
      if (ripple.parentNode) {
        ripple.parentNode.removeChild(ripple);
      }
    }, 600);
  }

  addScrollMicrointeractions() {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('framer-elastic');
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0) scale(1)';
          
          // Add staggered animation delay
          const delay = Array.from(entry.target.parentNode.children).indexOf(entry.target) * 100;
          entry.target.style.transitionDelay = delay + 'ms';
        }
      });
    }, {
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px'
    });

    document.querySelectorAll('.professional-card, .professional-btn').forEach(el => {
      el.style.opacity = '0';
      el.style.transform = 'translateY(30px) scale(0.95)';
      el.style.transition = 'all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55)';
      observer.observe(el);
    });
  }
}

// ===== PROFESSIONAL ENTERPRISE-LEVEL INTERACTIONS =====

// Professional Smooth Interactions
class ProfessionalInteractions {
  constructor() {
    this.init();
  }

  init() {
    this.addSmoothHoverEffects();
    this.addProfessionalFeedback();
    this.addSubtleAnimations();
  }

  addSmoothHoverEffects() {
    // All hover effects are now handled by CSS classes
    // This keeps the code clean and performant
    console.log('Professional hover effects loaded via CSS');
  }

  addProfessionalFeedback() {
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach(button => {
      button.addEventListener('click', () => {
        button.classList.add('feedback-success');
        setTimeout(() => {
          button.classList.remove('feedback-success');
        }, 300);
      });
    });
  }

  addSubtleAnimations() {
    // Optimized Intersection Observer with better performance
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Use CSS class instead of inline styles for better performance
          entry.target.classList.add('visible');
          // Unobserve after animation to save resources
          observer.unobserve(entry.target);
        }
      });
    }, { 
      threshold: 0.1,
      rootMargin: '0px 0px -50px 0px' // Start animation before element is fully visible
    });

    // Apply to all fade-in elements across all pages
    document.querySelectorAll('.fade-in').forEach(el => {
      observer.observe(el);
    });

    // Also observe any new elements that get added dynamically
    const mutationObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) { // Element node
            const fadeElements = node.querySelectorAll ? node.querySelectorAll('.fade-in') : [];
            fadeElements.forEach(el => observer.observe(el));
            
            if (node.classList && node.classList.contains('fade-in')) {
              observer.observe(node);
            }
          }
        });
      });
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
}

// Professional Loading States
class ProfessionalLoading {
  constructor() {
    this.init();
  }

  init() {
    // Add smooth page entrance
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.4s ease';
    
    setTimeout(() => {
      document.body.style.opacity = '1';
    }, 100);
  }
}

// OPTIMIZED Dynamic Effects with Performance Boost
class DynamicEffects {
  constructor() {
    this.isRunning = false;
    this.particles = [];
    this.maxParticles = 15; // Limit particles for performance
    this.init();
  }

  init() {
    this.createOptimizedCursor();
    this.addOptimizedRipples();
    this.addMagneticEffects();
    this.addParticleTrail();
    this.addDynamicClasses();
    this.optimizeAnimations();
  }

  createOptimizedCursor() {
    // Create optimized cursor with throttling
    const cursor = document.createElement('div');
    cursor.className = 'dynamic-cursor';
    document.body.appendChild(cursor);

    let mouseX = 0, mouseY = 0;
    let cursorX = 0, cursorY = 0;
    let animationId;

    // Optimized cursor following with requestAnimationFrame
    const updateCursor = () => {
      const deltaX = mouseX - cursorX;
      const deltaY = mouseY - cursorY;
      
      // Only update if movement is significant (performance optimization)
      if (Math.abs(deltaX) > 0.1 || Math.abs(deltaY) > 0.1) {
        cursorX += deltaX * 0.15;
        cursorY += deltaY * 0.15;
        cursor.style.transform = `translate3d(${cursorX}px, ${cursorY}px, 0)`;
      }
      
      animationId = requestAnimationFrame(updateCursor);
    };

    // Throttled mouse move event
    let mouseMoveTimeout;
    document.addEventListener('mousemove', (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
      
      if (!this.isRunning) {
        this.isRunning = true;
        updateCursor();
      }
    }, { passive: true });

    // Optimized hover effects
    document.querySelectorAll('button, .professional-card, .cool-icon').forEach(el => {
      el.addEventListener('mouseenter', () => cursor.classList.add('hover'), { passive: true });
      el.addEventListener('mouseleave', () => cursor.classList.remove('hover'), { passive: true });
    });
  }

  addOptimizedRipples() {
    // Optimized ripple effects with object pooling
    const ripplePool = [];
    const maxRipples = 5;

    const createRipple = (x, y) => {
      let ripple = ripplePool.pop();
      
      if (!ripple) {
        ripple = document.createElement('div');
        ripple.className = 'dynamic-ripple';
        document.body.appendChild(ripple);
      }
      
      ripple.style.left = x - 25 + 'px';
      ripple.style.top = y - 25 + 'px';
      ripple.style.width = '50px';
      ripple.style.height = '50px';
      ripple.style.opacity = '1';
      ripple.style.transform = 'scale(0)';
      
      // Animate with CSS
      requestAnimationFrame(() => {
        ripple.style.transform = 'scale(4)';
        ripple.style.opacity = '0';
      });
      
      setTimeout(() => {
        if (ripplePool.length < maxRipples) {
          ripplePool.push(ripple);
        } else {
          ripple.remove();
        }
      }, 600);
    };

    document.addEventListener('click', (e) => {
      createRipple(e.clientX, e.clientY);
    }, { passive: true });
  }

  addMagneticEffects() {
    // Add magnetic attraction to buttons and cards
    document.querySelectorAll('button, .professional-card').forEach(el => {
      el.classList.add('magnetic-button', 'cool-glow');
      
      el.addEventListener('mousemove', (e) => {
        const rect = el.getBoundingClientRect();
        const x = e.clientX - rect.left - rect.width / 2;
        const y = e.clientY - rect.top - rect.height / 2;
        
        // Subtle magnetic effect
        const moveX = x * 0.1;
        const moveY = y * 0.1;
        
        el.style.transform = `translate3d(${moveX}px, ${moveY}px, 0) scale(1.02)`;
      }, { passive: true });
      
      el.addEventListener('mouseleave', () => {
        el.style.transform = 'translate3d(0, 0, 0) scale(1)';
      }, { passive: true });
    });
  }

  addParticleTrail() {
    // Optimized particle trail system
    const trailContainer = document.createElement('div');
    trailContainer.className = 'particle-trail';
    document.body.appendChild(trailContainer);

    let lastParticleTime = 0;
    const particleInterval = 100; // Limit particle creation

    document.addEventListener('mousemove', (e) => {
      const now = Date.now();
      
      if (now - lastParticleTime > particleInterval && this.particles.length < this.maxParticles) {
        this.createParticle(e.clientX, e.clientY, trailContainer);
        lastParticleTime = now;
      }
    }, { passive: true });
  }

  createParticle(x, y, container) {
    const particle = document.createElement('div');
    particle.className = 'particle';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    
    container.appendChild(particle);
    this.particles.push(particle);
    
    setTimeout(() => {
      particle.remove();
      this.particles = this.particles.filter(p => p !== particle);
    }, 1000);
  }

  addMouseAttraction() {
    const orbs = document.querySelectorAll('.beautiful-orb');
    
    document.addEventListener('mousemove', (e) => {
      orbs.forEach(orb => {
        const rect = orb.getBoundingClientRect();
        const orbX = rect.left + rect.width / 2;
        const orbY = rect.top + rect.height / 2;
        
        const distance = Math.sqrt(
          Math.pow(e.clientX - orbX, 2) + Math.pow(e.clientY - orbY, 2)
        );
        
        if (distance < 200) {
          orb.classList.add('mouse-attracted');
        } else {
          orb.classList.remove('mouse-attracted');
        }
      });
    });
  }

  addDynamicClasses() {
    // Add dynamic classes to elements with performance optimization
    requestAnimationFrame(() => {
      document.querySelectorAll('.cool-icon svg').forEach(icon => {
        icon.classList.add('dynamic-icon');
      });

      document.querySelectorAll('.section-heading').forEach(heading => {
        heading.classList.add('dynamic-text');
      });

      document.querySelectorAll('.professional-card').forEach((card, index) => {
        if (index % 4 === 0) { // Reduced frequency for better performance
          card.classList.add('dynamic-card');
        }
        card.classList.add('section-reveal');
      });
    });
  }

  optimizeAnimations() {
    // Pause animations when tab is not visible (performance boost)
    document.addEventListener('visibilitychange', () => {
      const isHidden = document.hidden;
      document.querySelectorAll('.beautiful-orb, .dynamic-card').forEach(el => {
        el.style.animationPlayState = isHidden ? 'paused' : 'running';
      });
    });

    // Reduce motion for users who prefer it
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      document.documentElement.style.setProperty('--animation-duration', '0.01s');
    }

    // Add page entrance animation
    document.body.classList.add('page-enter');
  }
}

// BEAUTIFUL Notification System
class NotificationSystem {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceNotifications();
  }

  enhanceNotifications() {
    // Auto-detect notification types and apply appropriate styling
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.classList && node.classList.contains('notification-toast')) {
            this.enhanceNotification(node);
          }
        });
      });
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }

  enhanceNotification(notification) {
    const text = notification.textContent.toLowerCase();
    
    // Remove existing type classes
    notification.classList.remove('notification-success', 'notification-error', 'notification-warning', 'notification-info');
    
    // Apply appropriate type based on content
    if (text.includes('success') || text.includes('completed') || text.includes('done') || text.includes('finished')) {
      notification.classList.add('notification-success');
      this.updateIcon(notification, 'success');
    } else if (text.includes('error') || text.includes('failed') || text.includes('problem')) {
      notification.classList.add('notification-error');
      this.updateIcon(notification, 'error');
    } else if (text.includes('warning') || text.includes('careful') || text.includes('attention')) {
      notification.classList.add('notification-warning');
      this.updateIcon(notification, 'warning');
    } else if (text.includes('analyzing') || text.includes('processing') || text.includes('loading')) {
      notification.classList.add('notification-info');
      this.updateIcon(notification, 'loading');
    } else {
      notification.classList.add('notification-info');
      this.updateIcon(notification, 'info');
    }
  }

  updateIcon(notification, type) {
    const iconElement = notification.querySelector('.notification-icon svg');
    if (!iconElement) return;

    const icons = {
      success: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>',
      error: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>',
      warning: '<path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/>',
      loading: '<path d="M12 6v3l4-4-4-4v3c-4.42 0-8 3.58-8 8 0 1.57.46 3.03 1.24 4.26L6.7 14.8c-.45-.83-.7-1.79-.7-2.8 0-3.31 2.69-6 6-6zm6.76 1.74L17.3 9.2c.44.84.7 1.79.7 2.8 0 3.31-2.69 6-6 6v-3l-4 4 4 4v-3c4.42 0 8-3.58 8-8 0-1.57-.46-3.03-1.24-4.26z"/>',
      info: '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/>'
    };

    iconElement.innerHTML = icons[type] || icons.info;
  }
}

// AMAZING Icon Animation System
class IconAnimationSystem {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceAllIcons();
    this.addSpecialIconEffects();
    this.watchForNewIcons();
  }

  enhanceAllIcons() {
    // Auto-detect and enhance all icons
    setTimeout(() => {
      this.applyIconEnhancements();
    }, 500);
  }

  applyIconEnhancements() {
    // Enhance SVG icons
    const svgIcons = document.querySelectorAll('svg');
    svgIcons.forEach((icon, index) => {
      if (!icon.classList.contains('icon-enhanced')) {
        icon.classList.add('icon-enhanced');
        this.categorizeAndEnhanceIcon(icon);
      }
    });

    // Enhance icon containers
    const iconContainers = document.querySelectorAll('.cool-icon, .hero-feature-icon, .hero-stat-icon, .chat-avatar, .document-avatar');
    iconContainers.forEach(container => {
      if (!container.classList.contains('container-enhanced')) {
        container.classList.add('container-enhanced');
        this.addContainerEffects(container);
      }
    });
  }

  categorizeAndEnhanceIcon(icon) {
    const iconPath = icon.querySelector('path')?.getAttribute('d') || '';
    const viewBox = icon.getAttribute('viewBox') || '';
    
    // Add specific classes based on icon type
    if (iconPath.includes('M15.5 14') || iconPath.includes('search')) {
      icon.classList.add('search-icon');
    } else if (iconPath.includes('M6 18L18 6') || iconPath.includes('M6 6l12 12')) {
      icon.classList.add('close-icon');
    } else if (iconPath.includes('M9 5l7 7') || iconPath.includes('arrow')) {
      icon.classList.add('arrow-icon');
    } else if (iconPath.includes('L13.09 8.26') || iconPath.includes('star')) {
      icon.classList.add('star-icon');
    } else if (iconPath.includes('M10 4H4C2.89') || iconPath.includes('folder')) {
      icon.classList.add('folder-icon');
    } else if (iconPath.includes('M14 2H6C4.9') || iconPath.includes('document')) {
      icon.classList.add('document-icon');
    } else if (viewBox.includes('24 24')) {
      icon.classList.add('general-icon');
    }

    // Add interactive class for clickable icons
    if (icon.closest('button') || icon.closest('[onclick]') || icon.closest('a')) {
      icon.classList.add('interactive-icon');
    }
  }

  addContainerEffects(container) {
    // Add hover effects to icon containers
    container.addEventListener('mouseenter', () => {
      const icon = container.querySelector('svg');
      if (icon) {
        icon.style.transform = 'scale(1.2) rotate(10deg)';
        icon.style.filter = 'drop-shadow(0 6px 15px rgba(139, 115, 85, 0.4))';
      }
    });

    container.addEventListener('mouseleave', () => {
      const icon = container.querySelector('svg');
      if (icon) {
        icon.style.transform = '';
        icon.style.filter = '';
      }
    });
  }

  addSpecialIconEffects() {
    // Add special effects for hero icons
    this.enhanceHeroIcons();
    
    // Add notification icon effects
    this.enhanceNotificationIcons();
    
    // Add chat avatar effects
    this.enhanceChatAvatars();
  }

  enhanceHeroIcons() {
    const heroIcons = document.querySelectorAll('.hero-feature-icon svg, .hero-stat-icon svg');
    heroIcons.forEach(icon => {
      icon.addEventListener('mouseenter', () => {
        icon.style.animation = 'heroIconMagic 1s ease-in-out';
      });

      icon.addEventListener('animationend', () => {
        icon.style.animation = '';
      });
    });
  }

  enhanceNotificationIcons() {
    const notificationIcons = document.querySelectorAll('.notification-icon svg');
    notificationIcons.forEach(icon => {
      icon.addEventListener('mouseenter', () => {
        icon.style.animation = 'notificationSpin 1s ease-in-out';
      });

      icon.addEventListener('animationend', () => {
        icon.style.animation = '';
      });
    });
  }

  enhanceChatAvatars() {
    const avatars = document.querySelectorAll('.chat-avatar, .document-avatar');
    avatars.forEach(avatar => {
      avatar.addEventListener('mouseenter', () => {
        avatar.style.animation = 'avatarSpin 1.2s ease-in-out';
      });

      avatar.addEventListener('animationend', () => {
        avatar.style.animation = '';
      });
    });
  }

  watchForNewIcons() {
    // Watch for dynamically added icons
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === 1) { // Element node
            // Check if the node itself is an SVG
            if (node.tagName === 'svg') {
              this.categorizeAndEnhanceIcon(node);
            }
            
            // Check for SVGs within the added node
            const svgs = node.querySelectorAll && node.querySelectorAll('svg');
            if (svgs) {
              svgs.forEach(svg => this.categorizeAndEnhanceIcon(svg));
            }
          }
        });
      });
    });

    observer.observe(document.body, { childList: true, subtree: true });
  }
}

// BEAUTIFUL Chat Enhancement System
class ChatEnhancementSystem {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceChatElements();
    this.addChatAnimations();
    this.addScrollToBottom();
  }

  enhanceChatElements() {
    // Auto-detect and enhance chat elements
    setTimeout(() => {
      this.applyChatEnhancements();
    }, 500);

    // Watch for new messages
    const observer = new MutationObserver(() => {
      this.applyChatEnhancements();
    });

    const chatContainer = document.querySelector('.chat-messages');
    if (chatContainer) {
      observer.observe(chatContainer, { childList: true, subtree: true });
    }
  }

  applyChatEnhancements() {
    // Add message entrance animations
    const messages = document.querySelectorAll('.message-wrapper');
    messages.forEach((message, index) => {
      if (!message.classList.contains('enhanced')) {
        message.classList.add('enhanced');
        message.style.animationDelay = `${index * 100}ms`;
      }
    });

    // Add hover effects to avatars
    const avatars = document.querySelectorAll('.chat-avatar');
    avatars.forEach(avatar => {
      if (!avatar.classList.contains('enhanced')) {
        avatar.classList.add('enhanced');
        this.addAvatarEffects(avatar);
      }
    });

    // Add message bubble effects
    const bubbles = document.querySelectorAll('.user-message-bubble, .assistant-message-bubble');
    bubbles.forEach(bubble => {
      if (!bubble.classList.contains('enhanced')) {
        bubble.classList.add('enhanced');
        this.addBubbleEffects(bubble);
      }
    });
  }

  addAvatarEffects(avatar) {
    avatar.addEventListener('mouseenter', () => {
      avatar.style.transform = 'scale(1.15) rotate(10deg)';
      avatar.style.boxShadow = '0 8px 25px rgba(139, 115, 85, 0.4)';
    });

    avatar.addEventListener('mouseleave', () => {
      avatar.style.transform = 'scale(1)';
      avatar.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
    });
  }

  addBubbleEffects(bubble) {
    bubble.addEventListener('mouseenter', () => {
      bubble.style.transform = 'translateY(-3px) scale(1.03)';
    });

    bubble.addEventListener('mouseleave', () => {
      bubble.style.transform = 'translateY(0) scale(1)';
    });
  }

  addChatAnimations() {
    // Add typing indicator animation
    const typingDots = document.querySelectorAll('.typing-dot');
    typingDots.forEach((dot, index) => {
      dot.style.animationDelay = `${index * 200}ms`;
    });
  }

  addScrollToBottom() {
    // Auto-scroll to bottom when new messages arrive
    const chatMessages = document.querySelector('.chat-messages');
    if (chatMessages) {
      const observer = new MutationObserver(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
      });

      observer.observe(chatMessages, { childList: true });
    }
  }
}

// AMAZING Homepage Effects System
class HomepageEffects {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceHomepageElements();
    this.addCounterAnimations();
    this.addStaggeredAnimations();
  }

  enhanceHomepageElements() {
    // Auto-detect and enhance homepage elements with multiple attempts
    console.log('Starting homepage enhancement...'); // Debug log
    
    // Try immediately
    this.applyEnhancements();
    
    // Try after 1 second
    setTimeout(() => {
      console.log('Trying homepage enhancement after 1s...'); // Debug log
      this.applyEnhancements();
    }, 1000);
    
    // Try after 3 seconds
    setTimeout(() => {
      console.log('Trying homepage enhancement after 3s...'); // Debug log
      this.applyEnhancements();
    }, 3000);
  }

  applyEnhancements() {
    this.enhanceStatCards();
    this.enhanceFeatureCards();
    this.enhanceButtons();
  }

  enhanceStatCards() {
    // Find stat cards and numbers - more specific detection
    const statElements = document.querySelectorAll('.professional-card.dark-card');
    
    statElements.forEach((card, index) => {
      const cardText = card.textContent.toLowerCase();
      
      // Check if this is a stat card (contains specific stat keywords)
      if (cardText.includes('document proposals') || 
          cardText.includes('drive files') || 
          cardText.includes('duplicate groups') || 
          cardText.includes('categories')) {
        
        console.log('Found stat card:', cardText); // Debug log
        card.classList.add('hero-stat-card');
        
        // Add icon container
        const icon = card.querySelector('.cool-icon');
        if (icon) {
          icon.classList.add('hero-stat-icon');
        }
        
        // Find and enhance the large numbers (text-3xl font-bold)
        const numberElements = card.querySelectorAll('.text-3xl');
        numberElements.forEach(el => {
          el.classList.add('hero-stat-number');
          console.log('Added hero-stat-number to:', el.textContent); // Debug log
        });
        
        // Find and enhance the labels (card-description)
        const labelElements = card.querySelectorAll('.card-description');
        labelElements.forEach(el => {
          el.classList.add('hero-stat-label');
          console.log('Added hero-stat-label to:', el.textContent); // Debug log
        });
      }
    });
  }

  enhanceFeatureCards() {
    // Find feature description cards - more specific detection
    const featureSelectors = [
      'Smart Organization',
      'Intelligent Search', 
      'Lightning Fast'
    ];
    
    document.querySelectorAll('.professional-card.dark-card').forEach(card => {
      const cardText = card.textContent;
      
      featureSelectors.forEach(feature => {
        if (cardText.includes(feature)) {
          console.log('Found feature card:', feature); // Debug log
          card.classList.add('hero-feature-card');
          
          // Add icon enhancement
          const icon = card.querySelector('.cool-icon');
          if (icon) {
            icon.classList.add('hero-feature-icon');
          }
          
          // Enhance titles (h3 with card-heading class)
          const headings = card.querySelectorAll('h3.card-heading');
          headings.forEach(h => {
            h.classList.add('hero-feature-title');
            console.log('Added hero-feature-title to:', h.textContent); // Debug log
          });
          
          // Enhance descriptions (p with card-description class)
          const descriptions = card.querySelectorAll('p.card-description');
          descriptions.forEach(p => {
            p.classList.add('hero-feature-description');
            console.log('Added hero-feature-description to:', p.textContent); // Debug log
          });
        }
      });
    });
  }

  enhanceButtons() {
    // Find and enhance CTA buttons - more specific detection
    const buttons = document.querySelectorAll('button');
    
    buttons.forEach(button => {
      const buttonText = button.textContent.toLowerCase();
      
      if (buttonText.includes('start smart scan')) {
        console.log('Found CTA button:', buttonText); // Debug log
        button.classList.add('hero-cta-button');
      }
    });
  }

  addCounterAnimations() {
    // Animate numbers counting up
    const numberElements = document.querySelectorAll('.hero-stat-number');
    
    numberElements.forEach(el => {
      const finalNumber = parseInt(el.textContent) || 0;
      if (finalNumber > 0) {
        this.animateCounter(el, 0, finalNumber, 1500);
      }
    });
  }

  animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    
    const updateCounter = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const currentNumber = Math.floor(start + (end - start) * easeOutQuart);
      
      element.textContent = currentNumber;
      
      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      } else {
        element.textContent = end;
      }
    };
    
    requestAnimationFrame(updateCounter);
  }

  addStaggeredAnimations() {
    // Add staggered entrance animations
    const statCards = document.querySelectorAll('.hero-stat-card');
    const featureCards = document.querySelectorAll('.hero-feature-card');
    
    // Create containers for proper grid layout
    if (statCards.length > 0) {
      const statsContainer = this.createContainer(statCards, 'hero-stats-container');
    }
    
    if (featureCards.length > 0) {
      const featuresContainer = this.createContainer(featureCards, 'hero-features-container');
    }
  }

  createContainer(elements, className) {
    if (elements.length === 0) return;
    
    const firstElement = elements[0];
    const parent = firstElement.parentNode;
    
    // Create container
    const container = document.createElement('div');
    container.className = className;
    
    // Move elements to container
    elements.forEach(el => {
      container.appendChild(el);
    });
    
    // Insert container
    parent.appendChild(container);
    
    return container;
  }

  isStatCard(text) {
    const statKeywords = [
      'document proposals',
      'drive files', 
      'duplicate groups',
      'categories',
      'proposals',
      'files',
      'duplicates'
    ];
    
    return statKeywords.some(keyword => text.includes(keyword.toLowerCase()));
  }

  isNumber(text) {
    const trimmed = text.trim();
    return /^\d+$/.test(trimmed) && trimmed.length <= 3;
  }

  isStatLabel(text) {
    const labels = [
      'document proposals',
      'drive files',
      'duplicate groups', 
      'categories'
    ];
    
    return labels.some(label => 
      text.toLowerCase().includes(label) && 
      text.length < 50
    );
  }
}

// OPTIMIZED Performance Initialization
class PerformanceOptimizer {
  constructor() {
    this.init();
  }

  init() {
    // Optimize scroll performance
    this.optimizeScrolling();
    
    // Debounce resize events
    this.optimizeResize();
    
    // Preload critical animations
    this.preloadAnimations();
  }

  optimizeScrolling() {
    let ticking = false;
    
    const updateScrollEffects = () => {
      // Batch DOM reads and writes
      requestAnimationFrame(() => {
        ticking = false;
      });
    };

    window.addEventListener('scroll', () => {
      if (!ticking) {
        requestAnimationFrame(updateScrollEffects);
        ticking = true;
      }
    }, { passive: true });
  }

  optimizeResize() {
    let resizeTimeout;
    
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(() => {
        // Recalculate layouts after resize
        document.querySelectorAll('.fade-in.visible').forEach(el => {
          el.style.transform = 'translateY(0) translateZ(0)';
        });
      }, 150);
    }, { passive: true });
  }

  preloadAnimations() {
    // Force browser to prepare animation properties
    const testEl = document.createElement('div');
    testEl.style.transform = 'translateZ(0)';
    testEl.style.backfaceVisibility = 'hidden';
    testEl.style.perspective = '1000px';
    document.body.appendChild(testEl);
    
    // Remove after a frame
    requestAnimationFrame(() => {
      document.body.removeChild(testEl);
    });
  }
}

// Professional Effects Class for enhanced UI interactions
class ProfessionalEffects {
  constructor() {
    this.init();
  }

  init() {
    this.enhanceButtons();
    this.enhanceCards();
    this.addHoverEffects();
  }

  enhanceButtons() {
    const buttons = document.querySelectorAll('button, .professional-button');
    buttons.forEach(button => {
      if (!button.classList.contains('effects-enhanced')) {
        button.classList.add('effects-enhanced');
        this.addButtonEffects(button);
      }
    });
  }

  enhanceCards() {
    const cards = document.querySelectorAll('.professional-card, .glass-card');
    cards.forEach(card => {
      if (!card.classList.contains('effects-enhanced')) {
        card.classList.add('effects-enhanced');
        this.addCardEffects(card);
      }
    });
  }

  addButtonEffects(button) {
    button.addEventListener('mouseenter', () => {
      button.style.transform = 'translateY(-2px) scale(1.02)';
      button.style.boxShadow = '0 8px 24px rgba(45, 36, 22, 0.16)';
    });

    button.addEventListener('mouseleave', () => {
      button.style.transform = '';
      button.style.boxShadow = '';
    });
  }

  addCardEffects(card) {
    card.addEventListener('mouseenter', () => {
      card.style.transform = 'translateY(-2px) scale(1.01)';
      card.style.boxShadow = '0 8px 24px rgba(45, 36, 22, 0.16)';
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
      card.style.boxShadow = '';
    });
  }

  addHoverEffects() {
    // Add subtle hover effects to interactive elements
    const interactiveElements = document.querySelectorAll('input, textarea, select');
    interactiveElements.forEach(element => {
      if (!element.classList.contains('effects-enhanced')) {
        element.classList.add('effects-enhanced');
        element.addEventListener('focus', () => {
          element.style.transform = 'translateY(-1px)';
        });
        element.addEventListener('blur', () => {
          element.style.transform = '';
        });
      }
    });
  }
}

// Initialize all professional effects when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new PerformanceOptimizer();
  new ProfessionalEffects();
  new ProfessionalLoading();
  new DynamicEffects(); // Add dynamic interactive effects
  new NotificationSystem(); // Add beautiful notification system
  window.homepageEffects = new HomepageEffects(); // Add amazing homepage glowing effects
  window.chatEnhancement = new ChatEnhancementSystem(); // Add beautiful chat enhancements
  window.iconAnimations = new IconAnimationSystem(); // Add amazing icon animations
});

// Manual trigger function for testing
window.applyHomepageEffects = () => {
  console.log('Manually applying homepage effects...');
  if (window.homepageEffects) {
    window.homepageEffects.applyEnhancements();
  } else {
    window.homepageEffects = new HomepageEffects();
  }
};

} // End of ClarifileInteractions namespace