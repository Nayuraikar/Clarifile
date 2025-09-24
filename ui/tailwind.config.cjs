/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        /* New elegant warm color palette */
        primary: {
          50: '#F5E7DE',
          100: '#F2E6DC',
          200: '#EAD5C4',
          300: '#E2C4AC',
          400: '#DAB394',
          500: '#C4B5AC',
          600: '#9A8F87',
          700: '#7A716A',
          800: '#5A544D',
          900: '#3A3632',
        },
        accent: {
          400: '#F2C4A8',
          500: '#F2BFA4',
          600: '#E8A884',
          700: '#DE9164',
        },
        warm: {
          400: '#D4A574',
          500: '#C19660',
          600: '#AE874C',
          700: '#9C7838',
        },
        success: {
          500: '#8FBC8F',
          600: '#7AA67A',
          700: '#659065',
        },
        /* Legacy colors for backward compatibility */
        sand50: '#F5E7DE',
        sand200: '#EAD5C4',
        charcoal900: '#1E1A17',
        mink500: '#9A8F87',
        pearl400: '#C4B5AC',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Roboto', 'Arial', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'fade-in-up': 'fadeInUp 0.6s ease-out',
        'fade-in-down': 'fadeInDown 0.6s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'slide-left': 'slideLeft 0.4s ease-out',
        'slide-right': 'slideRight 0.4s ease-out',
        'bounce-gentle': 'bounceGentle 0.8s ease-in-out',
        'pulse-slow': 'pulseSlow 3s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'float': 'float 3s ease-in-out infinite',
        'shimmer': 'shimmer 2s ease-in-out infinite',
        'scale-in': 'scaleIn 0.4s ease-out',
        'rotate-in': 'rotateIn 0.5s ease-out',
        'float-particle': 'floatParticle 20s infinite linear',
        'geometric-float': 'geometricFloat 25s infinite ease-in-out',
        'wave-motion': 'waveMotion 8s ease-in-out infinite',
        'grid-move': 'gridMove 20s linear infinite',
        'aurora-flow': 'auroraFlow 15s ease-in-out infinite',
        'beam-sweep': 'beamSweep 12s ease-in-out infinite',
        'trail-fade': 'trailFade 1s ease-out forwards',
        'gradient-shift': 'gradientShift 4s ease-in-out infinite',
        'glow-pulse': 'glowPulse 2s ease-in-out infinite alternate',
        'particle-float': 'particleFloat 20s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeInDown: {
          '0%': { opacity: '0', transform: 'translateY(-30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideLeft: {
          '0%': { opacity: '0', transform: 'translateX(20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        slideRight: {
          '0%': { opacity: '0', transform: 'translateX(-20px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        bounceGentle: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        pulseSlow: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.7' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(242, 191, 164, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(242, 191, 164, 0.8)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        scaleIn: {
          '0%': { opacity: '0', transform: 'scale(0.8)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        rotateIn: {
          '0%': { opacity: '0', transform: 'rotate(-180deg)' },
          '100%': { opacity: '1', transform: 'rotate(0deg)' },
        },
        floatParticle: {
          '0%': { transform: 'translateY(100vh) rotate(0deg)', opacity: '0' },
          '10%': { opacity: '0.6' },
          '90%': { opacity: '0.6' },
          '100%': { transform: 'translateY(-100vh) rotate(360deg)', opacity: '0' },
        },
        geometricFloat: {
          '0%, 100%': { transform: 'translate(0, 0) rotate(0deg) scale(1)', opacity: '0.3' },
          '25%': { transform: 'translate(100px, -50px) rotate(90deg) scale(1.1)', opacity: '0.5' },
          '50%': { transform: 'translate(-50px, 100px) rotate(180deg) scale(0.9)', opacity: '0.4' },
          '75%': { transform: 'translate(50px, 50px) rotate(270deg) scale(1.05)', opacity: '0.6' },
        },
        waveMotion: {
          '0%, 100%': { transform: 'translateX(0) translateY(0)' },
          '50%': { transform: 'translateX(-50px) translateY(-20px)' },
        },
        gridMove: {
          '0%': { transform: 'translate(0, 0)' },
          '100%': { transform: 'translate(50px, 50px)' },
        },
        auroraFlow: {
          '0%, 100%': { transform: 'rotate(0deg) scale(1)', opacity: '0.6' },
          '33%': { transform: 'rotate(120deg) scale(1.1)', opacity: '0.8' },
          '66%': { transform: 'rotate(240deg) scale(0.9)', opacity: '0.7' },
        },
        beamSweep: {
          '0%, 100%': { transform: 'translateX(-100px) scaleY(0.5)', opacity: '0' },
          '50%': { transform: 'translateX(100px) scaleY(1)', opacity: '0.8' },
        },
        trailFade: {
          '0%': { transform: 'scale(1)', opacity: '0.8' },
          '100%': { transform: 'scale(0)', opacity: '0' },
        },
        gradientShift: {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        glowPulse: {
          '0%': { textShadow: '0 0 20px rgba(242, 191, 164, 0.5), 0 0 40px rgba(242, 191, 164, 0.3), 0 0 60px rgba(242, 191, 164, 0.1)' },
          '100%': { textShadow: '0 0 30px rgba(242, 191, 164, 0.7), 0 0 50px rgba(242, 191, 164, 0.5), 0 0 80px rgba(242, 191, 164, 0.3)' },
        },
        particleFloat: {
          '0%, 100%': { transform: 'translate(0, 0) rotate(0deg)' },
          '33%': { transform: 'translate(30px, -30px) rotate(120deg)' },
          '66%': { transform: 'translate(-20px, 20px) rotate(240deg)' },
        },
      },
      backdropBlur: {
        xs: '2px',
      },
      boxShadow: {
        'glass': '0 8px 32px rgba(0, 0, 0, 0.1)',
        'glass-hover': '0 12px 40px rgba(0, 0, 0, 0.15)',
        'button': '0 4px 12px rgba(59, 130, 246, 0.3)',
        'button-hover': '0 6px 20px rgba(59, 130, 246, 0.4)',
      },
    },
  },
  plugins: [],
}
