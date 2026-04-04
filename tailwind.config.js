module.exports = {
  darkMode: 'class',
  content: [
    './src/cyber_inference/web/templates/**/*.html',
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          black: '#0a0a0f',
          darker: '#0d0d14',
          dark: '#12121a',
          gray: '#1a1a24',
          light: '#2a2a38',
          neon: '#00ff9f',
          neonDark: '#00cc7f',
          neonGlow: '#00ff9f40',
          pink: '#ff00ff',
          blue: '#00d4ff',
          orange: '#ff6600',
          red: '#ff3333',
          yellow: '#ffff00',
        },
      },
      fontFamily: {
        cyber: ['Orbitron', 'sans-serif'],
        mono: ['Share Tech Mono', 'monospace'],
        display: ['Rajdhani', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        glow: 'glow 2s ease-in-out infinite alternate',
        scan: 'scan 8s linear infinite',
        flicker: 'flicker 0.15s infinite',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #00ff9f, 0 0 10px #00ff9f, 0 0 15px #00ff9f' },
          '100%': { boxShadow: '0 0 10px #00ff9f, 0 0 20px #00ff9f, 0 0 30px #00ff9f' },
        },
        scan: {
          '0%': { backgroundPosition: '0 -100vh' },
          '100%': { backgroundPosition: '0 100vh' },
        },
        flicker: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
      },
    },
  },
}
