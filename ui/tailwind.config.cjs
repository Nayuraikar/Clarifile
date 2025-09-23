/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './index.html',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        sand50: '#F5E7DE',
        sand200: '#F2BFA4',
        charcoal900: '#1E1A17',
        mink500: '#9A8F87',
        pearl400: '#C4B5AC',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Roboto', 'Arial', 'sans-serif'],
      },
    },
  },
  plugins: [],
};

