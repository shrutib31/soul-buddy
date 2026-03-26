/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        buddy: {
          50: '#f0f4ff',
          100: '#dce6ff',
          200: '#b9ccff',
          300: '#8aaaff',
          400: '#577dff',
          500: '#3355ff',
          600: '#1a33f5',
          700: '#1326e1',
          800: '#1620b6',
          900: '#18228f',
        },
      },
    },
  },
  plugins: [],
}
