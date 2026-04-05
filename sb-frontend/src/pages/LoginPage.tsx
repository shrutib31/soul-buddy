import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Heart, Eye, EyeOff, Loader2, Mail, Lock, ArrowRight, Sparkles } from 'lucide-react'

type Mode = 'login' | 'signup'

function GoogleIcon() {
  return (
    <svg width="20" height="20" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg" className="shrink-0">
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615Z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18Z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332Z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58Z" fill="#EA4335"/>
    </svg>
  )
}

function Spinner() {
  return <Loader2 className="w-5 h-5 animate-spin" />
}

export default function LoginPage() {
  const { signIn, signUp, signInWithGoogle } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [googleLoading, setGoogleLoading] = useState(false)
  const [signupDone, setSignupDone] = useState(false)

  async function handleGoogleSignIn() {
    setError(null)
    setGoogleLoading(true)
    try {
      const err = await signInWithGoogle()
      if (err) setError(err)
    } finally {
      setGoogleLoading(false)
    }
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === 'login') {
        const err = await signIn(email, password)
        if (err) setError(err)
        else navigate('/onboarding')
      } else {
        const err = await signUp(email, password)
        if (err) setError(err)
        else setSignupDone(true)
      }
    } finally {
      setLoading(false)
    }
  }

  if (signupDone) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-buddy-50 via-indigo-50 to-purple-50 px-4">
        <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl shadow-buddy-100 border border-white p-10 max-w-sm w-full text-center">
          <div className="w-16 h-16 bg-green-50 rounded-2xl flex items-center justify-center mx-auto mb-5 text-3xl">
            📬
          </div>
          <h2 className="text-2xl font-bold text-gray-900 mb-2">Check your inbox</h2>
          <p className="text-gray-500 text-sm leading-relaxed mb-8">
            We sent a confirmation link to{' '}
            <span className="font-semibold text-gray-700">{email}</span>.
            Verify your email then sign in.
          </p>
          <button
            onClick={() => { setMode('login'); setSignupDone(false) }}
            className="w-full py-3 bg-buddy-600 text-white rounded-2xl font-semibold hover:bg-buddy-700 active:scale-[0.98] transition-all"
          >
            Go to Sign In
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex bg-gradient-to-br from-buddy-50 via-indigo-50 to-purple-50 relative overflow-hidden">
      {/* Decorative background orbs */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-buddy-200 rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-[pulse_8s_ease-in-out_infinite]" />
        <div className="absolute -bottom-40 -right-40 w-96 h-96 bg-indigo-200 rounded-full mix-blend-multiply filter blur-3xl opacity-40 animate-[pulse_10s_ease-in-out_infinite_2s]" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-purple-100 rounded-full mix-blend-multiply filter blur-3xl opacity-30 animate-[pulse_12s_ease-in-out_infinite_4s]" />
      </div>

      {/* Left panel — branding (hidden on small screens) */}
      <div className="hidden lg:flex flex-col justify-between w-[420px] shrink-0 p-12 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-buddy-600 rounded-xl flex items-center justify-center shadow-lg shadow-buddy-300">
            <Heart className="w-5 h-5 text-white" fill="currentColor" />
          </div>
          <span className="text-xl font-bold text-gray-900">SoulBuddy</span>
        </div>

        <div>
          <div className="mb-8">
            <div className="inline-flex items-center gap-2 bg-buddy-100 text-buddy-700 text-xs font-semibold px-3 py-1.5 rounded-full mb-6">
              <Sparkles className="w-3.5 h-3.5" />
              AI-powered wellness
            </div>
            <h2 className="text-4xl font-bold text-gray-900 leading-tight mb-4">
              Your personal<br />
              <span className="text-buddy-600">mental wellness</span><br />
              companion
            </h2>
            <p className="text-gray-500 text-base leading-relaxed">
              Talk freely. Get support. Feel better — with a companion that listens, understands, and grows with you.
            </p>
          </div>

          <div className="space-y-3">
            {[
              { emoji: '🧠', text: 'Intelligent, empathetic conversations' },
              { emoji: '🔒', text: 'Private & secure by design' },
              { emoji: '✨', text: 'Personalised to your journey' },
            ].map(({ emoji, text }) => (
              <div key={text} className="flex items-center gap-3 text-sm text-gray-600">
                <span className="text-base">{emoji}</span>
                {text}
              </div>
            ))}
          </div>
        </div>

        <p className="text-xs text-gray-400">© 2025 SoulBuddy. All rights reserved.</p>
      </div>

      {/* Right panel — auth form */}
      <div className="flex-1 flex items-center justify-center px-4 py-12 relative z-10">
        <div className="w-full max-w-[400px]">
          {/* Mobile logo */}
          <div className="flex flex-col items-center mb-8 lg:hidden">
            <div className="w-14 h-14 bg-buddy-600 rounded-2xl flex items-center justify-center shadow-lg shadow-buddy-300 mb-3">
              <Heart className="w-7 h-7 text-white" fill="currentColor" />
            </div>
            <h1 className="text-2xl font-bold text-gray-900">SoulBuddy</h1>
            <p className="text-gray-400 text-sm mt-1">Your mental wellness companion</p>
          </div>

          <div className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl shadow-gray-200/60 border border-white/60 p-8">
            {/* Heading */}
            <div className="mb-7">
              <h2 className="text-2xl font-bold text-gray-900">
                {mode === 'login' ? 'Welcome back' : 'Create account'}
              </h2>
              <p className="text-gray-500 text-sm mt-1">
                {mode === 'login'
                  ? 'Sign in to continue your journey'
                  : 'Start your wellness journey today'}
              </p>
            </div>

            {/* Google — primary CTA */}
            <button
              type="button"
              onClick={handleGoogleSignIn}
              disabled={loading || googleLoading}
              className="w-full flex items-center justify-center gap-3 py-3 bg-white border-2 border-gray-100 rounded-2xl text-sm font-semibold text-gray-700 hover:border-gray-200 hover:shadow-md hover:shadow-gray-100 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
            >
              {googleLoading ? <Spinner /> : <GoogleIcon />}
              Continue with Google
            </button>

            {/* Divider */}
            <div className="flex items-center gap-3 my-5">
              <div className="flex-1 h-px bg-gray-100" />
              <span className="text-xs font-medium text-gray-400">or continue with email</span>
              <div className="flex-1 h-px bg-gray-100" />
            </div>

            {/* Tab switcher */}
            <div className="flex rounded-2xl bg-gray-50 p-1 mb-5 border border-gray-100">
              {(['login', 'signup'] as Mode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => { setMode(m); setError(null) }}
                  className={`flex-1 py-2 text-sm font-semibold rounded-xl transition-all duration-200 ${
                    mode === m
                      ? 'bg-white text-buddy-700 shadow-sm shadow-gray-200'
                      : 'text-gray-400 hover:text-gray-600'
                  }`}
                >
                  {m === 'login' ? 'Sign In' : 'Sign Up'}
                </button>
              ))}
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="relative">
                <label className="block text-xs font-semibold text-gray-600 mb-1.5 ml-1">
                  Email address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    className="w-full pl-10 pr-4 py-3 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-buddy-400 focus:border-transparent text-sm transition-all placeholder:text-gray-300"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-600 mb-1.5 ml-1">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    required
                    minLength={6}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Min. 6 characters"
                    className="w-full pl-10 pr-11 py-3 rounded-xl border border-gray-200 bg-gray-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-buddy-400 focus:border-transparent text-sm transition-all placeholder:text-gray-300"
                  />
                  <button
                    type="button"
                    tabIndex={-1}
                    onClick={() => setShowPassword((v) => !v)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              {error && (
                <div className="flex items-start gap-2 text-red-600 bg-red-50 border border-red-100 px-3.5 py-2.5 rounded-xl text-sm">
                  <span className="mt-0.5 shrink-0">⚠️</span>
                  <span>{error}</span>
                </div>
              )}

              <button
                type="submit"
                disabled={loading || googleLoading}
                className="w-full flex items-center justify-center gap-2 py-3 bg-buddy-600 text-white rounded-2xl font-semibold hover:bg-buddy-700 active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-buddy-200 mt-1"
              >
                {loading ? (
                  <Spinner />
                ) : (
                  <>
                    {mode === 'login' ? 'Sign In' : 'Create Account'}
                    <ArrowRight className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>

            {/* Incognito link */}
            <div className="mt-5 text-center">
              <button
                onClick={() => navigate('/chat')}
                className="text-xs text-gray-400 hover:text-gray-600 transition-colors"
              >
                Continue without an account (Incognito)
              </button>
            </div>
          </div>

          <p className="text-center text-xs text-gray-400 mt-5">
            By continuing, you agree to our{' '}
            <span className="underline underline-offset-2 cursor-pointer hover:text-gray-600 transition-colors">Terms</span>
            {' & '}
            <span className="underline underline-offset-2 cursor-pointer hover:text-gray-600 transition-colors">Privacy Policy</span>
          </p>
        </div>
      </div>
    </div>
  )
}
