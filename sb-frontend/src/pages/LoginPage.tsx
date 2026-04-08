import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import type { FormEvent } from 'react'
import { Heart, Eye, EyeOff, Loader2, Mail, Lock, ArrowRight } from 'lucide-react'
import '../styles.css'

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

  async function handleSubmit(e: FormEvent) {
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
      <div className="centered min-h-screen" style={{ background: 'var(--background)' }}>
        <div className="card max-w-sm w-full text-center" style={{ borderRadius: 'var(--border-radius)' }}>
          <div style={{ width: 64, height: 64, background: '#e6ffed', borderRadius: 20, display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: 32 }}>
            📬
          </div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-main)', marginBottom: 8 }}>Check your inbox</h2>
          <p className="text-muted" style={{ fontSize: '1rem', marginBottom: 32 }}>
            We sent a confirmation link to{' '}
            <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>{email}</span>.
            Verify your email then sign in.
          </p>
          <button
            onClick={() => { setMode('login'); setSignupDone(false) }}
            className="btn-primary w-full"
          >
            Go to Sign In
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="centered min-h-screen" style={{ background: 'var(--background)' }}>
      {/* Decorative background orbs removed for simplicity and performance */}

      {/* Branding */}
      <div className="centered mb-8">
        <div style={{ width: 48, height: 48, background: 'var(--primary)', borderRadius: 16, display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: 'var(--shadow)', marginBottom: 12 }}>
          <Heart className="w-6 h-6" style={{ color: '#fff' }} fill="currentColor" />
        </div>
        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-main)' }}>SoulBuddy</h1>
        <p className="text-muted" style={{ marginTop: 4 }}>Your mental wellness companion</p>
      </div>

      <div className="card" style={{ maxWidth: 400, width: '100%', borderRadius: 'var(--border-radius)' }}>
        <div className="mb-7">
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-main)' }}>
            {mode === 'login' ? 'Welcome back' : 'Create account'}
          </h2>
          <p className="text-muted" style={{ marginTop: 4 }}>
            {mode === 'login'
              ? 'Sign in to continue your journey'
              : 'Start your wellness journey today'}
          </p>
        </div>
        <button
          type="button"
          onClick={handleGoogleSignIn}
          disabled={loading || googleLoading}
          className="btn-primary w-full"
          style={{ background: '#fff', color: 'var(--text-main)', border: '1px solid #e0e0e0', marginBottom: 16 }}
        >
          {googleLoading ? <Spinner /> : <GoogleIcon />}
          Continue with Google
        </button>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, margin: '20px 0' }}>
          <div style={{ flex: 1, height: 1, background: '#eee' }} />
          <span className="text-muted" style={{ fontSize: 12 }}>or continue with email</span>
          <div style={{ flex: 1, height: 1, background: '#eee' }} />
        </div>
        <div style={{ display: 'flex', background: '#f5f6fa', borderRadius: 16, padding: 4, marginBottom: 20, border: '1px solid #eee' }}>
          {(['login', 'signup'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setError(null) }}
              className={mode === m ? 'btn-primary' : 'text-muted'}
              style={{
                flex: 1,
                borderRadius: 12,
                background: mode === m ? 'var(--primary)' : 'transparent',
                color: mode === m ? '#fff' : 'var(--text-muted)',
                fontWeight: 600,
                fontSize: 14,
                padding: '10px 0',
                boxShadow: mode === m ? 'var(--shadow)' : 'none',
                border: 'none',
                transition: 'all 0.2s',
              }}
            >
              {m === 'login' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ textAlign: 'left' }}>
            <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)', marginBottom: 6, display: 'block' }}>
              Email address
            </label>
            <div style={{ position: 'relative' }}>
              <Mail style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 16, height: 16, color: 'var(--text-muted)', pointerEvents: 'none' }} />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                style={{
                  width: '100%',
                  padding: '12px 16px 12px 40px',
                  borderRadius: 16,
                  border: '1px solid #e0e0e0',
                  background: '#fff',
                  fontSize: 15,
                  marginBottom: 0,
                  boxSizing: 'border-box',
                }}
              />
            </div>
          </div>
          <div style={{ textAlign: 'left' }}>
            <label style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-main)', marginBottom: 6, display: 'block' }}>
              Password
            </label>
            <div style={{ position: 'relative' }}>
              <Lock style={{ position: 'absolute', left: 14, top: '50%', transform: 'translateY(-50%)', width: 16, height: 16, color: 'var(--text-muted)', pointerEvents: 'none' }} />
              <input
                type={showPassword ? 'text' : 'password'}
                required
                minLength={6}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Min. 6 characters"
                style={{
                  width: '100%',
                  padding: '12px 40px 12px 40px',
                  borderRadius: 16,
                  border: '1px solid #e0e0e0',
                  background: '#fff',
                  fontSize: 15,
                  marginBottom: 0,
                  boxSizing: 'border-box',
                }}
              />
              <button
                type="button"
                tabIndex={-1}
                onClick={() => setShowPassword((v) => !v)}
                style={{ position: 'absolute', right: 14, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer' }}
              >
                {showPassword ? <EyeOff style={{ width: 16, height: 16 }} /> : <Eye style={{ width: 16, height: 16 }} />}
              </button>
            </div>
          </div>
          {error && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#d32f2f', background: '#fff0f0', border: '1px solid #ffd6d6', padding: '12px 16px', borderRadius: 16, fontSize: 14 }}>
              <span style={{ marginTop: 2 }}>⚠️</span>
              <span>{error}</span>
            </div>
          )}
          <button
            type="submit"
            disabled={loading || googleLoading}
            className="btn-primary w-full"
            style={{ marginTop: 4 }}
          >
            {loading ? (
              <Spinner />
            ) : (
              <>
                {mode === 'login' ? 'Sign In' : 'Create Account'}
                <ArrowRight style={{ width: 16, height: 16, marginLeft: 6 }} />
              </>
            )}
          </button>
        </form>
        <div style={{ marginTop: 20, textAlign: 'center' }}>
          <button
            onClick={() => navigate('/chat')}
            style={{ fontSize: 13, color: 'var(--text-muted)', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
          >
            Continue without an account (Incognito)
          </button>
        </div>
        <p className="text-muted" style={{ textAlign: 'center', fontSize: 12, marginTop: 20 }}>
          By continuing, you agree to our{' '}
          <span style={{ textDecoration: 'underline', cursor: 'pointer' }}>Terms</span>
          {' & '}
          <span style={{ textDecoration: 'underline', cursor: 'pointer' }}>Privacy Policy</span>
        </p>
      </div>
    </div>
  )
}
