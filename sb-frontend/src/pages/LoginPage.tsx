import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { Heart } from 'lucide-react'

type Mode = 'login' | 'signup'

export default function LoginPage() {
  const { signIn, signUp } = useAuth()
  const navigate = useNavigate()
  const [mode, setMode] = useState<Mode>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [signupDone, setSignupDone] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      if (mode === 'login') {
        const err = await signIn(email, password)
        if (err) {
          setError(err)
        } else {
          navigate('/onboarding')
        }
      } else {
        const err = await signUp(email, password)
        if (err) {
          setError(err)
        } else {
          setSignupDone(true)
        }
      }
    } finally {
      setLoading(false)
    }
  }

  if (signupDone) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-buddy-50 to-indigo-50">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-sm w-full text-center">
          <div className="text-5xl mb-4">📬</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Check your inbox</h2>
          <p className="text-gray-500 mb-6">
            We've sent a confirmation link to <strong>{email}</strong>. Verify your email then sign in.
          </p>
          <button
            onClick={() => { setMode('login'); setSignupDone(false) }}
            className="w-full py-3 bg-buddy-600 text-white rounded-xl font-semibold hover:bg-buddy-700 transition-colors"
          >
            Go to Sign In
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-buddy-50 to-indigo-50 px-4">
      <div className="bg-white rounded-2xl shadow-lg p-8 max-w-sm w-full">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="w-14 h-14 bg-buddy-100 rounded-2xl flex items-center justify-center mb-3">
            <Heart className="w-7 h-7 text-buddy-600" fill="currentColor" />
          </div>
          <h1 className="text-2xl font-bold text-gray-800">SoulBuddy</h1>
          <p className="text-gray-400 text-sm mt-1">Your mental wellness companion</p>
        </div>

        {/* Tab switcher */}
        <div className="flex rounded-xl bg-gray-100 p-1 mb-6">
          {(['login', 'signup'] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setError(null) }}
              className={`flex-1 py-2 text-sm font-semibold rounded-lg transition-all ${
                mode === m
                  ? 'bg-white text-buddy-700 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {m === 'login' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-buddy-400 text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <input
              type="password"
              required
              minLength={6}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-buddy-400 text-sm"
            />
          </div>

          {error && (
            <p className="text-red-500 text-sm bg-red-50 px-3 py-2 rounded-lg">{error}</p>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-buddy-600 text-white rounded-xl font-semibold hover:bg-buddy-700 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Please wait…' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => navigate('/chat')}
            className="text-sm text-gray-400 hover:text-gray-600 underline underline-offset-2 transition-colors"
          >
            Continue without an account (Incognito)
          </button>
        </div>
      </div>
    </div>
  )
}
