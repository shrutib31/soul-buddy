import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useBuddy, PERSONALITY_OPTIONS, DOMAIN_OPTIONS } from '../contexts/BuddyContext'
import type { ChatPreference, Domain } from '../types'
import { ArrowRight, ArrowLeft } from 'lucide-react'

type Step = 'personality' | 'domain' | 'name'

export default function OnboardingPage() {
  const { saveConfig } = useBuddy()
  const navigate = useNavigate()
  const [step, setStep] = useState<Step>('personality')
  const [personality, setPersonality] = useState<ChatPreference | null>(null)
  const [domain, setDomain] = useState<Domain | null>(null)
  const [buddyName, setBuddyName] = useState('')

  function handleFinish() {
    if (!personality || !domain || !buddyName.trim()) return
    saveConfig({ name: buddyName.trim(), personality, domain })
    navigate('/chat')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-buddy-50 to-indigo-50 flex items-center justify-center px-4 py-12">
      <div className="bg-white rounded-2xl shadow-lg p-8 max-w-lg w-full">

        {/* Progress */}
        <div className="flex gap-2 mb-8">
          {(['personality', 'domain', 'name'] as Step[]).map((s, i) => (
            <div
              key={s}
              className={`h-1.5 flex-1 rounded-full transition-colors ${
                ['personality', 'domain', 'name'].indexOf(step) >= i
                  ? 'bg-buddy-500'
                  : 'bg-gray-200'
              }`}
            />
          ))}
        </div>

        {/* Step: Personality */}
        {step === 'personality' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-800 mb-1">Choose your buddy's style</h2>
            <p className="text-gray-500 text-sm mb-6">How would you like your buddy to communicate with you?</p>
            <div className="space-y-3">
              {PERSONALITY_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setPersonality(opt.value)}
                  className={`w-full text-left p-4 rounded-xl border-2 transition-all ${
                    personality === opt.value
                      ? 'border-buddy-500 bg-buddy-50'
                      : 'border-gray-100 hover:border-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{opt.emoji}</span>
                    <div>
                      <div className="font-semibold text-gray-800">{opt.label}</div>
                      <div className="text-xs text-gray-500 mt-0.5">{opt.description}</div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
            <button
              disabled={!personality}
              onClick={() => setStep('domain')}
              className="mt-6 w-full py-3 bg-buddy-600 text-white rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-buddy-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Next <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        )}

        {/* Step: Domain */}
        {step === 'domain' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-800 mb-1">What describes you best?</h2>
            <p className="text-gray-500 text-sm mb-6">This helps personalise support for your context.</p>
            <div className="grid grid-cols-2 gap-3">
              {DOMAIN_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setDomain(opt.value)}
                  className={`p-4 rounded-xl border-2 text-center transition-all ${
                    domain === opt.value
                      ? 'border-buddy-500 bg-buddy-50'
                      : 'border-gray-100 hover:border-gray-200'
                  }`}
                >
                  <div className="text-3xl mb-2">{opt.emoji}</div>
                  <div className="text-sm font-semibold text-gray-700">{opt.label}</div>
                </button>
              ))}
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => setStep('personality')}
                className="flex-1 py-3 border-2 border-gray-200 text-gray-600 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-gray-50 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" /> Back
              </button>
              <button
                disabled={!domain}
                onClick={() => setStep('name')}
                className="flex-1 py-3 bg-buddy-600 text-white rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-buddy-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Next <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Step: Name */}
        {step === 'name' && (
          <div>
            <h2 className="text-2xl font-bold text-gray-800 mb-1">Name your buddy</h2>
            <p className="text-gray-500 text-sm mb-6">Give your buddy a name you'll feel comfortable with.</p>

            <div className="bg-gray-50 rounded-xl p-4 mb-6 text-sm text-gray-600">
              <span className="font-medium">Your buddy:</span>{' '}
              {personality && PERSONALITY_OPTIONS.find((o) => o.value === personality)?.emoji}{' '}
              {personality && PERSONALITY_OPTIONS.find((o) => o.value === personality)?.label} ·{' '}
              {domain && DOMAIN_OPTIONS.find((o) => o.value === domain)?.label}
            </div>

            <input
              type="text"
              value={buddyName}
              onChange={(e) => setBuddyName(e.target.value)}
              placeholder="e.g. Luna, Max, Aria…"
              maxLength={30}
              className="w-full px-4 py-3 rounded-xl border border-gray-200 focus:outline-none focus:ring-2 focus:ring-buddy-400 text-sm mb-2"
            />
            <p className="text-xs text-gray-400 text-right">{buddyName.length}/30</p>

            <div className="flex gap-3 mt-4">
              <button
                onClick={() => setStep('domain')}
                className="flex-1 py-3 border-2 border-gray-200 text-gray-600 rounded-xl font-semibold flex items-center justify-center gap-2 hover:bg-gray-50 transition-colors"
              >
                <ArrowLeft className="w-4 h-4" /> Back
              </button>
              <button
                disabled={!buddyName.trim()}
                onClick={handleFinish}
                className="flex-1 py-3 bg-buddy-600 text-white rounded-xl font-semibold hover:bg-buddy-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Start chatting 🎉
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
