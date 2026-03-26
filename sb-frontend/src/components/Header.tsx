import type { ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { useBuddy, PERSONALITY_OPTIONS } from '../contexts/BuddyContext'
import type { ChatPreference } from '../types'
import { LogOut, Settings, Heart, Ghost } from 'lucide-react'

interface HeaderProps {
  buddyName: string
  isIncognito: boolean
  personality?: ChatPreference
  leftSlot?: ReactNode
}

export default function Header({ buddyName, isIncognito, personality, leftSlot }: HeaderProps) {
  const { signOut, user } = useAuth()
  const { clearConfig } = useBuddy()
  const navigate = useNavigate()

  const personalityLabel = personality
    ? PERSONALITY_OPTIONS.find((o) => o.value === personality)?.emoji ?? ''
    : ''

  async function handleSignOut() {
    clearConfig()
    await signOut()
    navigate('/login')
  }

  return (
    <header className="h-14 border-b border-gray-100 bg-white flex items-center px-4 gap-3 shrink-0">
      {leftSlot}

      <div className="flex items-center gap-2 flex-1">
        <div className="w-8 h-8 bg-buddy-100 rounded-xl flex items-center justify-center">
          <Heart className="w-4 h-4 text-buddy-600" fill="currentColor" />
        </div>
        <div>
          <span className="font-semibold text-gray-800 text-sm">{buddyName}</span>
          {personality && (
            <span className="ml-2 text-xs text-gray-400">{personalityLabel}</span>
          )}
        </div>
      </div>

      {isIncognito ? (
        <div className="flex items-center gap-1.5 text-xs text-gray-400 bg-gray-100 px-3 py-1.5 rounded-full">
          <Ghost className="w-3.5 h-3.5" />
          Incognito
        </div>
      ) : null}

      {user && (
        <div className="flex items-center gap-1">
          <button
            onClick={() => navigate('/onboarding')}
            className="p-2 rounded-lg hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
            title="Change buddy settings"
          >
            <Settings className="w-4 h-4" />
          </button>
          <button
            onClick={handleSignOut}
            className="p-2 rounded-lg hover:bg-red-50 text-gray-400 hover:text-red-500 transition-colors"
            title="Sign out"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      )}

      {!user && (
        <button
          onClick={() => navigate('/login')}
          className="text-xs text-buddy-600 font-semibold hover:underline"
        >
          Sign In
        </button>
      )}
    </header>
  )
}
