import { createContext, useContext, useState, useEffect } from 'react'
import type { ReactNode } from 'react'
import type { BuddyConfig, ChatPreference, Domain } from '../types'

const STORAGE_KEY = 'soulbuddy_config'

interface BuddyContextValue {
  config: BuddyConfig | null
  saveConfig: (c: BuddyConfig) => void
  clearConfig: () => void
  isConfigured: boolean
}

const BuddyContext = createContext<BuddyContextValue | null>(null)

function loadFromStorage(): BuddyConfig | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<BuddyConfig>
    if (parsed.name && parsed.personality && parsed.domain) {
      return parsed as BuddyConfig
    }
    return null
  } catch {
    return null
  }
}

export function BuddyProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<BuddyConfig | null>(loadFromStorage)

  useEffect(() => {
    if (config) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(config))
    } else {
      localStorage.removeItem(STORAGE_KEY)
    }
  }, [config])

  function saveConfig(c: BuddyConfig) {
    setConfig(c)
  }

  function clearConfig() {
    setConfig(null)
  }

  return (
    <BuddyContext.Provider
      value={{ config, saveConfig, clearConfig, isConfigured: config !== null }}
    >
      {children}
    </BuddyContext.Provider>
  )
}

export function useBuddy(): BuddyContextValue {
  const ctx = useContext(BuddyContext)
  if (!ctx) throw new Error('useBuddy must be used within BuddyProvider')
  return ctx
}

export const PERSONALITY_OPTIONS: { value: ChatPreference; label: string; description: string; emoji: string }[] = [
  {
    value: 'gentle_reflective',
    label: 'Gentle & Reflective',
    description: 'Warm, empathetic and thoughtful. Takes time to understand your feelings.',
    emoji: '🌱',
  },
  {
    value: 'direct_practical',
    label: 'Direct & Practical',
    description: 'Honest, solution-focused and straightforward. Helps you take action.',
    emoji: '⚡',
  },
  {
    value: 'general',
    label: 'Balanced',
    description: 'A flexible mix of empathy and practicality, adapts to what you need.',
    emoji: '☯️',
  },
]

export const DOMAIN_OPTIONS: { value: Domain; label: string; emoji: string }[] = [
  { value: 'student', label: 'Student', emoji: '📚' },
  { value: 'employee', label: 'Working Professional', emoji: '💼' },
  { value: 'corporate', label: 'Corporate Leader', emoji: '🏢' },
  { value: 'general', label: 'General', emoji: '🌍' },
]
